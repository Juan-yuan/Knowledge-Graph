import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from common.config import Config
from common.llm import my_llm
from langgraph_more_nodes.agent_state import AgentState

conf = Config()
CYPHER_GENERATION_PROMPT = conf.require("CYPHER_GENERATION_PROMPT")
TCM_METADATA = conf.TCM_METADATA

# matched_* field -> Neo4j node label
MATCHED_KEY_TO_LABEL = {
    "matched_effects": "Effect",
    "matched_diseases": "Disease",
    "matched_symptoms": "Symptom",
    "matched_formulas": "Formula",
    "matched_herbs": "Herb",
    "matched_sources": "Source",
}

# Read-only guard: drop any generated query containing a write clause
WRITE_KEYWORDS = ("CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "LOAD CSV", "CALL")


class CypherQueries(BaseModel):
    """Read-only Cypher queries that retrieve the graph data needed to answer the user question."""

    queries: List[str] = Field(
        default_factory=list,
        description="Standalone read-only Cypher statements, one per distinct piece of information needed.",
    )


# function_calling is the most widely supported mode across OpenAI-compatible providers
llm = my_llm.with_structured_output(CypherQueries, method="function_calling")


def _is_read_only(query: str) -> bool:
    tokens = f" {' '.join(query.upper().split())} "
    return not any(f" {keyword} " in tokens for keyword in WRITE_KEYWORDS)


def _clean_queries(queries: List[str]) -> List[str]:
    result = []
    for query in queries:
        query = query.strip().strip("`").strip().rstrip(";").strip()
        if query.lower().startswith("cypher"):
            query = query[len("cypher"):].strip()
        if query and _is_read_only(query) and query not in result:
            result.append(query)
    return result


def generate_neo4j_cypher_node(state: AgentState) -> AgentState:
    """Generate Cypher queries from the TCM metadata, user input and matched entities."""
    matched_entities = {
        label: state.get(key) or []
        for key, label in MATCHED_KEY_TO_LABEL.items()
    }
    if not any(matched_entities.values()):
        state["cypher_query"] = []
        return state

    entity_lines = "\n".join(
        f"- {label}: {json.dumps(names, ensure_ascii=False)}"
        for label, names in matched_entities.items()
        if names
    )
    messages = [
        SystemMessage(content=f"{CYPHER_GENERATION_PROMPT}\n\nGraph metadata:\n{TCM_METADATA}"),
        HumanMessage(content=f"User question:\n{state['input']}\n\nMatched entities (Neo4j node names):\n{entity_lines}"),
    ]

    result = llm.invoke(messages) or CypherQueries()

    state["cypher_query"] = _clean_queries(result.queries)
    return state


if __name__ == "__main__":
    for test_state in [
        {
            "input": "What are the effects and contraindications of Ginseng?",
            "matched_herbs": ["Ginseng (Ren Shen)"],
        },
        {
            "input": "What Chinese herbs can help with insomnia?",
            "matched_diseases": ["insomnia"],
        },
        {
            "input": "Which formulas from the Shang Han Lun treat fever and aversion to cold caused by exterior wind-cold?",
            "matched_diseases": ["exterior excess due to wind-cold"],
            "matched_symptoms": ["fever", "aversion to cold", "Aversion to cold"],
            "matched_sources": ["Shang Han Lun"],
        },
        {
            "input": "What herbs are in Ephedra Decoction (Ma Huang Tang) and what are its effects?",
            "matched_formulas": ["Ephedra Decoction (Ma Huang Tang)"],
        },
        {
            "input": "How is the weather today?",
        },
    ]:
        result = generate_neo4j_cypher_node(test_state)
        print(test_state["input"])
        for query in result["cypher_query"]:
            print(f"  {query}")
        print()
