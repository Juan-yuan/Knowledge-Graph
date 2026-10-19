import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import SystemMessage, HumanMessage

from common.config import Config
from common.llm import my_llm
from langgraph_more_nodes.agent_state import AgentState

llm = my_llm
conf = Config()
NEO4J_ANSWER_PROMPT = conf.require("NEO4J_ANSWER_PROMPT")


def neo4j_answer_generate_node(state: AgentState) -> AgentState:
    """Answer the user input from the Cypher results; write it to neo4j_answer and output."""
    cypher_results = state.get("cypher_results") or []
    # default=str keeps Neo4j types (dates, nodes, etc.) serializable
    results_text = json.dumps(cypher_results, ensure_ascii=False, indent=2, default=str)
    messages = [
        SystemMessage(content=NEO4J_ANSWER_PROMPT),
        HumanMessage(content=f"User question:\n{state['input']}\n\nNeo4j query results:\n{results_text}"),
    ]

    answer = llm.invoke(messages).content.strip()

    state["neo4j_answer"] = answer
    state["output"] = answer
    return state


if __name__ == "__main__":
    for test_state in [
        {
            "input": "What are the effects of Ginseng?",
            "cypher_results": [
                [
                    {"herb": "Ginseng (Ren Shen)", "effect": "Greatly tonifies primordial qi"},
                    {"herb": "Ginseng (Ren Shen)", "effect": "Tonifies the spleen and lungs"},
                    {"herb": "Ginseng (Ren Shen)", "effect": "Generates fluids and calms the spirit"},
                ],
            ],
        },
        {
            "input": "What herbs make up Ephedra Decoction (Ma Huang Tang)?",
            "cypher_results": [
                [
                    {"formula": "Ephedra Decoction (Ma Huang Tang)", "herb": "Ephedra (Ma Huang)"},
                    {"formula": "Ephedra Decoction (Ma Huang Tang)", "herb": "Cinnamon twig (Gui Zhi)"},
                    {"formula": "Ephedra Decoction (Ma Huang Tang)", "herb": "Apricot kernel (Xing Ren)"},
                    {"formula": "Ephedra Decoction (Ma Huang Tang)", "herb": "Licorice (Gan Cao)"},
                ],
                [],
            ],
        },
        {
            "input": "What Chinese herbs can help with insomnia?",
            "cypher_results": [[]],
        },
        {
            "input": "What Chinese herbs can help with insomnia?",
        },
    ]:
        result = neo4j_answer_generate_node(test_state)
        print(test_state["input"])
        print(f"  neo4j_answer: {result['neo4j_answer']}")
        print(f"  output == neo4j_answer: {result['output'] == result['neo4j_answer']}")
        print()
