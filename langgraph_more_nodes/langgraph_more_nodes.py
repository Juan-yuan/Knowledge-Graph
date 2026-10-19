import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from common.config import Config
from langgraph_more_nodes.agent_state import AgentState
from langgraph_more_nodes.nodes.check_cypher_node import check_cypher_node
from langgraph_more_nodes.nodes.extract_entity_from_user_input_node import extract_entity_from_user_input_node
from langgraph_more_nodes.nodes.generate_neo4j_cypher_node import generate_neo4j_cypher_node
from langgraph_more_nodes.nodes.llm_direct_out_node import llm_direct_out_node
from langgraph_more_nodes.nodes.match_entity_from_neo4j_node import match_entity_from_neo4j_node
from langgraph_more_nodes.nodes.neo4j_answer_generate_node import neo4j_answer_generate_node
from langgraph_more_nodes.nodes.run_cypher_node import run_cypher_node
from langgraph_more_nodes.nodes.zhongyi_intent_node import zhongyi_intent_node

conf = Config()
# Max Cypher checks before giving up on regenerating invalid queries
MAX_CYPHER_CHECKS = int(conf.require("MAX_CYPHER_CHECKS"))


def route_after_intent(state: AgentState) -> str:
    """TCM questions go down the knowledge-graph path; everything else is answered directly."""
    return "extract_entity_from_user_input" if state["is_zhongyi_intent"] else "llm_direct_out"


def route_after_check(state: AgentState) -> str:
    """Valid Cypher runs; invalid Cypher is regenerated until the retry limit is hit."""
    if state["is_all_validate_cypher"]:
        return "run_cypher"
    # No queries means no matched entities, so regenerating would produce nothing new
    if state.get("cypher_query") and state["cypher_check_count"] < MAX_CYPHER_CHECKS:
        return "generate_neo4j_cypher"
    return "neo4j_answer_generate"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("zhongyi_intent", zhongyi_intent_node)
    builder.add_node("extract_entity_from_user_input", extract_entity_from_user_input_node)
    builder.add_node("match_entity_from_neo4j", match_entity_from_neo4j_node)
    builder.add_node("generate_neo4j_cypher", generate_neo4j_cypher_node)
    builder.add_node("check_cypher", check_cypher_node)
    builder.add_node("run_cypher", run_cypher_node)
    builder.add_node("neo4j_answer_generate", neo4j_answer_generate_node)
    builder.add_node("llm_direct_out", llm_direct_out_node)

    builder.add_edge(START, "zhongyi_intent")
    builder.add_conditional_edges(
        "zhongyi_intent",
        route_after_intent,
        ["extract_entity_from_user_input", "llm_direct_out"],
    )
    builder.add_edge("extract_entity_from_user_input", "match_entity_from_neo4j")
    builder.add_edge("match_entity_from_neo4j", "generate_neo4j_cypher")
    builder.add_edge("generate_neo4j_cypher", "check_cypher")
    builder.add_conditional_edges(
        "check_cypher",
        route_after_check,
        ["run_cypher", "generate_neo4j_cypher", "neo4j_answer_generate"],
    )
    builder.add_edge("run_cypher", "neo4j_answer_generate")
    builder.add_edge("neo4j_answer_generate", END)
    builder.add_edge("llm_direct_out", END)

    return builder.compile()


graph = build_graph()


def main():
    print(graph.get_graph().draw_mermaid())

    for text in [
        "What are the effects of Ephedra (Ma Huang)?",
        "What herbs make up Ephedra Decoction (Ma Huang Tang)?",
        "What Chinese herbs can help with a cold with chills and no sweating?",
        "How is the weather today?",
        "Write a Python quicksort for me",
    ]:
        print("=" * 80)
        print(f"Input: {text}")
        # stream_mode="updates" yields {node_name: state} after each node runs
        result = {}
        for update in graph.stream({"input": text}, stream_mode="updates"):
            for node_name, node_state in update.items():
                print(f"  -> {node_name}")
                result = node_state
        print(f"  is_zhongyi_intent: {result.get('is_zhongyi_intent')}")
        if result.get("is_zhongyi_intent"):
            print(f"  matched_herbs: {result.get('matched_herbs')}")
            print(f"  matched_formulas: {result.get('matched_formulas')}")
            print(f"  cypher_query: {result.get('cypher_query')}")
            print(f"  is_all_validate_cypher: {result.get('is_all_validate_cypher')}")
        print(f"Output:\n{result.get('output')}")


if __name__ == "__main__":
    main()
