import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.neo4j_manager import neo4j_client
from langgraph_more_nodes.agent_state import AgentState


def check_cypher_node(state: AgentState) -> AgentState:
    """Check that every generated Cypher query is valid (EXPLAIN against Neo4j)."""
    queries = state.get("cypher_query") or []
    # No queries means nothing to run, so treat it as not valid
    state["is_all_validate_cypher"] = bool(queries) and all(
        neo4j_client.validate_cypher(query) for query in queries
    )
    return state


if __name__ == "__main__":
    try:
        for test_state in [
            {
                "cypher_query": [
                    "MATCH (h:Herb {name: 'Ginseng (Ren Shen)'}) RETURN h",
                    "MATCH (f:Formula)-[r]->(h:Herb) RETURN f.name, type(r), h.name LIMIT 5",
                ],
            },
            {
                "cypher_query": [
                    "MATCH (h:Herb {name: 'Ginseng (Ren Shen)'}) RETURN h",
                    "MATCH (h:Herb WHERE RETURN h",
                ],
            },
            {
                "cypher_query": [],
            },
        ]:
            result = check_cypher_node(test_state)
            print(f"{result['cypher_query']} -> {result['is_all_validate_cypher']}")
    finally:
        neo4j_client.close()
