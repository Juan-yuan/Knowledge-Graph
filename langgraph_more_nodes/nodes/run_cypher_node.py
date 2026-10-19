import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.neo4j_manager import neo4j_client
from langgraph_more_nodes.agent_state import AgentState


def run_cypher_node(state: AgentState) -> AgentState:
    """Execute each Cypher query against Neo4j; one result list per query."""
    results = []
    for query in state.get("cypher_query") or []:
        try:
            results.append(neo4j_client.run_cypher(query))
        except Exception as e:
            print(f"Cypher execution error: {e}")
            results.append([])
    state["cypher_results"] = results
    return state


if __name__ == "__main__":
    try:
        for test_state in [
            {
                "cypher_query": [
                    "MATCH (h:Herb {name: 'Ginseng (Ren Shen)'}) RETURN h.name AS name LIMIT 1",
                    "MATCH (f:Formula)-[r]->(h:Herb) RETURN f.name AS formula, type(r) AS rel, h.name AS herb LIMIT 5",
                ],
            },
            {
                "cypher_query": ["MATCH (h:Herb WHERE RETURN h"],
            },
            {
                "cypher_query": [],
            },
        ]:
            result = run_cypher_node(test_state)
            for query, records in zip(result["cypher_query"], result["cypher_results"]):
                print(query)
                for record in records:
                    print(f"  {record}")
            print(f"cypher_results: {len(result['cypher_results'])} result list(s)\n")
    finally:
        neo4j_client.close()
