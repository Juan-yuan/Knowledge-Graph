import json
import sys
from pathlib import Path

from tqdm import tqdm
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config

conf = Config()


class Neo4jClient:
    def __init__(self, uri, user, password):
        """Open a connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the connection."""
        driver = getattr(self, "driver", None)
        if driver is not None:
            driver.close()
            self.driver = None

    def __del__(self):
        """Close the connection."""
        self.close()

    def run_cypher(self, query, parameters=None):
        """Execute one Cypher statement and return the results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_multiple_cypher(self, queries_with_params):
        """Execute multiple Cypher statements in one write transaction."""
        progress_desc = conf.require("NEO4J_CYPHER_PROGRESS_DESC")
        with self.driver.session() as session:
            def transaction_logic(tx):
                for query, params in tqdm(queries_with_params, desc=progress_desc):
                    tx.run(query, params or {})

            session.execute_write(transaction_logic)

    def export_tcm_metadata_to_json(self, output_path="tcm_metadata.json"):
        with self.driver.session() as session:

            label_query = """
            MATCH (n)
            WHERE n.project = "TCM"
            UNWIND labels(n) AS label
            RETURN DISTINCT label
            """
            labels = [record["label"] for record in session.run(label_query)]

            rel_query = """
            MATCH (n)-[r]-()
            WHERE n.project = "TCM"
            RETURN DISTINCT type(r) AS rel_type
            """
            rel_types = [record["rel_type"] for record in session.run(rel_query)]

            triple_query = """
            MATCH (n)-[r]->(m)
            WHERE n.project = "TCM"
            WITH head(labels(n)) AS from_label, type(r) AS rel_type, head(labels(m)) AS to_label
            RETURN DISTINCT from_label, rel_type, to_label
            """
            triples = [{
                "from": record["from_label"],
                "rel_type": record["rel_type"],
                "to": record["to_label"],
                "description": ""
            } for record in session.run(triple_query)]

            node_props_query = """
            MATCH (n)
            WHERE n.project = "TCM"
            UNWIND labels(n) AS label
            UNWIND keys(n) AS prop
            RETURN DISTINCT label, prop
            ORDER BY label, prop
            """
            label_props = {}
            for record in session.run(node_props_query):
                label = record["label"]
                prop = record["prop"]
                if prop == "project":  # ignore the project field
                    continue
                label_props.setdefault(label, []).append({
                    "name": prop,
                    "description": ""
                })

            rel_props_query = """
            MATCH (n)-[r]->(m)
            WHERE n.project = "TCM"
            UNWIND keys(r) AS prop
            RETURN DISTINCT type(r) AS rel_type, prop
            ORDER BY rel_type, prop
            """
            rel_type_props = {}
            for record in session.run(rel_props_query):
                rel_type = record["rel_type"]
                prop = record["prop"]
                rel_type_props.setdefault(rel_type, []).append({
                    "name": prop,
                    "description": ""
                })

            json_obj = {
                "labels": [
                    {
                        "name": label,
                        "description": "",
                        "properties": label_props.get(label, [])
                    } for label in labels
                ],
                "relationships": [
                    {
                        "type": rel,
                        "description": "",
                        "properties": rel_type_props.get(rel, [])
                    } for rel in rel_types
                ],
                "triples": triples
            }

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)

            return output_path

    def get_all_node_names(self, label: str = None):
        """
        Get the name property of every node with the given label.
        :param label: node label, for example 'Effect', 'Symptom', or 'Disease'
        :return: List[str]
        """
        if label is None:
            query = """
            MATCH (n)
            RETURN DISTINCT n.name AS name
            ORDER BY name
            """
        else:
            query = f"""
            MATCH (n:{label})
            RETURN DISTINCT n.name AS name
            ORDER BY name
            """
        with self.driver.session() as session:
            result = session.run(query)
            return [record["name"] for record in result if record["name"]]

    def validate_cypher(self, query: str) -> bool:
        """
        Check whether a Cypher query is syntactically valid.
        :param query: Cypher statement to check
        :return: True when the statement is valid, False otherwise
        """
        try:
            with self.driver.session() as session:
                # EXPLAIN parses the query without executing it
                session.run(f"EXPLAIN {query}")
            return True
        except Exception as e:
            print(f"Cypher syntax error: {e}")
            return False

    def clear_database(self):
        """
        Delete every node, relationship, index, and constraint in the database.
        Use with care.
        """
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")

            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                name = record.get("name")
                if name:
                    session.run(f"DROP INDEX {name}")

            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                name = record.get("name")
                if name:
                    session.run(f"DROP CONSTRAINT {name}")

        print("Database cleared: all nodes, relationships, indexes, and constraints were deleted.")


neo4j_client = Neo4jClient(conf.NEO4J_URI, conf.NEO4J_USER, conf.NEO4J_PASSWORD)

if __name__ == "__main__":
    result = neo4j_client.run_cypher("MATCH (n) RETURN count(n) AS count")
    print(result)
    neo4j_client.close()
