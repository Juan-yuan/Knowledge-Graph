"""
Tests for import_tcm_graph.

    python -m unittest create_neo4j_database.test_import_tcm_graph -v

The Neo4jImportTest cases write the real extraction data into the configured Neo4j
(idempotent MERGE, nothing is deleted) and are skipped when Neo4j is unreachable.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_neo4j_database.import_tcm_graph import (
    DEFAULT_DATA_DIRS,
    ENTITY_LABELS,
    PROJECT,
    RELATION_SCHEMA,
    build_graph,
    build_write_queries,
    import_graph,
    load_extract_dicts,
)


def _doc(entities, relations, file="test.json"):
    return {"file": file, "entities": entities, "relations": relations}


class BuildGraphTest(unittest.TestCase):
    def test_nodes_attributes_and_relations(self):
        nodes, rels = build_graph([_doc(
            [
                {"name": "Bupleurum", "type": "Herb", "herb_attributes": {
                    "nature": "slightly cold", "meridians": ["Liver"], "scientific_name": None}},
                {"name": "Raises yang", "type": "Effect"},
                {"name": "Xiao Chai Hu Tang", "type": "Formula", "formula_attributes": None},
            ],
            [
                {"source": "Bupleurum", "target": "Raises yang", "type": "HAS_EFFECT"},
                {"source": "Xiao Chai Hu Tang", "target": "Bupleurum", "type": "HAS_INGREDIENT"},
            ],
        )])
        self.assertEqual(nodes[("Herb", "Bupleurum")],
                         {"nature": "slightly cold", "meridians": ["Liver"]})
        self.assertEqual(nodes[("Formula", "Xiao Chai Hu Tang")], {})
        self.assertEqual(rels, {
            ("Herb", "Bupleurum", "HAS_EFFECT", "Effect", "Raises yang"),
            ("Formula", "Xiao Chai Hu Tang", "HAS_INGREDIENT", "Herb", "Bupleurum"),
        })

    def test_deduplicates_across_files_and_keeps_richest_attributes(self):
        own_page = _doc([{"name": "Ephedra", "type": "Herb",
                          "herb_attributes": {"nature": "warm", "flavor": "pungent"}}], [])
        mention = _doc([{"name": "Ephedra", "type": "Herb",
                         "herb_attributes": {"nature": "hot"}}], [])
        for docs in ([own_page, mention], [mention, own_page]):
            nodes, _ = build_graph(docs)
            self.assertEqual(nodes, {("Herb", "Ephedra"): {"nature": "warm", "flavor": "pungent"}})

    def test_same_name_different_labels_are_separate_nodes(self):
        nodes, _ = build_graph([_doc([{"name": "diarrhea", "type": "Disease"},
                                      {"name": "diarrhea", "type": "Symptom"}], [])])
        self.assertIn(("Disease", "diarrhea"), nodes)
        self.assertIn(("Symptom", "diarrhea"), nodes)

    def test_relation_picks_label_matching_relation_type(self):
        _, rels = build_graph([_doc(
            [{"name": "Gan Cao", "type": "Herb"},
             {"name": "diarrhea", "type": "Disease"},
             {"name": "diarrhea", "type": "Symptom"}],
            [{"source": "Gan Cao", "target": "diarrhea", "type": "ALLEVIATES_SYMPTOM"}],
        )])
        self.assertEqual(rels, {("Herb", "Gan Cao", "ALLEVIATES_SYMPTOM", "Symptom", "diarrhea")})

    def test_undeclared_endpoints_are_inferred(self):
        nodes, rels = build_graph([
            _doc([{"name": "Ju Hua", "type": "Herb"}], [], file="a.json"),
            _doc([], [
                {"source": "Yin Qiao San", "target": "wind-warmth", "type": "TREATS_DISEASE"},
                {"source": "Some Soup", "target": "Ju Hua", "type": "HAS_INGREDIENT"},
            ], file="b.json"),
        ])
        self.assertEqual(rels, {
            ("Formula", "Yin Qiao San", "TREATS_DISEASE", "Disease", "wind-warmth"),
            ("Formula", "Some Soup", "HAS_INGREDIENT", "Herb", "Ju Hua"),
        })
        self.assertIn(("Disease", "wind-warmth"), nodes)
        self.assertIn(("Formula", "Some Soup"), nodes)

    def test_reversed_from_source_is_fixed(self):
        _, rels = build_graph([_doc(
            [{"name": "Ben Cao", "type": "Source"}, {"name": "Gao Ben", "type": "Herb"}],
            [{"source": "Ben Cao", "target": "Gao Ben", "type": "FROM_SOURCE"}],
        )])
        self.assertEqual(rels, {("Herb", "Gao Ben", "FROM_SOURCE", "Source", "Ben Cao")})

    def test_unknown_types_and_blank_names_are_ignored(self):
        nodes, rels = build_graph([_doc(
            [{"name": " ", "type": "Herb"}, {"name": "X", "type": "Mineral"},
             {"name": "Y", "type": "Herb"}],
            [{"source": "Y", "target": "Z", "type": "UNKNOWN_REL"}],
        )])
        self.assertEqual(set(nodes), {("Herb", "Y")})
        self.assertEqual(rels, set())

    def test_write_queries_are_valid_batches(self):
        nodes = {("Herb", f"h{i}"): {} for i in range(1200)}
        queries = build_write_queries(nodes, set())
        self.assertEqual([len(p["rows"]) for _, p in queries], [500, 500, 200])
        self.assertTrue(all("MERGE (n:Herb {name: row.name})" in q for q, _ in queries))


class RealDataTest(unittest.TestCase):
    def test_real_files_parse_and_every_relation_has_nodes(self):
        docs = load_extract_dicts(DEFAULT_DATA_DIRS)
        self.assertEqual(len(docs), 84)
        nodes, rels = build_graph(docs)
        self.assertGreater(len(nodes), 1000)
        self.assertGreater(len(rels), 2000)
        for src_label, src, _, tgt_label, tgt in rels:
            self.assertIn((src_label, src), nodes)
            self.assertIn((tgt_label, tgt), nodes)
        self.assertEqual(nodes[("Herb", "Bupleurum (Chai Hu)")]["meridians"],
                         ["Liver Channel", "Gallbladder Channel"])


class Neo4jImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from common.neo4j_manager import neo4j_client
            neo4j_client.driver.verify_connectivity()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"Neo4j not available: {e}")
        cls.client = neo4j_client
        cls.nodes, cls.rels = import_graph(cls.client)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def _counts(self):
        n = self.client.run_cypher(
            "MATCH (n {project: $p}) RETURN count(n) AS c", {"p": PROJECT})[0]["c"]
        r = self.client.run_cypher(
            "MATCH ({project: $p})-[r]->() RETURN count(r) AS c", {"p": PROJECT})[0]["c"]
        return n, r

    def test_counts_match_parsed_graph(self):
        self.assertEqual(self._counts(), (len(self.nodes), len(self.rels)))

    def test_reimport_is_idempotent(self):
        before = self._counts()
        import_graph(self.client)
        self.assertEqual(self._counts(), before)

    def test_herb_properties(self):
        row = self.client.run_cypher(
            "MATCH (h:Herb {name: $name}) RETURN properties(h) AS p",
            {"name": "Bupleurum (Chai Hu)"})[0]["p"]
        self.assertEqual(row["nature"], "slightly cold")
        self.assertEqual(row["meridians"], ["Liver Channel", "Gallbladder Channel"])
        self.assertEqual(row["project"], PROJECT)
        self.assertNotIn("scientific_name", {k for k, v in row.items() if v is None})

    def test_formula_ingredients_and_properties(self):
        rows = self.client.run_cypher(
            "MATCH (f:Formula {name: $name})-[:HAS_INGREDIENT]->(h:Herb) "
            "RETURN f.dosage AS dosage, collect(h.name) AS herbs",
            {"name": "Kudzu Decoction (Ge Gen Tang)"})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["dosage"])
        self.assertEqual(len(rows[0]["herbs"]), 7)
        self.assertIn("Ephedra (Ma Huang)", rows[0]["herbs"])

    def test_metadata_export_sees_all_labels(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.client.export_tcm_metadata_to_json(f"{tmp}/meta.json")
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        self.assertEqual({label["name"] for label in meta["labels"]}, ENTITY_LABELS)
        self.assertEqual(
            {rel["type"] for rel in meta["relationships"]},
            set(RELATION_SCHEMA),
        )


if __name__ == "__main__":
    unittest.main()
