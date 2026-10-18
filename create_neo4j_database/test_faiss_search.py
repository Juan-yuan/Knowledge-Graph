"""
Tests for the Neo4j entity-name FAISS index under langgraph_more_nodes/.

    python -m unittest create_neo4j_database.test_faiss_search -v

Loads:
    langgraph_more_nodes/nero4j_embedding_faiss.index
    langgraph_more_nodes/nero4j_embedding_faiss_id2text.pkl
"""
import os
import pickle
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from common.config import Config


def search_texts(index, id2text, query_vec, top_k: int = 5):
    """Return [(text, distance), ...] for the nearest neighbors of query_vec."""
    distances, indices = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        results.append((id2text[int(idx)], float(dist)))
    return results


class FaissIndexLoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = Config()
        cls.index_path = conf.ENTITY_INDEX_PATH
        cls.mapping_path = conf.ENTITY_ID2TEXT_PATH
        if not os.path.exists(cls.index_path) or not os.path.exists(cls.mapping_path):
            raise unittest.SkipTest(
                f"FAISS artifacts missing: {cls.index_path} / {cls.mapping_path}. "
                "Run create_neo4j_database/faiss_embedding.py first."
            )
        cls.index = faiss.read_index(cls.index_path)
        with open(cls.mapping_path, "rb") as f:
            cls.id2text = pickle.load(f)

    def test_files_are_non_empty(self):
        self.assertGreater(os.path.getsize(self.index_path), 0)
        self.assertGreater(os.path.getsize(self.mapping_path), 0)

    def test_index_and_mapping_sizes_match(self):
        self.assertIsInstance(self.id2text, dict)
        self.assertEqual(self.index.ntotal, len(self.id2text))
        self.assertGreater(self.index.ntotal, 0)

    def test_mapping_ids_are_contiguous(self):
        self.assertEqual(set(self.id2text.keys()), set(range(self.index.ntotal)))

    def test_mapping_values_are_non_empty_strings(self):
        sample = [self.id2text[i] for i in range(min(20, self.index.ntotal))]
        self.assertTrue(all(isinstance(t, str) and t.strip() for t in sample))

    def test_reconstruct_self_search(self):
        """A reconstructed vector should retrieve itself as the nearest neighbor."""
        target_id = min(10, self.index.ntotal - 1)
        query_vec = self.index.reconstruct(target_id).reshape(1, -1).astype(np.float32)
        results = search_texts(self.index, self.id2text, query_vec, top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0][0], self.id2text[target_id])
        self.assertAlmostEqual(results[0][1], 0.0, places=5)


class FaissSemanticSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = Config()
        cls.index_path = conf.ENTITY_INDEX_PATH
        cls.mapping_path = conf.ENTITY_ID2TEXT_PATH
        if not os.path.exists(cls.index_path) or not os.path.exists(cls.mapping_path):
            raise unittest.SkipTest(
                f"FAISS artifacts missing: {cls.index_path} / {cls.mapping_path}."
            )
        try:
            from common.embedding_model import embedding_model
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"Embedding model not available: {e}")

        cls.index = faiss.read_index(cls.index_path)
        with open(cls.mapping_path, "rb") as f:
            cls.id2text = pickle.load(f)
        cls.embedding_model = embedding_model
        cls.text_to_ids = {}
        for i, text in cls.id2text.items():
            cls.text_to_ids.setdefault(text, []).append(i)

    def _embed(self, text: str):
        return self.embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        ).astype(np.float32)

    def test_exact_name_ranks_first(self):
        # Prefer a known herb name if present; otherwise use any indexed text.
        query = "Ephedra (Ma Huang)"
        if query not in self.text_to_ids:
            query = self.id2text[0]
        results = search_texts(self.index, self.id2text, self._embed(query), top_k=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0][0], query)

    def test_semantic_query_returns_related_herb(self):
        query = "ma huang herb for sweating"
        results = search_texts(self.index, self.id2text, self._embed(query), top_k=10)
        self.assertEqual(len(results), 10)
        hit_texts = [text for text, _ in results]
        # At least one hit should mention Ephedra / Ma Huang if that entity exists
        if any("Ma Huang" in t or "Ephedra" in t for t in self.id2text.values()):
            self.assertTrue(
                any("Ma Huang" in t or "Ephedra" in t for t in hit_texts),
                f"Expected Ephedra/Ma Huang in top hits, got: {hit_texts}",
            )

    def test_top_k_distances_are_non_decreasing(self):
        results = search_texts(
            self.index,
            self.id2text,
            self._embed("fever and aversion to cold"),
            top_k=5,
        )
        distances = [dist for _, dist in results]
        self.assertEqual(distances, sorted(distances))


if __name__ == "__main__":
    unittest.main()
