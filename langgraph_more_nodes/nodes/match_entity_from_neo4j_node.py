import os
import pickle
import sys
from pathlib import Path
from typing import List

# Avoid tokenizer/OpenMP multiprocessing crashes on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np

from common.config import Config
from common.embedding_model import embedding_model
from langgraph_more_nodes.agent_state import AgentState

conf = Config()
index = faiss.read_index(conf.ENTITY_INDEX_PATH)
with open(conf.ENTITY_ID2TEXT_PATH, "rb") as f:
    id2text = pickle.load(f)

# user_input_* field -> matched_* field
INPUT_TO_MATCHED_KEY = {
    "user_input_effects": "matched_effects",
    "user_input_diseases": "matched_diseases",
    "user_input_symptoms": "matched_symptoms",
    "user_input_formulas": "matched_formulas",
    "user_input_herbs": "matched_herbs",
    "user_input_sources": "matched_sources",
}

TOP_K = 5
# Squared L2 distance on normalized vectors (0 = identical, 4 = opposite).
# The nearest hit is accepted below MATCH_THRESHOLD; further hits only below
# NEAR_DUPLICATE_THRESHOLD (catches graph variants like "fever" / "Fever").
MATCH_THRESHOLD = 0.6
NEAR_DUPLICATE_THRESHOLD = 0.15


def search_texts(query_vecs: np.ndarray, top_k: int = TOP_K):
    """Return one [(text, distance), ...] list per query vector."""
    distances, indices = index.search(query_vecs, top_k)
    results = []
    for dist_row, idx_row in zip(distances, indices):
        results.append([
            (id2text[int(idx)], float(dist))
            for dist, idx in zip(dist_row, idx_row)
            if idx >= 0
        ])
    return results


def match_entities(entities: List[str]) -> List[str]:
    """Map extracted entity names to node names in the Neo4j graph."""
    if not entities:
        return []

    query_vecs = embedding_model.encode(
        entities,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    matched = []
    for hits in search_texts(query_vecs):
        if not hits or hits[0][1] > MATCH_THRESHOLD:
            continue
        for rank, (text, dist) in enumerate(hits):
            if (rank == 0 or dist <= NEAR_DUPLICATE_THRESHOLD) and text not in matched:
                matched.append(text)
    return matched


def match_entity_from_neo4j_node(state: AgentState) -> AgentState:
    """Match user_input_* entities to Neo4j node names via FAISS similarity search."""
    for input_key, matched_key in INPUT_TO_MATCHED_KEY.items():
        state[matched_key] = match_entities(state.get(input_key) or [])
    return state


if __name__ == "__main__":
    result = match_entity_from_neo4j_node({
        "user_input_effects": ["induce sweating", "release exterior"],
        "user_input_diseases": ["exterior wind-cold", "insomnia"],
        "user_input_symptoms": ["fever", "aversion to cold", "vivid dreams"],
        "user_input_formulas": ["Ephedra Decoction (Ma Huang Tang)", "Cinnamon Twig Decoction (Gui Zhi Tang)"],
        "user_input_herbs": ["Astragalus (Huang Qi)", "Ginseng"],
        "user_input_sources": ["Shang Han Lun"],
    })
    for input_key, matched_key in INPUT_TO_MATCHED_KEY.items():
        print(f"{result[input_key]} -> {result[matched_key]}")
