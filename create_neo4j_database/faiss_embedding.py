"""
Build a FAISS index over Neo4j node names for entity retrieval.

Output paths default to Config.ENTITY_INDEX_PATH / ENTITY_ID2TEXT_PATH.
"""
import os
import pickle
import sys
from pathlib import Path
from typing import List

# Avoid tokenizer/OpenMP multiprocessing crashes on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.embedding_model import embedding_model
from common.neo4j_manager import neo4j_client

import faiss


def build_faiss_index(
    sentences: List[str],
    index_path: str,
    mapping_path: str,
    batch_size: int = 32,
) -> None:
    """
    Build a FAISS index from a list of strings and save it to disk.

    Args:
        sentences: input texts to index
        index_path: path for the FAISS index file
        mapping_path: path for the id -> original text pickle mapping
        batch_size: embedding batch size
    """
    if not sentences:
        raise ValueError("sentences must not be empty")

    embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, index_path)

    id2text = {i: s for i, s in enumerate(sentences)}
    with open(mapping_path, "wb") as f:
        pickle.dump(id2text, f)

    print(f"Index saved to {index_path}; mapping saved to {mapping_path}")
    print(f"Indexed {len(sentences)} text(s), dimension={dim}.")


def main() -> None:
    conf = Config()
    node_names = neo4j_client.get_all_node_names()
    print(f"Loaded {len(node_names)} node name(s) from Neo4j.")
    build_faiss_index(
        node_names,
        index_path=conf.ENTITY_INDEX_PATH,
        mapping_path=conf.ENTITY_ID2TEXT_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        neo4j_client.close()
