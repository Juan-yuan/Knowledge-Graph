import os
import sys
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.path_utils import get_file_path

device = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path(configured_path: str | None) -> str:
    """Resolve EMBEDDING_MODEL_PATH against the project root."""
    if not configured_path:
        raise FileNotFoundError("EMBEDDING_MODEL_PATH is not set.")
    path = configured_path if os.path.isabs(configured_path) else get_file_path(configured_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding model not found: {path}")
    return path


class EmbeddingModel:
    """Turns text into vector embeddings."""

    def __init__(self):
        """Load the embedding model from EMBEDDING_MODEL_PATH."""
        config = Config()
        model_path = resolve_model_path(config.EMBEDDING_MODEL_PATH)
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: List[str], convert_to_numpy=False, normalize_embeddings=False) -> List[List[float]]:
        """
        Convert a list of texts into vectors.

        Args:
            texts: texts to encode
            convert_to_numpy: whether to return a numpy array
            normalize_embeddings: whether to normalize each vector

        Returns:
            One vector per text, as a list or a numpy array
        """
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )


embedding_model = EmbeddingModel()

if __name__ == "__main__":
    print(device)
    print(embedding_model)
    print(embedding_model.encode(["Hello"]))
    print(embedding_model.model.get_embedding_dimension())
