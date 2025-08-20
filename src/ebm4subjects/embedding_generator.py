import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str, embedding_dimensions: int, **kwargs) -> None:
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions

        self.model = SentenceTransformer(
            model_name, truncate_dim=embedding_dimensions, **kwargs
        )

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dimensions))

        embeddings = self.model.encode(texts, **kwargs)

        return embeddings
