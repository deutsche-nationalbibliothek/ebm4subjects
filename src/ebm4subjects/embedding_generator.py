import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    def __init__(self, model_name: str, embedding_dimensions: int) -> None:
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name, 
            device=device,
            truncate_dim=embedding_dimensions,
            trust_remote_code=True)

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 1,
        use_tqdm: bool = False,
        **kwargs
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dimensions))

        iterator = range(0, len(texts), batch_size)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Generating embeddings")

        embeddings = []
        for i in iterator:
            batch = texts[i : i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                **kwargs
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        return embeddings
