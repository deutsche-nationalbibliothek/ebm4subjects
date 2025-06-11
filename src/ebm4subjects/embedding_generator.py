import numpy as np
import torch
from transformers import AutoModel


class EmbeddingGenerator:
    def __init__(self, model_name: str, embedding_dimensions: int) -> None:
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.eval()

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 1,
        task: str = "text-matching",
    ) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                embeddings.append(
                    self.model.encode(
                        texts[i : i + batch_size],
                        truncate_dim=self.embedding_dimensions,
                        task=task,
                    )
                )

        return np.concatenate(embeddings)
