import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm


class EmbeddingGenerator:
    def __init__(self, model_name: str, embedding_dimensions: int) -> None:
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(device)  # Move to device first

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for model inference.")
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()

    def generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 1,
        task: str = "text-matching",
        use_tqdm: bool = False
    ) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            iterator = range(0, len(texts), batch_size)
            if use_tqdm:
                iterator = tqdm(iterator, desc="Generating embeddings")
            for i in iterator:
                embeddings.append(
                    self.model.module.encode(
                        texts[i : i + batch_size],
                        truncate_dim=self.embedding_dimensions,
                        task=task,
                    ) if isinstance(self.model, torch.nn.DataParallel) else
                    self.model.encode(
                        texts[i : i + batch_size],
                        truncate_dim=self.embedding_dimensions,
                        task=task,
                    )
                )

        return np.concatenate(embeddings)
