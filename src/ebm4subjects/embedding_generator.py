import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
import inspect


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
        task: str = None,  # default is None
        use_tqdm: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dimensions))

        iterator = range(0, len(texts), batch_size)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Generating embeddings")

        embeddings = []
        for i in iterator:
            batch = texts[i : i + batch_size]
            # Check if the model's encode method supports the 'prompt_name' argument
            # as does jinai ai embeddings v3
            encode_params = inspect.signature(self.model.encode).parameters
            encode_args = {}
            # ToDo: Check if the model supports specific encoding tasks
            if 'prompt_name' in encode_params and task is not None:
                encode_args['prompt_name'] = task
            elif 'prompt_name' in encode_params and task is None:
                pass  # don't set prompt_name if task is None
            else:
                warnings.warn(f"Model {self.model_name} does not support specific encoding tasks", UserWarning)

            batch_embeddings = self.model.encode(
                batch,
                batch_size = batch_size,
                show_progress_bar=True if use_tqdm else False,
                **encode_args
            )
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        return embeddings
