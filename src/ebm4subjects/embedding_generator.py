import os

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    A class for generating embeddings using a given SentenceTransformer model.

    Args:
        model_name (str): The name of the SentenceTransformer model to use.
        embedding_dimensions (int): The dimensionality of the generated embeddings.
        **kwargs: Additional keyword arguments to pass to the model.

    Attributes:
        model_name (str): The name of the SentenceTransformer model.
        embedding_dimensions (int): The dimensionality of the generated embeddings.
        model (SentenceTransformer): The SentenceTransformer model instance.
    """

    def __init__(self, model_name: str, embedding_dimensions: int, **kwargs) -> None:
        """
        Initializes the EmbeddingGenerator.

        Sets the model name, embedding dimensions, and creates a
        SentenceTransformer model instance.
        """
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions

        # Create a SentenceTransformer model instance with the given
        # model name and embedding dimensions
        self.model = SentenceTransformer(
            model_name, truncate_dim=embedding_dimensions, **kwargs
        )

        # Disabel parallelism for tokenizer
        # Needed because process might be already parallelized
        # before embedding creation
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using the
        SentenceTransformer model.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments to pass to the
                SentenceTransformer model.

        Returns:
            np.ndarray: A numpy array of shape (len(texts), embedding_dimensions)
                containing the generated embeddings.
        """
        # Check if the input list is empty
        if not texts:
            # If empty, return an empty numpy array with the correct shape
            return np.empty((0, self.embedding_dimensions))

        # Generate embeddings using the SentenceTransformer model and return them
        return self.model.encode(texts, **kwargs)
