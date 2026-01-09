import os

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingGenerator:
    """
    A base class for embedding generators.
    """

    def __init__(self) -> None:
        """
        Base method fot the initialization of an EmbeddingGenerator.
        """
        pass

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Base method fot the creating embeddings with an EmbeddingGenerator.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: A numpy array of shape (len(texts), embedding_dimensions)
                containing the generated embeddings.
        """
        pass


class EmbeddingGeneratorAPI(EmbeddingGenerator):
    """
    A base class for API embedding generators.

    Attributes:
        embedding_dimensions (int): The dimensionality of the generated embeddings.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dimensions: int,
        **kwargs,
    ) -> None:
        """
        Initializes the API EmbeddingGenerator.

        Sets the embedding dimensions, and initiliazes and
        prepares a session with the API.
        """

        self.embedding_dimensions = embedding_dimensions
        self.model_name = model_name
        self.session = requests.Session()
        self.api_address = kwargs.get("api_address")
        self.headers = kwargs.get("headers", {"Content-Type": "application/json"})


class EmbeddingGeneratorHuggingFaceTEI(EmbeddingGeneratorAPI):
    """
    A class for generating embeddings using the HuggingFaceTEI API.
    """

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using a model
        via the HuggingFaceTEI API.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments to pass to the
                SentenceTransformer model.

        Returns:
            np.ndarray: A numpy array of shape (len(texts), embedding_dimensions)
                containing the generated embeddings.
        """
        # prepare list for return
        embeddings = []

        # Check if the input list is empty
        if not texts:
            # If empty, return an empty numpy array with the correct shape
            return np.empty((0, self.embedding_dimensions))

        # process each text
        for text in tqdm(texts, desc="Generating embeddings"):
            # send a request to the HuggingFaceTEI API
            data = {"inputs": text}
            response = self.session.post(
                self.api_address, headers=self.headers, json=data
            )

            # add generated embeddings to return list if request was successfull
            if response.status_code == 200:
                embeddings.append(response.json()[0])
            else:
                # TODO: write warning to logger
                embeddings.append([0 for _ in range(self.embedding_dimensions)])

        return np.array(embeddings)
    
class EmbeddingGeneratorOpenAI(EmbeddingGeneratorAPI):
    """
    A class for generating embeddings using any OpenAI compatibleAPI.
    """

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using a model
        via an OpenAI compatible API.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments to pass to the
                SentenceTransformer model.

        Returns:
            np.ndarray: A numpy array of shape (len(texts), embedding_dimensions)
                containing the generated embeddings.
        """
        # prepare list for return
        embeddings = []

        # Check if the input list is empty
        if not texts:
            # If empty, return an empty numpy array with the correct shape
            return np.empty((0, self.embedding_dimensions))

        # Process in smaller batches to avoid memory overload
        batch_size = min(200, len(texts))  
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            data = {
                "input": batch_texts,
                "model": self.model_name,
                "encoding_format": "float",
                **kwargs
            }

            response = self.session.post(
                self.api_address, headers=self.headers, json=data
            )

            # Process all embeddings from the batch response
            if response.status_code == 200:
                response_data = response.json()
                for i, text in enumerate(batch_texts):
                    embedding = response_data['data'][i]['embedding']
                    embeddings.append(embedding)
            else:
                # TODO: write warning to logger
                for _ in batch_texts:
                    embeddings.append([0 for _ in range(self.embedding_dimensions)])

        return np.array(embeddings)


class EmbeddingGeneratorOfflineInference(EmbeddingGenerator):
    """
    A class for generating embeddings using a given SentenceTransformer model
    loaded offline with SentenceTransformer.

    Args:
        model_name (str): The name of the SentenceTransformer model.
        embedding_dimensions (int): The dimensionality of the generated embeddings.
        **kwargs: Additional keyword arguments to pass to the model.

    Attributes:
        model_name (str): The name of the SentenceTransformer model.
        embedding_dimensions (int): The dimensionality of the generated embeddings.
    """

    def __init__(self, model_name: str, embedding_dimensions: int, **kwargs) -> None:
        """
        Initializes the EmbeddingGenerator in offline inference mode.

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


class EmbeddingGeneratorMock(EmbeddingGenerator):
    """
    A mock class for generating fake embeddings. Used for testing.

    Args:
        embedding_dimensions (int): The dimensionality of the generated embeddings.
        **kwargs: Additional keyword arguments to pass to the model.

    Attributes:
        embedding_dimensions (int): The dimensionality of the generated embeddings.
    """

    def __init__(self, embedding_dimensions: int, **kwargs) -> None:
        """
        Initializes the mock EmbeddingGenerator.

        Sets the embedding dimensions.
        """
        self.embedding_dimensions = embedding_dimensions

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: A numpy array of shape (len(texts), embedding_dimensions)
                containing the generated embeddings.
        """
        # Check if the input list is empty
        if not texts:
            # If empty, return an empty numpy array with the correct shape
            return np.empty((0, self.embedding_dimensions))

        # Generate mock embeddings return them
        return np.ones((len(texts), 1024))
