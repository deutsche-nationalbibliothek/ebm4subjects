import logging
import os

import numpy as np
import requests
from openai import BadRequestError, NotFoundError, OpenAI
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


class EmbeddingGeneratorHuggingFaceTEI(EmbeddingGenerator):
    """
    A class for generating embeddings using the HuggingFaceTEI API.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dimensions: int,
        logger: logging.Logger,
        **kwargs,
    ) -> None:
        """
        Initializes the HuggingFaceTEI API EmbeddingGenerator.

        Sets the embedding dimensions, and initiliazes and
        prepares a session with the API.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            embedding_dimensions (int): The dimensionality of the generated embeddings.
            logger (Logger): A logger for the embedding generator.
            **kwargs: Additional keyword arguments to pass to the model.
        """

        self.embedding_dimensions = embedding_dimensions
        self.model_name = model_name
        self.session = requests.Session()
        self.api_address = kwargs.get("api_address")
        self.headers = kwargs.get("headers", {"Content-Type": "application/json"})

        self.logger = logger
        self._test_api()

    def _test_api(self):
        """
        Tests if the API is working with the given parameters
        """
        response = self.session.post(
            self.api_address,
            headers=self.headers,
            json={"inputs": "This is a test request!", "truncate": True},
        )
        if response.status_code == 200:
            self.logger.debug(
                "API call successful. Everything seems to be working fine."
            )
        elif response.status_code == 404:
            self.logger.error(
                "API not found under given adress! Please check the corresponding parameter!"
            )
            raise RuntimeError(
                "API not found under given adress! Please check the corresponding parameter!"
            )
        else:
            self.logger.error(
                "Request to API not possible! Please check the corresponding parameters!"
            )
            raise RuntimeError(
                "Request to API not possible! Please check the corresponding parameters!"
            )

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using a model
        via the HuggingFaceTEI API.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments to pass to the API.

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
        batch_size = min(32, len(texts))  # HuggingFaceTEI has a limit of 32 as default

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i : i + batch_size]
            # send a request to the HuggingFaceTEI API
            data = {"inputs": batch_texts, "truncate": True}
            response = self.session.post(
                self.api_address, headers=self.headers, json=data
            )

            # add generated embeddings to return list if request was successfull
            if response.status_code == 200:
                embeddings.extend(response.json())
            else:
                self.logger.warn("Call to API NOT successful! Returning 0's.")
                for _ in batch_texts:
                    embeddings.append(
                        [
                            0
                            for _ in range(
                                min(
                                    self.embedding_dimensions,
                                    kwargs.get("truncate_prompt_tokens", float("inf")),
                                ),
                            )
                        ]
                    )

        return np.array(embeddings)


class EmbeddingGeneratorOpenAI(EmbeddingGenerator):
    """
    A class for generating embeddings using any OpenAI compatible API.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dimensions: int,
        logger: logging.Logger,
        **kwargs,
    ) -> None:
        """
        Initializes the OpenAI API EmbeddingGenerator.

        Sets the embedding dimensions, and initiliazes and
        prepares a session with the API.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            embedding_dimensions (int): The dimensionality of the generated embeddings.
            logger (Logger): A logger for the embedding generator.
            **kwargs: Additional keyword arguments to pass to the model.
        """

        self.embedding_dimensions = embedding_dimensions
        self.model_name = model_name

        if not (api_key := os.environ.get("OPENAI_API_KEY")):
            api_key = ""

        self.client = OpenAI(api_key=api_key, base_url=kwargs.get("api_address"))

        self.logger = logger
        self._test_api()

    def _test_api(self):
        """
        Tests if the API is working with the given parameters
        """
        try:
            _ = self.client.embeddings.create(
                input="This is a test request!",
                model=self.model_name,
                encoding_format="float",
            )
            self.logger.debug(
                "API call successful. Everything seems to be working fine."
            )
        except NotFoundError:
            self.logger.error(
                "API not found under given adress! Please check the corresponding parameter!"
            )
            raise RuntimeError(
                "API not found under given adress! Please check the corresponding parameter!"
            )
        except BadRequestError:
            self.logger.error(
                "Request to API not possible! Please check the corresponding parameters!"
            )
            raise RuntimeError(
                "Request to API not possible! Please check the corresponding parameters!"
            )

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using a model
        via an OpenAI compatible API.

        Args:
            texts (list[str]): A list of input texts.
            **kwargs: Additional keyword arguments to pass to the API.

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
            batch_texts = texts[i : i + batch_size]

            # Try to get embeddings for the batch from the API
            try:
                embedding_response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name,
                    encoding_format="float",
                    extra_body={**kwargs},
                )

                # Process all embeddings from the batch response
                for i, _ in enumerate(batch_texts):
                    embeddings.append(embedding_response.data[i].embedding)
            except (NotFoundError, BadRequestError):
                self.logger.warn("Call to API NOT successful! Returning 0's.")
                for _ in batch_texts:
                    embeddings.append([0 for _ in range(self.embedding_dimensions)])

        return np.array(embeddings)


class EmbeddingGeneratorInProcess(EmbeddingGenerator):
    """
    A class for generating embeddings using a given SentenceTransformer model
    loaded in-process with SentenceTransformer.

    Args:
        model_name (str): The name of the SentenceTransformer model.
        embedding_dimensions (int): The dimensionality of the generated embeddings.
        logger (Logger): A logger for the embedding generator.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dimensions: int,
        logger: logging.Logger,
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingGenerator in 'in-process' mode.

        Sets the model name, embedding dimensions, and creates a
        SentenceTransformer model instance.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions

        # Create a SentenceTransformer model instance with the given
        # model name and embedding dimensions
        self.model = SentenceTransformer(
            model_name, truncate_dim=embedding_dimensions, **kwargs
        )
        self.logger = logger
        self.logger.debug(f"SentenceTransfomer model running on {self.model.device}")

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
