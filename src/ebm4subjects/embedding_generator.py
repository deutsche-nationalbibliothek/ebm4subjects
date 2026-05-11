import logging
import os

import numpy as np
import requests
from openai import OpenAI, OpenAIError
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from tqdm import tqdm


class EmbeddingGenerator:
    """
    A base class for embedding generators.
    """

    def __init__(self) -> None:
        """
        Base method for the initialization of an EmbeddingGenerator.
        """
        pass

    def generate_embeddings(self, texts: list[str], **kwargs) -> np.ndarray:
        """
        Base method for the creating embeddings with an EmbeddingGenerator.

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
        cache_url: str = "",
        **kwargs,
    ) -> None:
        """
        Initializes the HuggingFaceTEI API EmbeddingGenerator.

        Sets the embedding dimensions, and initializes and
        prepares a session with the API.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            embedding_dimensions (int): The dimensionality of the generated embeddings.
            logger (Logger): A logger for the embedding generator.
            cache_url (str, optional): URL to a cache to save generated embeddings.
            **kwargs: Additional keyword arguments to pass to the model.
        """

        self.embedding_dimensions = embedding_dimensions
        self.model_name = model_name
        self.session = requests.Session()
        self.api_address = kwargs.get("api_address")
        self.headers = kwargs.get("headers", {"Content-Type": "application/json"})

        self.redis_cache = None
        if cache_url:
            self.redis_cache = RedisCacheConnector(
                model_name=self.model_name,
                cache_url=cache_url,
            )

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
        else:
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

            cached_texts = []
            new_texts = batch_texts
            generated_embeddings = []

            # Check for which texts embeddings are saved if a cache is existing
            if self.redis_cache:
                self.logger.debug(
                    f"Checking cache for previous generated embeddings for {len(batch_texts)} texts"
                )
                cached_texts, new_texts = self.redis_cache.check_batch(batch_texts)
                if cached_texts:
                    self.logger.debug(
                        f"Retrieving {len(cached_texts)} previous generated embeddings from cache"
                    )
                    cached_embeddings = self.redis_cache.get_batch(cached_texts)

            # send a request to the HuggingFaceTEI API
            if new_texts:
                self.logger.debug(
                    f"Generating new embeddings for {len(new_texts)} texts"
                )
                data = {"inputs": new_texts, "truncate": True}
                response = self.session.post(
                    self.api_address, headers=self.headers, json=data
                )

                # Add generated embeddings to return list if request was successful
                if response.status_code == 200:
                    generated_embeddings = response.json()

                    # Store all generated embddings in cache if existing
                    if self.redis_cache:
                        self.logger.debug(
                            f"Storing {len(generated_embeddings)} generated embeddings in cache"
                        )
                        self.redis_cache.add_batch(new_texts, generated_embeddings)

                # Retur 0's if call to API was not successful
                else:
                    self.logger.warning("Call to API NOT successful! Returning 0's.")
                    for _ in batch_texts:
                        generated_embeddings.append(
                            [0 for _ in range(self.embedding_dimensions)]
                        )

            # Combine list of cached and generated embeddings into return list
            for text in batch_texts:
                if text in cached_texts:
                    embeddings.append(cached_embeddings[cached_texts.index(text)])
                if text in new_texts:
                    embeddings.append(generated_embeddings[new_texts.index(text)])

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
        cache_url: str = "",
        **kwargs,
    ) -> None:
        """
        Initializes the OpenAI API EmbeddingGenerator.

        Sets the embedding dimensions, and initializes and
        prepares a session with the API.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            embedding_dimensions (int): The dimensionality of the generated embeddings.
            cache_url (str, optional): URL to a cache to save generated embeddings.
            logger (Logger): A logger for the embedding generator.
            **kwargs: Additional keyword arguments to pass to the model.
        """

        self.embedding_dimensions = embedding_dimensions
        self.model_name = model_name

        if not (api_key := os.environ.get("OPENAI_API_KEY")):
            api_key = ""

        self.client = OpenAI(api_key=api_key, base_url=kwargs.get("api_address"))

        self.redis_cache = None
        if cache_url:
            self.redis_cache = RedisCacheConnector(
                model_name=self.model_name,
                cache_url=cache_url,
            )

        self.logger = logger
        self._test_api()

    def _test_api(self):
        """
        Tests if the API is working with the given parameters
        """
        _ = self.client.embeddings.create(
            input="This is a test request!",
            model=self.model_name,
            encoding_format="float",
        )
        self.logger.debug("API call successful. Everything seems to be working fine.")

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

            cached_texts = []
            new_texts = batch_texts
            generated_embeddings = []

            # Check for which texts embeddings are saved if a cache is existing
            if self.redis_cache:
                self.logger.debug(
                    f"Checking cache for previous generated embeddings for {len(batch_texts)} texts"
                )
                cached_texts, new_texts = self.redis_cache.check_batch(batch_texts)
                if cached_texts:
                    self.logger.debug(
                        f"Retrieving {len(cached_texts)} previous generated embeddings from cache"
                    )
                    cached_embeddings = self.redis_cache.get_batch(cached_texts)

            # Try to get embeddings for the (new texts ot the) batch from the API
            if new_texts:
                try:
                    self.logger.debug(
                        f"Generating new embeddings for {len(new_texts)} texts"
                    )
                    embedding_response = self.client.embeddings.create(
                        input=new_texts,
                        model=self.model_name,
                        encoding_format="float",
                        extra_body={**kwargs},
                    )

                    # Process all embeddings from the batch response
                    for i, _ in enumerate(new_texts):
                        generated_embeddings.append(
                            embedding_response.data[i].embedding
                        )

                    # Store all generated embddings in cache if existing
                    if self.redis_cache:
                        self.logger.debug(
                            f"Storing {len(generated_embeddings)} generated embeddings in cache"
                        )
                        self.redis_cache.add_batch(new_texts, generated_embeddings)

                # Retur 0's if call to API was not successful
                except OpenAIError:
                    self.logger.warning("Call to API NOT successful! Returning 0's.")
                    for _ in new_texts:
                        generated_embeddings.append(
                            [0 for _ in range(self.embedding_dimensions)]
                        )

            # Combine list of cached and generated embeddings into return list
            for text in batch_texts:
                if text in cached_texts:
                    embeddings.append(cached_embeddings[cached_texts.index(text)])
                if text in new_texts:
                    embeddings.append(generated_embeddings[new_texts.index(text)])

        return np.array(embeddings)


class EmbeddingGeneratorInProcess(EmbeddingGenerator):
    """
    A class for generating embeddings using a given SentenceTransformer model
    loaded in-process with SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dimensions: int,
        logger: logging.Logger,
        cache_url: str = "",
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingGenerator in 'in-process' mode.

        Sets the model name, embedding dimensions, and creates a
        SentenceTransformer model instance.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            embedding_dimensions (int): The dimensionality of the generated embeddings.
            logger (Logger): A logger for the embedding generator.
            cache_url (str, optional): URL to a cache to save generated embeddings.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions

        # Create a SentenceTransformer model instance with the given
        # model name and embedding dimensions
        self.model = SentenceTransformer(
            model_name, truncate_dim=embedding_dimensions, **kwargs
        )

        self.redis_cache = None
        if cache_url:
            self.redis_cache = RedisCacheConnector(
                model_name=self.model_name,
                cache_url=cache_url,
            )

        self.logger = logger
        self.logger.debug(f"SentenceTransformer model running on {self.model.device}")

        # Disable parallelism for tokenizer
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
        # If input list is empty, return an empty numpy array with the correct shape
        if not texts:
            return np.empty((0, self.embedding_dimensions))

        embeddings = []
        cached_texts = []
        new_texts = texts
        generated_embeddings = []

        # Check for which texts embeddings are saved if a cache is existing
        if self.redis_cache:
            self.logger.debug(
                f"Checking cache for previous generated embeddings for {len(texts)} texts"
            )
            cached_texts, new_texts = self.redis_cache.check_batch(texts)
            if cached_texts:
                self.logger.debug(
                    f"Retrieving {len(cached_texts)} previous generated embeddings from cache"
                )
                cached_embeddings = self.redis_cache.get_batch(cached_texts)

        if new_texts:
            # Generate embeddings using the SentenceTransformer model
            self.logger.debug(f"Generating new embeddings for {len(new_texts)} texts")
            generated_embeddings = self.model.encode(new_texts, **kwargs)

            # Store all generated embddings in cache if existing
            if self.redis_cache:
                self.logger.debug(
                    f"Storing {len(generated_embeddings)} generated embeddings in cache"
                )
                self.redis_cache.add_batch(new_texts, generated_embeddings)

        # Combine list of cached and generated embeddings into return list
        for text in texts:
            if text in cached_texts:
                embeddings.append(cached_embeddings[cached_texts.index(text)])
            if text in new_texts:
                embeddings.append(generated_embeddings[new_texts.index(text)])

        return np.array(embeddings)


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


class RedisCacheConnector:
    def __init__(
        self,
        model_name: str,
        cache_url: str = "redis://localhost:6379",
    ) -> None:
        """
        Initializes the connection to the set up redis cache.

        Args:
            model_name (str): The name of the SentenceTransformer model.
            cache_url (str): The URL to the set up redis cache.
        """
        self.model_name = model_name
        self.cache = EmbeddingsCache(redis_url=cache_url)

    def add_batch(
        self,
        texts: list[str],
        embeddings: list[np.ndarray],
    ) -> list[str]:
        """
        Adds a list of texts together with generated embeddings to the cache.

        Args:
            texts (list[str]): A list of input texts.
            embeddings (list[np.ndarray]): A list of corresponding embeddings.

        Returns:
            list[str]: A list of status reports.
        """
        # Generate list of items with model name, text and embedding
        batch_items = [
            {
                "content": texts[i],
                "model_name": self.model_name,
                "embedding": embeddings[i],
            }
            for i in range(len(texts))
        ]

        # Add items to cache
        keys = self.cache.mset(batch_items)
        return keys

    def check_batch(self, texts: list[str]) -> tuple[list[str], list[str]]:
        """
        Check for list of texts if there are already generated embeddings in the cache.

        Args:
            texts (list[str]): A list of input texts.

        Returns:
            tuple[list[str], list[str]]: A list of text that are already in the cache
                and a list of texts that are not in the cache
        """
        # Check which texts are already saved in the cache
        exist_results = self.cache.mexists(texts, self.model_name)

        # Split input list in list of already saved and unseen texts
        cached_texts = [text for i, text in enumerate(texts) if exist_results[i]]
        new_texts = [text for i, text in enumerate(texts) if not exist_results[i]]

        return cached_texts, new_texts

    def get_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for list of texts from cache.

        Args:
            texts (list[str]): A list of input texts.

        Returns:
            list[list[float]]: A list of embeddings saved for the input texts.
        """
        # Get saved embeddings for the list of texts
        results = self.cache.mget(texts, self.model_name)

        return [result["embedding"] for result in results]
