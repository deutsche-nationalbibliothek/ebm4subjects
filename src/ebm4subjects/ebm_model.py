from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import xgboost as xgb

from ebm4subjects import prepare_data
from ebm4subjects.chunker import Chunker
from ebm4subjects.duckdb_client import Duckdb_client
from ebm4subjects.ebm_logging import EbmLogger, NullLogger, XGBLogging
from ebm4subjects.embedding_generator import (
    EmbeddingGeneratorHuggingFaceTEI,
    EmbeddingGeneratorInProcess,
    EmbeddingGeneratorMock,
    EmbeddingGeneratorOpenAI,
)


class EbmModel:
    def __init__(
        self,
        embedding_dimensions: int | str,
        chunk_tokenizer: str | Any,
        max_chunk_count: int | str,
        max_chunk_length: int | str,
        chunking_jobs: int | str,
        max_sentence_count: int | str,
        candidates_per_chunk: int | str,
        candidates_per_doc: int | str,
        query_jobs: int | str,
        xgb_shrinkage: float | str,
        xgb_interaction_depth: int | str,
        xgb_subsample: float | str,
        xgb_rounds: int | str,
        xgb_jobs: int | str,
        duckdb_threads: int | str,
        db_path: str,
        collection_name: str = "my_collection",
        use_altLabels: bool = True,
        hnsw_index_params: dict | str | None = None,
        embedding_model_name: str | None = None,
        embedding_model_deployment: str = "mock",
        embedding_model_args: dict | str | None = None,
        encode_args_vocab: dict | str | None = None,
        encode_args_documents: dict | str | None = None,
        log_path: str | None = None,
        logger: logging.Logger | None = None,
        logging_level: str = "info",
    ) -> None:
        """
        A class representing an Embedding-Based-Matching (EBM) model
        for automated subject indexing for texts.

        The EBM model integrates multiple components, including:
        - A DuckDB client for database operations
        - An EmbeddingGenerator for generating embeddings from text data
        - A Chunker for chunking text into smaller pieces
        - An XGBoost Ranker model for ranking candidate labels

        The EBM model provides methods for creating a vector database,
        preparing training data, training the XGBoost Ranker model,
        and making predictions on generated candidate labels.

        Attributes:
            client (DuckDB client): The DuckDB client instance
            generator (EmbeddingGenerator): The EmbeddingGenerator instance
            chunker (Chunker): The Chunker instance
            model (XGBoost Ranker model): The trained XGBoost Ranker model

        Methods:
            create_vector_db: Creates a vector database by loading an existing
                vocabulary with embeddings or generating a new
                vocabulary with embeddings
            prepare_train: Prepares the training data for the EBM model
            train: Trains the XGBoost Ranker model using the provided training data
            predict: Generates predictions for given candidates using the trained model
            save: Saves the current state of the EBM model to a file
            load: Loads an EBM model from a file

        Notes:
            All parameters with type hints like 'TYPE | str' are expecting a parameter
            of type TYPE, but can also accept the parameter as string. The parameter is
            then cast to the needed type.
        """
        # Parameters for duckdb
        self.client = None
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_altLabels = use_altLabels
        self.duckdb_threads = duckdb_threads
        self.hnsw_index_params = hnsw_index_params

        # Parameters for embedding generator
        self.generator = None
        self.embedding_model_deployment = embedding_model_deployment
        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = embedding_dimensions

        self.embedding_model_args = embedding_model_args
        self.encode_args_vocab = encode_args_vocab
        self.encode_args_documents = encode_args_documents

        # Parameters for chunker
        self.chunk_tokenizer = chunk_tokenizer
        self.max_chunk_count = max_chunk_count
        self.max_chunk_length = max_chunk_length
        self.max_sentence_count = max_sentence_count
        self.chunking_jobs = chunking_jobs

        # Parameters for vector search
        self.candidates_per_chunk = candidates_per_chunk
        self.candidates_per_doc = candidates_per_doc
        self.query_jobs = query_jobs

        # Parameters for XGB boost ranker
        self.train_shrinkage = xgb_shrinkage
        self.train_interaction_depth = xgb_interaction_depth
        self.train_subsample = xgb_subsample
        self.train_rounds = xgb_rounds
        self.train_jobs = xgb_jobs

        # Initiliaze logging
        self.init_logger(log_path, logger, logging_level)

        # Initialize EBM model
        self.model = None

    def _init_duckdb_client(self, params: dict[str:Any] = {}) -> None:
        """
        Initializes the DuckDB client if it does not already exist.

        This method creates a new DuckDB client if it is not already
        initiliazed and configures it with the provided database path,
        thread settings, and HNSW index parameters.

        Args:
            params (dict, optional): A dictionary with parameters to overwrite
                model configuration.

        Returns:
            None
        """
        # parse duckdb client params
        duckdb_threads = int(params.get("duckdb_threads", self.duckdb_threads))
        db_path = params.get("db_path", self.db_path)
        hnsw_index_params = params.get("hnsw_index_params", self.hnsw_index_params)
        if isinstance(hnsw_index_params, str) or not hnsw_index_params:
            hnsw_index_params = (
                ast.literal_eval(hnsw_index_params) if hnsw_index_params else {}
            )

        # initiliaze client
        if self.client is None:
            self.logger.info(
                f"initializing DuckDB client with duckdb_threads: {duckdb_threads}"
            )
            self.client = Duckdb_client(
                db_path=db_path,
                config={
                    "hnsw_enable_experimental_persistence": True,
                    "threads": duckdb_threads,
                },
                hnsw_index_params=hnsw_index_params,
            )

    def _init_generator(self, params: dict[str:Any] = {}) -> None:
        """
        Initializes the embedding generator if it does not already exist.

        If the generator is not initialized, it creates a new EmbeddingGenerator
        with the specified model name, embedding dimensions, and model arguments.

        Args:
            params (dict, optional): A dictionary with parameters to overwrite
                model configuration.

        Returns:
            None
        """
        # parse embedding generator params
        embedding_model_deployment = params.get(
            "embedding_model_deployment", self.embedding_model_deployment
        ).lower()
        model_name = params.get("embedding_model_name", self.embedding_model_name)
        embedding_dimensions = int(self.embedding_dimensions)
        embedding_model_args = params.get(
            "embedding_model_args", self.embedding_model_args
        )
        if isinstance(embedding_model_args, str) or not embedding_model_args:
            embedding_model_args = (
                ast.literal_eval(embedding_model_args) if embedding_model_args else {}
            )

        # initiliaze embedding generator
        if self.generator is None:
            if embedding_model_deployment == "in-process":
                self.logger.info("initializing in-process embedding generator")
                self.generator = EmbeddingGeneratorInProcess(
                    model_name=model_name,
                    embedding_dimensions=embedding_dimensions,
                    logger=self.logger,
                    **embedding_model_args,
                )
            elif embedding_model_deployment == "mock":
                self.logger.info("initializing mock embedding generator")
                self.generator = EmbeddingGeneratorMock(embedding_dimensions)
            elif embedding_model_deployment == "huggingfacetei":
                self.logger.info("initializing API embedding generator")
                self.generator = EmbeddingGeneratorHuggingFaceTEI(
                    model_name=model_name,
                    embedding_dimensions=embedding_dimensions,
                    logger=self.logger,
                    **embedding_model_args,
                )
            elif embedding_model_deployment == "openai":
                self.logger.info("initializing API embedding generator")
                self.generator = EmbeddingGeneratorOpenAI(
                    model_name=model_name,
                    embedding_dimensions=embedding_dimensions,
                    logger=self.logger,
                    **embedding_model_args,
                )
            else:
                raise NotImplementedError(
                    f"Unsupportet deployment '{embedding_model_deployment}' for embedding generator"
                )

    def init_logger(
        self,
        log_path: str | None = None,
        logger: logging.Logger | None = None,
        logging_level: str = "info",
    ) -> None:
        """
        Initializes the logging for the EBM model.

        Returns:
            None
        """
        if log_path:
            self.logger = EbmLogger(log_path, logging_level).get_logger()
            self.xgb_logger = XGBLogging(self.logger, epoch_log_interval=1)
            self.xgb_callbacks = [self.xgb_logger]
        elif logger:
            self.logger = logger
            self.xgb_logger = XGBLogging(self.logger, epoch_log_interval=1)
            self.xgb_callbacks = [self.xgb_logger]
        else:
            self.logger = NullLogger()
            self.xgb_logger = None
            self.xgb_callbacks = None

    def create_vector_db(
        self,
        vocab_in_path: str | None = None,
        vocab_out_path: str | None = None,
        force: bool = False,
    ) -> None:
        """
        Creates a vector database by either loading an existing vocabulary
        with embeddings or generating a new vocabulary with embeddings from scratch.

        If a vocabulary with embeddings already exists at the specified output path,
        it will be loaded. Otherwise, a new vocabulary will be generated from the input
        vocabulary path, and the resulting vocabulary with embeddings will be saved to
        the output path if specified.

        Args:
            vocab_in_path (optional): The path to the input vocabulary file.
            vocab_out_path (optional): The path to the output vocabulary file
                with embeddings.
            force: Whether to overwrite an existing output file (default: False).

        Returns:
            None

        Raises:
            ValueError: If no vocabulary is provided.
        """
        # Check if output path exists and load existing vocabulary if so
        if vocab_out_path and Path(vocab_out_path).exists():
            self.logger.info(
                f"loading vocabulary with embeddings from {vocab_out_path}"
            )
            collection_df = pl.read_ipc(vocab_out_path)
        # Parse input vocabulary if provided
        elif vocab_in_path:
            self.logger.info("parsing vocabulary")
            vocab = prepare_data.parse_vocab(
                vocab_path=vocab_in_path,
                use_altLabels=self.use_altLabels,
            )

            # Initialize generator and add embeddings to vocabulary
            self._init_generator()

            encode_args_vocab = self.encode_args_vocab
            if isinstance(encode_args_vocab, str) or not encode_args_vocab:
                encode_args_vocab = (
                    ast.literal_eval(encode_args_vocab) if encode_args_vocab else {}
                )
            self.logger.info("adding embeddings to vocabulary")
            collection_df = prepare_data.add_vocab_embeddings(
                vocab=vocab,
                generator=self.generator,
                encode_args=encode_args_vocab,
            )

            # Save vocabulary to output path if specified
            if vocab_out_path:
                # Check if file already exists and warn if so
                if Path(vocab_out_path).exists() and not force:
                    self.logger.warning(
                        f"""cant't save vocabulary to {vocab_out_path}. 
                        File already exists"""
                    )
                else:
                    self.logger.info(f"saving vocabulary to {vocab_out_path}")
                    collection_df.write_ipc(vocab_out_path)
        else:
            # If no existing vocabulary and no input vocabulary is provided,
            # raise an error
            raise ValueError("vocabulary path is required")

        # Initialize DuckDB client and create collection
        self._init_duckdb_client()
        self.logger.info("creating collection")
        self.client.create_collection(
            collection_df=collection_df,
            collection_name=self.collection_name,
            embedding_dimensions=int(self.embedding_dimensions),
            hnsw_index_name="hnsw_index",
            hnsw_metric="cosine",
            force=force,
        )

    def prepare_train(
        self,
        doc_ids: list[str],
        label_ids: list[str],
        texts: list[str],
        train_candidates: pl.DataFrame = None,
        params: dict[str:Any] = {},
    ) -> pl.DataFrame:
        """
        Prepares the training data for the EBM model.

        This function generates candidate training data and a gold standard
        data frame. It then compares the candidates to the gold standard,
        computes the necessary features, and returns the resulting training data.

        Args:
            doc_ids (list[str]): A list of document IDs.
            label_ids (list[str]): A list of label IDs.
            texts (list[str]): A list of text data.
            train_candidates (pl.DataFrame, optional): Pre-computed candidate
                training data (default: None).
            params (dict, optional): A dictionary with parameters to overwrite
                model configuration.

        Returns:
            pl.DataFrame: The prepared training data.
        """
        # Check if pre-computed candidate training data is provided
        # If not, generate candidate training data in batches
        if not train_candidates:
            train_candidates = self.generate_candidates_batch(
                texts=texts, doc_ids=doc_ids, params=params
            )

        # Create a gold standard data frame from the provided doc IDs and label IDs
        gold_standard = pl.DataFrame(
            {
                "doc_id": doc_ids,
                "label_id": label_ids,
            }
        ).with_columns(
            pl.col("doc_id").cast(pl.String), pl.col("label_id").cast(pl.String)
        )

        # Compare the candidate training data to the gold standard
        # and prepare data for the training of the XGB ranker model
        self.logger.info("prepare training data and gold standard for training")
        training_data = (
            self._compare_to_gold_standard(train_candidates, gold_standard)
            .with_columns(pl.when(pl.col("gold")).then(1).otherwise(0).alias("gold"))
            .filter(pl.col("doc_id").is_not_null())
            .select(
                [
                    "score",
                    "occurrences",
                    "min_cosine_similarity",
                    "max_cosine_similarity",
                    "first_occurence",
                    "last_occurence",
                    "spread",
                    "is_prefLabel",
                    "n_chunks",
                    "gold",
                ]
            )
        )

        # Return the prepared training data
        return training_data

    def _compare_to_gold_standard(
        self,
        candidates: pl.DataFrame,
        gold_standard: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compare the model's suggested labels to the gold standard labels.

        This method joins the model's suggested labels with the gold standard labels
        on the 'doc_id' and 'label_id' columns, filling any missing values with False.
        It then filters the resulting DataFrame to only include suggested labels.

        Args:
            candidates (pl.DataFrame): The model's suggested labels.
            gold_standard (pl.DataFrame): The gold standard labels.

        Returns:
            pl.DataFrame: A DataFrame containing the model's suggested labels that match
                the gold standard labels.
        """
        return (
            # Mark suggested candidates and gold standard labels
            # Join candidates and gold standard
            candidates.with_columns(pl.lit(True).alias("suggested"))
            .join(
                other=gold_standard.with_columns(pl.lit(True).alias("gold")),
                on=["doc_id", "label_id"],
                how="full",
            )
            # Fill dataframe so that all not suggested labels which are not part of
            # the gold standard and all gold standard labels which where not
            # suggested are marked
            .with_columns(
                pl.col("suggested").fill_null(False),
                pl.col("gold").fill_null(False),
            )
            # Keep only suggested labels
            .filter(pl.col("suggested"))
        )

    def generate_candidates(
        self, text: str, doc_id: int, params: dict[str:Any] = {}
    ) -> pl.DataFrame:
        """
        Generates candidate labels for a given text and document ID.

        This method chunks the input text, generates embeddings for each chunk,
        and then uses vector search to find similar documents in the database.

        Args:
            text (str): The input text.
            doc_id (int): The document ID.
            params (dict, optional): A dictionary with parameters to overwrite
                model configuration.

        Returns:
            pl.DataFrame: A DataFrame containing the generated candidate labels.
        """
        # process text if not empty
        if text:
            # Create a Chunker instance with specified parameters
            self.logger.info("chunking text")
            chunker = Chunker(
                self.chunk_tokenizer,
                int(params.get("max_chunk_count", self.max_chunk_count)),
                int(params.get("max_chunk_length", self.max_chunk_length)),
                int(params.get("max_sentence_count", self.max_sentence_count)),
            )
            # Chunk the input text
            text_chunks = chunker.chunk_text(text)

            # Initialize the generator
            self._init_generator(params)
            self.logger.info("creating embeddings for text chunks")

            # Parse the 'encode_args_documents' parameter
            encode_args_documents = params.get(
                "encode_args_documents", self.encode_args_documents
            )
            if isinstance(encode_args_documents, str) or not encode_args_documents:
                encode_args_documents = (
                    ast.literal_eval(encode_args_documents)
                    if encode_args_documents
                    else {}
                )

            # Generate embeddings for the text chunks
            embeddings = self.generator.generate_embeddings(
                # Use the text chunks as input
                texts=text_chunks,
                # Use the encode arguments for documents if provided
                **encode_args_documents,
            )

            # Create a query DataFrame
            self.logger.info("creating query dataframe")
            query_df = pl.DataFrame(
                {
                    # Create a column for the query ID
                    "query_id": [i + 1 for i in range(len(text_chunks))],
                    # Create a column for the query document ID
                    "query_doc_id": [doc_id for _ in range(len(text_chunks))],
                    # Create a column for the chunk position
                    "chunk_position": [i + 1 for i in range(len(text_chunks))],
                    # Create a column for the number of chunks
                    "n_chunks": [len(text_chunks) for _ in range(len(text_chunks))],
                    # Create a column for the embeddings
                    "embeddings": embeddings,
                }
            )

            # Initialize the DuckDB client
            self._init_duckdb_client(params)
            query_jobs = int(params.get("query_jobs", self.query_jobs))
            self.logger.info(
                f"running vector search and creating candidates with query_jobs: {query_jobs}"
            )
            # Perform vector search using the query DataFrame
            # Using the parameters specified for the EBM model
            # and the optimal chunk size for the DuckDB
            candidates = self.client.vector_search(
                query_df=query_df,
                collection_name=self.collection_name,
                embedding_dimensions=int(self.embedding_dimensions),
                n_jobs=query_jobs,
                n_hits=int(
                    params.get("candidates_per_chunk", self.candidates_per_chunk)
                ),
                chunk_size=1024,
                top_k=int(params.get("candidates_per_doc", self.candidates_per_doc)),
                hnsw_metric_function="array_cosine_distance",
            )

            # Return generated candidates
            return candidates

        # return empty candidates dataframe if text is empty
        else:
            return pl.DataFrame(
                schema={
                    "doc_id": pl.String,
                    "label_id": pl.String,
                    "score": pl.Float64,
                    "occurrences": pl.Float64,
                    "min_cosine_similarity": pl.Float64,
                    "max_cosine_similarity": pl.Float64,
                    "first_occurence": pl.Float64,
                    "last_occurence": pl.Float64,
                    "spread": pl.Float64,
                    "is_prefLabel": pl.Boolean,
                    "n_chunks": pl.Int32,
                },
                data={
                    "doc_id": [str(doc_id)],
                    "label_id": None,
                    "score": None,
                    "occurrences": None,
                    "min_cosine_similarity": None,
                    "max_cosine_similarity": None,
                    "first_occurence": None,
                    "last_occurence": None,
                    "spread": None,
                    "is_prefLabel": None,
                    "n_chunks": None,
                },
            )

    def generate_candidates_batch(
        self,
        texts: list[str],
        doc_ids: list[int],
        params: dict[str:Any] = {},
    ) -> pl.DataFrame:
        """
        Generates candidate labels for a batch of given texts and document IDs.

        This method chunks the input texts, generates embeddings for each chunk,
        and then uses vector search to find similar documents in the database.

        Args:
            text (str): The input text.
            doc_id (int): The document ID.
            params (dict, optional): A dictionary with parameters to overwrite
                model configuration.

        Returns:
            pl.DataFrame: A DataFrame containing the generated candidate labels.
        """
        # Create a Chunker instance with specified parameters
        chunker = Chunker(
            self.chunk_tokenizer,
            int(params.get("max_chunk_count", self.max_chunk_count)),
            int(params.get("max_chunk_length", self.max_chunk_length)),
            int(params.get("max_sentence_count", self.max_sentence_count)),
        )
        # Chunk the input texts
        chunking_jobs = int(params.get("chunking_jobs", self.chunking_jobs))
        self.logger.info(f"chunking texts with chunking_jobs: {chunking_jobs}")
        text_chunks, chunk_index = chunker.chunk_batches(texts, doc_ids, chunking_jobs)

        # Parse the 'encode_args_documents' parameter
        encode_args_documents = params.get(
            "encode_args_documents", self.encode_args_documents
        )
        if isinstance(encode_args_documents, str) or not encode_args_documents:
            encode_args_documents = (
                ast.literal_eval(encode_args_documents) if encode_args_documents else {}
            )

        # Initialize the generator and chunk index
        self._init_generator(params)
        chunk_index = pl.concat(chunk_index).with_row_index("query_id")
        self.logger.info("creating embeddings for text chunks and query dataframe")
        embeddings = self.generator.generate_embeddings(
            texts=text_chunks,
            **encode_args_documents,
        )

        # Initialize the DuckDB client
        self._init_duckdb_client(params)
        # Extend chunk_index by a list column containing the embeddings
        query_df = chunk_index.with_columns(pl.Series("embeddings", embeddings))

        # Perform vector search using the query DataFrame
        # Using the parameters specified for the EBM model
        # and the optimal chunk size for the DuckDB
        query_jobs = int(params.get("query_jobs", self.query_jobs))
        self.logger.info(
            f"running vector search and creating candidates with query_jobs: {query_jobs}"
        )
        candidates = self.client.vector_search(
            query_df=query_df,
            collection_name=self.collection_name,
            embedding_dimensions=int(self.embedding_dimensions),
            n_jobs=query_jobs,
            n_hits=int(params.get("candidates_per_chunk", self.candidates_per_chunk)),
            chunk_size=1024,
            top_k=int(params.get("candidates_per_doc", self.candidates_per_doc)),
            hnsw_metric_function="array_cosine_distance",
        )

        # add empty columns for documents without candidates
        for missing_doc_id in [
            doc_id
            for doc_id in [str(d) for d in doc_ids]
            if doc_id not in candidates.get_column("doc_id").to_list()
        ]:
            candidates.extend(
                pl.DataFrame(
                    schema=candidates.schema,
                    data={
                        "doc_id": [str(missing_doc_id)],
                        "label_id": None,
                        "score": None,
                        "occurrences": None,
                        "min_cosine_similarity": None,
                        "max_cosine_similarity": None,
                        "first_occurence": None,
                        "last_occurence": None,
                        "spread": None,
                        "is_prefLabel": None,
                        "n_chunks": None,
                    },
                )
            )

        # Return generated candidates
        return candidates

    def train(self, train_data: pl.DataFrame, n_jobs: int = 0) -> None:
        """
        Trains the XGBoost Ranker model using the provided training data.

        Args:
            train_data: The data to be used for training.
            n_jobs (int, optional): The number of jobs to use for parallel
                processing (default: 0).

        Returns:
            None

        Raises:
            XGBoostError: If XGBoost is unable to train with candidates.
        """
        # Check if n_jobs is provided, if not use number of jobs
        # specified in model parameters
        if not n_jobs:
            n_jobs = int(self.train_jobs)

        # Select the required columns from the train_data DataFrame,
        # convert to a numpy array and afterwards to training matrix
        self.logger.info("creating training matrix")
        matrix = xgb.DMatrix(
            train_data.select(
                [
                    "score",
                    "occurrences",
                    "min_cosine_similarity",
                    "max_cosine_similarity",
                    "first_occurence",
                    "last_occurence",
                    "spread",
                    "is_prefLabel",
                    "n_chunks",
                ]
            ).to_numpy(),
            # Use the gold standard as the target
            train_data.select("gold").to_numpy(),
        )

        try:
            # Train the XGBoost model with the specified parameters
            self.logger.info(
                f"starting training of XGBoost Ranker with xgb_jobs: {n_jobs}"
            )
            model = xgb.train(
                # Train the XGBoost model with the specified parameters
                params={
                    "objective": "binary:logistic",  # Objective function to minimize
                    "eval_metric": "logloss",  # Evaluation metric
                    "eta": float(self.train_shrinkage),  # Learning rate
                    "max_depth": int(
                        self.train_interaction_depth
                    ),  # Maximum tree depth
                    "subsample": float(self.train_subsample),  # Sampling ratio
                    "nthread": n_jobs,  # Number of threads to use
                },
                # Use the training matrix as the input data
                dtrain=matrix,
                # Disable verbose evaluation
                verbose_eval=False,
                # Evaluate the model on the training data
                evals=[(matrix, "train")],
                # Specify the number of boosting rounds
                num_boost_round=int(self.train_rounds),
                # Use the specified callbacks
                callbacks=self.xgb_callbacks,
            )
            self.logger.info("training successful finished")
        except xgb.core.XGBoostError:
            self.logger.warning(
                "XGBoost can't train with candidates equal to gold standard "
                "or candidates with no match to gold standard at all - "
                "Check if your training data and gold standard are correct"
            )
            raise
        else:
            # Store the trained model
            self.model = model

    def predict(self, candidates: pl.DataFrame) -> list[pl.DataFrame]:
        """
        Generates predictions for the given candidates using the trained model.

        This method creates a matrix from the candidates DataFrame, makes predictions
        using the trained model, and returns a list of DataFrames containing the
        predicted scores and top-k labels for each document.

        Args:
            candidates (pl.DataFrame): A DataFrame containing the candidates to
                generate predictions for.

        Returns:
            list[pl.DataFrame]: A list of DataFrames, where each DataFrame contains
            the predicted scores and top-k labels for a document.
        """
        # Select relevant columns from the candidates DataFrame to create a matrix
        # for the trained model to make predictions
        self.logger.info("creating matrix of candidates to generate predictions")
        matrix = xgb.DMatrix(
            candidates.select(
                [
                    "score",
                    "occurrences",
                    "min_cosine_similarity",
                    "max_cosine_similarity",
                    "first_occurence",
                    "last_occurence",
                    "spread",
                    "is_prefLabel",
                    "n_chunks",
                ]
            )
        )

        # Use the trained model to make predictions on the created matrix
        self.logger.info("making predictions for candidates")
        predictions = self.model.predict(matrix)

        # Transform the predictions into a list of DataFrames containing the
        # predicted scores and top-k labels for each document
        return (
            # Add a new column with the predicted scores to the candidates DataFrame
            candidates.with_columns(pl.Series(predictions).alias("score"))
            # Select the relevant columns from the updated DataFrame
            .select(["doc_id", "label_id", "score"])
            # Set score to -1.0 if no candidate was found for document
            .with_columns(
                pl.when(pl.col("label_id").is_null())
                .then(-1)
                .otherwise(pl.col.score)
                .alias("score")
            )
            # Sort the DataFrame by document ID and score in ascending and
            # descending order, respectively
            .with_columns(pl.col("doc_id").cast(pl.Int64))
            .sort(["doc_id", "score"], descending=[False, True])
            # Group the DataFrame by document ID and aggregate the top-k labels
            # and scores for each group
            .group_by("doc_id")
            .agg(pl.all().head(int(self.candidates_per_doc)))
            # Explode the aggregated DataFrame to create separate rows for each
            # label and score
            .explode(["label_id", "score"])
            # Partition the DataFrame by document ID
            .partition_by("doc_id")
        )

    def save(self, output_path: str) -> list[str]:
        """
        Saves the ranker of the EBM model to a file using joblib.

        Args:
            output_path: The file path where the serialized ranker will be written to.

        Returns:
            list[str]: Output path of file.
        """
        return joblib.dump(self.model, output_path)

    def load(self, input_path: str) -> None:
        """
        Loads an EBM ranker model from a joblib serialized file.

        Args:
            input_path (str): Path to the joblib serialized file.
        """
        self.model = joblib.load(input_path)
