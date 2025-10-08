from __future__ import annotations

import ast
from pathlib import Path

import joblib
import polars as pl
import xgboost as xgb

from ebm4subjects import prepare_data
from ebm4subjects.chunker import Chunker
from ebm4subjects.duckdb_client import Duckdb_client
from ebm4subjects.ebm_logging import EbmLogger, NullLogger, XGBLogging
from ebm4subjects.embedding_generator import EmbeddingGenerator


class EbmModel:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        use_altLabels: bool,
        duckdb_threads: int | str,
        embedding_model_name: str,
        embedding_dimensions: int | str,
        chunk_tokenizer: str,
        max_chunks: int | str,
        max_chunk_size: int | str,
        chunking_jobs: int | str,
        max_sentences: int | str,
        max_query_hits: int | str,
        query_top_k: int | str,
        query_jobs: int | str,
        xgb_shrinkage: float | str,
        xgb_interaction_depth: int | str,
        xgb_subsample: float | str,
        xgb_rounds: int | str,
        xgb_jobs: int | str,
        hnsw_index_params: dict | str | None = None,
        model_args: dict | str | None = None,
        encode_args_vocab: dict | str | None = None,
        encode_args_documents: dict | str | None = None,
        log_path: str | None = None,
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
        self.duckdb_threads = int(duckdb_threads)
        if isinstance(hnsw_index_params, str) or not hnsw_index_params:
            hnsw_index_params = (
                ast.literal_eval(hnsw_index_params) if hnsw_index_params else {}
            )
        self.hnsw_index_params = hnsw_index_params

        # Parameters for embedding generator
        self.generator = None
        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = int(embedding_dimensions)
        if isinstance(model_args, str) or not model_args:
            model_args = ast.literal_eval(model_args) if model_args else {}
        self.model_args = model_args
        if isinstance(encode_args_vocab, str) or not encode_args_vocab:
            encode_args_vocab = (
                ast.literal_eval(encode_args_vocab) if encode_args_vocab else {}
            )
        self.encode_args_vocab = encode_args_vocab
        if isinstance(encode_args_documents, str) or not encode_args_documents:
            encode_args_documents = (
                ast.literal_eval(encode_args_documents) if encode_args_documents else {}
            )
        self.encode_args_documents = encode_args_documents

        # Parameters for chunker
        self.chunk_tokenizer = chunk_tokenizer
        self.max_chunks = int(max_chunks)
        self.max_chunk_size = int(max_chunk_size)
        self.max_sentences = int(max_sentences)
        self.chunking_jobs = int(chunking_jobs)

        # Parameters for vector search
        self.max_query_hits = int(max_query_hits)
        self.query_top_k = int(query_top_k)
        self.query_jobs = int(query_jobs)

        # Parameters for XGB boost ranker
        self.train_shrinkage = float(xgb_shrinkage)
        self.train_interaction_depth = int(xgb_interaction_depth)
        self.train_subsample = float(xgb_subsample)
        self.train_rounds = int(xgb_rounds)
        self.train_jobs = int(xgb_jobs)

        # Parameters for logger
        # Only create logger if path to log file is set
        self.logger = None
        self.xgb_logger = None
        self.xgb_callbacks = None
        if log_path:
            self.logger = EbmLogger(log_path, "info").get_logger()
            self.xgb_logger = XGBLogging(self.logger, epoch_log_interval=1)
            self.xgb_callbacks = [self.xgb_logger]
        else:
            self.logger = NullLogger()

        # Initialize EBM model
        self.model = None

    def _init_duckdb_client(self) -> None:
        """
        Initializes the DuckDB client if it does not already exist.

        This method creates a new DuckDB client if it is not already
        initiliazed and configures it with the provided database path,
        thread settings, and HNSW index parameters.

        Returns:
            None
        """
        if self.client is None:
            self.logger.info("Initializing DuckDB client")

            self.client = Duckdb_client(
                db_path=self.db_path,
                config={
                    "hnsw_enable_experimental_persistence": True,
                    "threads": self.duckdb_threads,
                },
                hnsw_index_params=self.hnsw_index_params,
            )

    def _init_generator(self) -> None:
        """
        Initializes the embedding generator if it does not already exist.

        If the generator is not initialized, it creates a new EmbeddingGenerator
        with the specified model name, embedding dimensions, and model arguments.

        Returns:
            None
        """
        if self.generator is None:
            self.logger.info("Initializing embedding generator")

            self.generator = EmbeddingGenerator(
                model_name=self.embedding_model_name,
                embedding_dimensions=self.embedding_dimensions,
                **self.model_args,
            )

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
                f"Loading vocabulary with embeddings from {vocab_out_path}"
            )
            collection_df = pl.read_ipc(vocab_out_path)
        # Parse input vocabulary if provided
        elif vocab_in_path:
            self.logger.info("Parsing vocabulary")
            vocab = prepare_data.parse_vocab(
                vocab_path=vocab_in_path,
                use_altLabels=self.use_altLabels,
            )

            # Initialize generator and add embeddings to vocabulary
            self._init_generator()
            self.logger.info("Adding embeddings to vocabulary")
            collection_df = prepare_data.add_vocab_embeddings(
                vocab=vocab,
                generator=self.generator,
                encode_args=self.encode_args_vocab,
            )

            # Save vocabulary to output path if specified
            if vocab_out_path:
                # Check if file already exists and warn if so
                if Path(vocab_out_path).exists() and not force:
                    self.logger.warn(
                        f"""Cant't save vocabulary to {vocab_out_path}. 
                        File already exists"""
                    )
                else:
                    self.logger.info(f"Saving vocabulary to {vocab_out_path}")
                    collection_df.write_ipc(vocab_out_path)
        else:
            # If no existing vocabulary and no input vocabulary is provided,
            # raise an error
            raise ValueError("Vocabulary path is required")

        # Initialize DuckDB client and create collection
        self._init_duckdb_client()
        self.logger.info("Creating collection")
        self.client.create_collection(
            collection_df=collection_df,
            collection_name=self.collection_name,
            embedding_dimensions=self.embedding_dimensions,
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
        n_jobs: int = 0,
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
            train_candidates (pl.DataFrame, optional): Pre-computed candidate training data (default: None).
            n_jobs (int, optional): The number of jobs to use for parallel processing (default: 0).

        Returns:
            pl.DataFrame: The prepared training data.
        """

        self.logger.info("Preparing training data")
        # Check if pre-computed candidate training data is provided
        if not train_candidates:
            # If not, generate candidate training data in batches
            # If n_jobs is 0, use parameter of EBM model; otherwise, use given number of jobs
            if not n_jobs:
                train_candidates = self.generate_candidates_batch(
                    texts=texts,
                    doc_ids=doc_ids,
                )
            else:
                train_candidates = self.generate_candidates_batch(
                    texts=texts,
                    doc_ids=doc_ids,
                    chunking_jobs=n_jobs,
                    query_jobs=n_jobs,
                )

        # Create a gold standard data frame from the provided doc IDs and label IDs
        self.logger.info("Preparing gold standard")
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
        self.logger.info("Prepare training data and gold standard for training")
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
                how="outer",
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
        self, text: str, doc_id: int, n_jobs: int = 0
    ) -> pl.DataFrame:
        """
        Generates candidate labels for a given text and document ID.

        This method chunks the input text, generates embeddings for each chunk,
        and then uses vector search to find similar documents in the database.

        Args:
            text (str): The input text.
            doc_id (int): The document ID.
            n_jobs (int, optional): The number of jobs to use for parallel
                rocessing (default: 0).

        Returns:
            pl.DataFrame: A DataFrame containing the generated candidate labels.
        """
        # Check if n_jobs is provided, if not use number of jobs
        # specified in model parameters
        if not n_jobs:
            n_jobs = self.query_jobs

        # Create a Chunker instance with specified parameters
        self.logger.info("Chunking text")
        chunker = Chunker(
            tokenizer_name=self.chunk_tokenizer,
            max_chunks=self.max_chunks,
            max_chunk_size=self.max_chunk_size,
            max_sentences=self.max_sentences,
        )
        # Chunk the input text
        text_chunks = chunker.chunk_text(text)

        # Initialize the generator
        self._init_generator()
        self.logger.info("Creating embeddings for text chunks")
        # Generate embeddings for the text chunks
        embeddings = self.generator.generate_embeddings(
            # Use the text chunks as input
            texts=text_chunks,
            # Use the encode arguments for documents if provided
            **(
                self.encode_args_documents
                if self.encode_args_documents is not None
                else {}
            ),
        )

        # Create a query DataFrame
        self.logger.info("Creating query dataframe")
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
        self._init_duckdb_client()
        self.logger.info("Running vector search and creating candidates")
        # Perform vector search using the query DataFrame
        # Using the parameters specified for the EBM model
        # and the optimal chunk size for the DuckDB
        candidates = self.client.vector_search(
            query_df=query_df,
            collection_name=self.collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=n_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

        # Return generated candidates
        return candidates

    def generate_candidates_batch(
        self,
        texts: list[str],
        doc_ids: list[int],
        chunking_jobs: int = 0,
        query_jobs: int = 0,
    ) -> pl.DataFrame:
        """
        Generates candidate labels for a batch of given texts and document IDs.

        This method chunks the input texts, generates embeddings for each chunk,
        and then uses vector search to find similar documents in the database.

        Args:
            text (str): The input text.
            doc_id (int): The document ID.
            chunking_jobs (int, optional): The number of jobs to use for parallel
                chunking (default: 0).
            query_jobs (int, optional): The number of jobs to use for parallel
                querying (default: 0).

        Returns:
            pl.DataFrame: A DataFrame containing the generated candidate labels.
        """
        # Check if number of jobs are provided, if not use number of jobs
        # specified in model parameters
        if not chunking_jobs:
            chunking_jobs = self.chunking_jobs
        if not query_jobs:
            query_jobs = self.query_jobs

        # Create a Chunker instance with specified parameters
        self.logger.info("Chunking texts in batches")
        chunker = Chunker(
            tokenizer_name=self.chunk_tokenizer,
            max_chunks=self.max_chunks,
            max_chunk_size=self.max_chunk_size,
            max_sentences=self.max_sentences,
        )
        # Chunk the input texts
        text_chunks, chunk_index = chunker.chunk_batches(texts, doc_ids, chunking_jobs)

        # Initialize the generator and chunk index
        self._init_generator()
        chunk_index = pl.concat(chunk_index).with_row_index("query_id")
        self.logger.info("Creating embeddings for text chunks and query dataframe")
        embeddings = self.generator.generate_embeddings(
            texts=text_chunks,
            **(
                self.encode_args_documents
                if self.encode_args_documents is not None
                else {}
            ),
        )

        # Initialize the DuckDB client
        self._init_duckdb_client()
        # Extend chunk_index by a list column containing the embeddings
        query_df = chunk_index.with_columns(pl.Series("embeddings", embeddings))

        # Perform vector search using the query DataFrame
        # Using the parameters specified for the EBM model
        # and the optimal chunk size for the DuckDB
        self.logger.info("Running vector search and creating candidates")
        candidates = self.client.vector_search(
            query_df=query_df,
            collection_name=self.collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
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
            n_jobs = self.train_jobs

        # Select the required columns from the train_data DataFrame,
        # convert to a Pandas DataFrame and afterwards to training matrix
        self.logger.info("Creating training matrix")
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
            ).to_pandas(),
            # Use the gold standard as the target
            train_data.to_pandas()["gold"],
        )

        try:
            # Train the XGBoost model with the specified parameters
            self.logger.info("Starting training of XGBoost Ranker")
            model = xgb.train(
                # Train the XGBoost model with the specified parameters
                params={
                    "objective": "binary:logistic",  # Objective function to minimize
                    "eval_metric": "logloss",  # Evaluation metric
                    "eta": self.train_shrinkage,  # Learning rate
                    "max_depth": self.train_interaction_depth,  # Maximum tree depth
                    "subsample": self.train_subsample,  # Sampling ratio
                    "nthread": n_jobs,  # Number of threads to use
                },
                # Use the training matrix as the input data
                dtrain=matrix,
                # Disable verbose evaluation
                verbose_eval=False,
                # Evaluate the model on the training data
                evals=[(matrix, "train")],
                # Specify the number of boosting rounds
                num_boost_round=self.train_rounds,
                # Use the specified callbacks
                callbacks=self.xgb_callbacks,
            )
            self.logger.info("Training successful finished")
        except xgb.core.XGBoostError:
            self.logger.critical(
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
        self.logger.info("Creating matrix of candidates to generate predictions")
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
        self.logger.info("Making predictions for candidates")
        predictions = self.model.predict(matrix)

        # Transform the predictions into a list of DataFrames containing the
        # predicted scores and top-k labels for each document
        return (
            # Add a new column with the predicted scores to the candidates DataFrame
            candidates.with_columns(pl.Series(predictions).alias("score"))
            # Select the relevant columns from the updated DataFrame
            .select(["doc_id", "label_id", "score"])
            # Sort the DataFrame by document ID and score in ascending and
            # descending order, respectively
            .sort(["doc_id", "score"], descending=[False, True])
            # Group the DataFrame by document ID and aggregate the top-k labels
            # and scores for each group
            .group_by("doc_id")
            .agg(pl.all().head(self.query_top_k))
            # Explode the aggregated DataFrame to create separate rows for each
            # label and score
            .explode(["label_id", "score"])
            # Partition the DataFrame by document ID
            .partition_by("doc_id")
        )

    def save(self, output_path: str) -> None:
        """
        Saves the current state of the EBM model to a file using joblib.

        This method serializes the model instance and writes it to the
        specified output path, allowing for later deserialization and
        restoration of the model's state.

        Args:
            output_path: The file path where the serialized model will be written.

        Notes:
            The model's client and generator attributes are reset to None.
        """
        self.client = None
        self.generator = None
        joblib.dump(self, output_path)

    @staticmethod
    def load(input_path: str) -> EbmModel:
        """
        Loads an EBM model from a joblib serialized file.

        Args:
        input_path (str): Path to the joblib serialized file containing the EBM model.

        Returns:
        EbmModel: The loaded EBM model instance.
        """
        return joblib.load(input_path)
