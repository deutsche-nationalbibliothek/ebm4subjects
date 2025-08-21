from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import joblib
import polars as pl
import xgboost as xgb
from tqdm import tqdm

from ebm4subjects import prepare_data
from ebm4subjects.chunker import Chunker
from ebm4subjects.duckdb_client import Duckdb_client
from ebm4subjects.ebm_logging import EbmLogger, XGBLogging
from ebm4subjects.embedding_generator import EmbeddingGenerator


class EbmModel:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        use_altLabels: bool,
        duckdb_threads: int,
        embedding_model_name: str,
        embedding_dimensions: int,
        chunk_tokenizer: str,
        max_chunks: int,
        max_chunk_size: int,
        chunking_jobs: int,
        max_sentences: int,
        max_query_hits: int,
        query_top_k: int,
        query_jobs: int,
        xgb_shrinkage: float,
        xgb_interaction_depth: int,
        xgb_subsample: float,
        xgb_rounds: int,
        xgb_jobs: int,
        hnsw_index_params: dict = None,
        model_args: dict = None,
        encode_args_vocab: dict = None,
        encode_args_documents: dict = None,
        log_path: str | None = None,
    ) -> None:
        # params for duckdb
        self.client = None
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_altLabels = use_altLabels
        self.duckdb_threads = duckdb_threads
        self.hnsw_index_params = (
            hnsw_index_params if hnsw_index_params is not None else {}
        )

        # params for embedding generator
        self.generator = None
        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = embedding_dimensions
        self.model_args = model_args if model_args is not None else {}
        self.encode_args_vocab = (
            encode_args_vocab if encode_args_vocab is not None else {}
        )
        self.encode_args_documents = (
            encode_args_documents if encode_args_documents is not None else {}
        )

        # params for chunker
        # create chunker with set params
        self.chunker = Chunker(
            tokenizer=chunk_tokenizer,
            max_chunks=max_chunks,
            max_chunk_size=max_chunk_size,
            max_sentences=max_sentences,
        )
        self.chunking_jobs = chunking_jobs

        # params for vector search
        self.max_query_hits = max_query_hits
        self.query_top_k = query_top_k
        self.query_jobs = query_jobs

        # params for xgb boost predictor
        self.train_shrinkage = xgb_shrinkage
        self.train_interaction_depth = xgb_interaction_depth
        self.train_subsample = xgb_subsample
        self.train_rounds = xgb_rounds
        self.train_jobs = xgb_jobs

        # params for logger
        # only create logger if path to log file is set
        self.logger = None
        self.xgb_logger = None
        self.xgb_callbacks = None
        if log_path:
            self.logger = EbmLogger(log_path, "info").get_logger()
            self.xgb_logger = XGBLogging(self.logger, epoch_log_interval=1)
            self.xgb_callbacks = [self.xgb_logger]

        # initialize ebm model
        self.model = None

    def _init_duckdb_client(self) -> None:
        if self.client is None:
            if self.logger:
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
        if self.generator is None:
            if self.logger:
                self.logger.info("Initializing embedding generator")

            self.generator = EmbeddingGenerator(
                model_name=self.embedding_model_name,
                embedding_dimensions=self.embedding_dimensions,
                **self.model_args,
            )

    def create_vector_db(
        self,
        vocab_in_path: str,
        vocab_out_path: str | None = None,
        force: bool = False,
    ) -> None:
        if vocab_out_path and Path(vocab_out_path).exists():
            if self.logger:
                self.logger.info(
                    f"Loading vocabulary with embeddings from {vocab_out_path}"
                )
            collection_df = pl.read_ipc(vocab_out_path)
        else:
            if self.logger:
                self.logger.info("Parsing vocabulary")
            vocab = prepare_data.parse_vocab(
                vocab_path=vocab_in_path,
                use_altLabels=self.use_altLabels,
            )

            if self.logger:
                self.logger.info("Adding embeddings to vocabulary")
            if self.generator is None:
                self._init_generator()
            collection_df = prepare_data.add_vocab_embeddings(
                vocab=vocab,
                generator=self.generator,
                encode_args=self.encode_args_vocab,
            )

            if vocab_out_path:
                if Path(vocab_out_path).exists() and not force:
                    if self.logger:
                        self.logger.warn(
                            f"Cant't save vocabulary to {vocab_out_path}. File already exists"
                        )
                else:
                    if self.logger:
                        self.logger.info(f"Saving vocabulary to {vocab_out_path}")
                    collection_df.write_ipc(vocab_out_path)

        if self.client is None:
            self._init_duckdb_client()
        if self.logger:
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
        gold_doc_ids: list[str],
        gold_label_ids: list[str],
        train_texts: list[str] = None,
        train_doc_ids: list[str] = None,
        train_candidates: pl.DataFrame = None,
    ) -> pl.DataFrame:
        if self.logger:
            self.logger.info("Preparing training data")
        if train_candidates is None:
            if train_texts is None or train_doc_ids is None:
                if self.logger:
                    self.logger.error("Training texts or document IDs are missing")
                return
            train_candidates = self._prepare_train_data(
                texts=train_texts, doc_ids=train_doc_ids
            )

        if self.logger:
            self.logger.info("Preparing gold standard")
        gold_standard = pl.DataFrame(
            {
                "doc_id": gold_doc_ids,
                "label_id": gold_label_ids,
            }
        ).with_columns(
            pl.col("doc_id").cast(pl.String), pl.col("label_id").cast(pl.String)
        )

        if self.logger:
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

        return training_data

    def _prepare_train_data(self, texts: list[str], doc_ids: list[int]) -> pl.DataFrame:
        candidates_dfs = []

        for text, doc_id in zip(texts, doc_ids):
            candidates_dfs.append(
                self.generate_candidates(
                    text=text,
                    doc_id=doc_id,
                )
            )

        return pl.concat(candidates_dfs)

    def _compare_to_gold_standard(
        self,
        candidates: pl.DataFrame,
        gold_standard: pl.DataFrame,
    ) -> pl.DataFrame:
        return (
            candidates.with_columns(pl.lit(True).alias("suggested"))
            .join(
                other=gold_standard.with_columns(pl.lit(True).alias("gold")),
                on=["doc_id", "label_id"],
                how="outer",
            )
            .with_columns(
                pl.col("suggested").fill_null(False),
                pl.col("gold").fill_null(False),
            )
            .filter(pl.col("suggested"))
        )

    def generate_candidates(
        self,
        text: str,
        doc_id: int,
    ) -> pl.DataFrame:
        if self.logger:
            self.logger.info("Chunking text")
        text_chunks = self.chunker.chunk_text(text)

        self._check_multi_device()

        if self.logger:
            self.logger.info("Creating embeddings for text chunks")
        if self.generator is None:
            self._init_generator()
        embeddings = self.generator.generate_embeddings(
            texts=text_chunks,
            **(
                self.encode_args_documents
                if self.encode_args_documents is not None
                else {}
            ),
        )

        if self.logger:
            self.logger.info("Creating query dataframe")
        query_df = pl.DataFrame(
            {
                "query_id": [i + 1 for i in range(len(text_chunks))],
                "query_doc_id": [doc_id for _ in range(len(text_chunks))],
                "chunk_position": [i + 1 for i in range(len(text_chunks))],
                "n_chunks": [len(text_chunks) for _ in range(len(text_chunks))],
                "embeddings": embeddings,
            }
        )

        if self.logger:
            self.logger.info("Running vector search and creating candidates")
        if self.client is None:
            self._init_duckdb_client()
        candidates = self.client.vector_search(
            query_df=query_df,
            collection_name=self.collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=self.query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

        return candidates

    def generate_candidates_batch(
        self, texts: list[str], doc_ids: list[int], use_tqdm: bool = False
    ) -> pl.DataFrame:
        if self.logger:
            self.logger.info("Chunking texts in batches")

        text_chunks = []
        chunk_index = []

        num_batches = self.chunking_jobs
        chunking_batch_size = ceil(len(texts) / num_batches)
        batch_args = [
            (
                doc_ids[i * chunking_batch_size : (i + 1) * chunking_batch_size],
                texts[i * chunking_batch_size : (i + 1) * chunking_batch_size],
                self.chunker,
            )
            for i in range(num_batches)
        ]

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.query_jobs) as executor:
            results = list(executor.map(self._chunk_batch, batch_args))

        for batch_chunks, batch_chunk_indices in results:
            text_chunks.extend(batch_chunks)
            chunk_index.extend(batch_chunk_indices)

        chunk_index = pl.concat(chunk_index).with_row_index("query_id")
        self.logger.info("Creating embeddings for text chunks and query dataframe")

        if self.generator is None:
            self._init_generator()

        embeddings = self.generator.generate_embeddings(
            texts=text_chunks,
            **(
                self.encode_args_documents
                if self.encode_args_documents is not None
                else {}
            ),
        )

        # Extend chunk_index by a list column containing the embeddings
        query_df = chunk_index.with_columns(pl.Series("embeddings", embeddings))

        if self.logger:
            self.logger.info("Running verctor search and creating candidates")

        # with multi-GPU processing in the generate embeddings process
        # duckdb may throw an error due to its incompatible handling of
        # multiple threads. Therefore late initialization of the client
        # is required.
        if self.client is None:
            self._init_duckdb_client()
        else:
            self._check_multi_device()

        candidates = self.client.vector_search(
            query_df=query_df,
            collection_name=self.collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=self.query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

        return candidates
    
    def _check_multi_device(self):
        if (
            self.encode_args_documents is not None
            and "device" in self.encode_args_documents
        ):
            device_val = self.encode_args_documents["device"]
            if isinstance(device_val, list) and len(device_val) > 1:
                warnings.warn(
                    "Multi-device (multi-GPU/CPU) processing may cause conflicts in "
                    "generate_candidates or with duckdb multithreading if the database "
                    "client is already initialized. Set enocding args to single device "
                    "or use generate_candidates_batch"
                )

    def train(self, train_data: pl.DataFrame) -> None:
        if self.logger:
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
            train_data.to_pandas()["gold"],
        )

        try:
            if self.logger:
                self.logger.info("Starting training of XGBoost Ranker")
            model = xgb.train(
                params={
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "eta": self.train_shrinkage,
                    "max_depth": self.train_interaction_depth,
                    "subsample": self.train_subsample,
                    "nthread": self.train_jobs,
                },
                dtrain=matrix,
                verbose_eval=False,
                evals=[(matrix, "train")],
                num_boost_round=self.train_rounds,
                callbacks=self.xgb_callbacks,
            )
            if self.logger:
                self.logger.info("Training successful finished")
        except xgb.core.XGBoostError:
            if self.logger:
                self.logger.critical(
                    "XGBoost can't train with candidates equal to gold standard "
                    "or candidates with no match to gold standard at all - "
                    "Check if your training data and gold standard are correct"
                )
            return

        self.model = model

    def predict(self, candidates: pl.DataFrame) -> list[pl.DataFrame]:
        if self.logger:
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

        if self.logger:
            self.logger.info("Making predictions for candidates")
        predictions = self.model.predict(matrix)

        return (
            candidates.with_columns(pl.Series(predictions).alias("score"))
            .select(["doc_id", "label_id", "score"])
            .sort(["doc_id", "score"], descending=[False, True])
            .group_by("doc_id")
            .agg(pl.all().head(self.query_top_k))
            .explode(["label_id", "score"])
            .partition_by("doc_id")
        )

    def _chunk_batch(args):
        batch_doc_ids, batch_texts, chunker = args
        batch_chunks = []
        batch_chunk_indices = []
        for doc_id, text in tqdm(
            zip(batch_doc_ids, batch_texts),
            total=len(batch_doc_ids),
            desc="Chunking batch",
        ):
            new_chunks = chunker.chunk_text(text)
            n_chunks = len(new_chunks)
            chunk_df = pl.DataFrame(
                {
                    "query_doc_id": [doc_id] * n_chunks,
                    "chunk_position": list(range(n_chunks)),
                    "n_chunks": [n_chunks] * n_chunks,
                }
            )
            batch_chunks.extend(new_chunks)
            batch_chunk_indices.append(chunk_df)
        return batch_chunks, batch_chunk_indices

    def save(self, output_path: str, force: bool = False) -> None:
        if Path(output_path).exists() and not force:
            if self.logger:
                self.logger.warn(
                    f"Cant't save model to {output_path}. Model already exists. Try force=True to overwrite model file"
                )
            return
        else:
            self.client = None
            self.generator = None
            joblib.dump(self, output_path)

    @staticmethod
    def load(input_path: str, load_generator: bool = True) -> EbmModel:
        model = joblib.load(input_path)
        model.client = model.init_duckdb_client()
        if load_generator:
            model.generator = model.init_generator()

        return model
