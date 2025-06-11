import pickle
from pathlib import Path

import polars as pl
import xgboost as xgb

from ebm4subjects import prepare_data
from ebm4subjects.chunker import Chunker
from ebm4subjects.duckdb_client import Duckdb_client
from ebm4subjects.embedding_generator import EmbeddingGenerator


class EbmModel:
    def __init__(
        self,
        db_path: Path,
        use_altLabels: bool,
        embedding_model_name: str,
        embedding_dimensions: int,
        embedding_batch_size: int,
        max_chunks: int,
        max_chunk_size: int,
        max_sentences: int,
        max_query_hits: int,
        query_top_k: int,
        query_jobs: int,
        train_shrinkage: float,
        train_interaction_depth: int,
        train_subsample: float,
        train_rounds: int,
        train_verbosity: int,
        train_jobs: int,
    ) -> None:
        self.client = Duckdb_client(
            db_path=db_path,
            config={"hnsw_enable_experimental_persistence": True},
        )

        self.chunker = Chunker(
            max_chunks=max_chunks,
            max_chunk_size=max_chunk_size,
            max_sentences=max_sentences,
        )

        self.use_altLabels = use_altLabels

        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = embedding_dimensions
        self.embedding_batch_size = embedding_batch_size

        self.max_query_hits = max_query_hits
        self.query_top_k = query_top_k
        self.query_jobs = query_jobs

        self.train_shrinkage = train_shrinkage
        self.train_interaction_depth = train_interaction_depth
        self.train_subsample = train_subsample
        self.train_rounds = train_rounds
        self.train_verbosity = train_verbosity
        self.train_jobs = train_jobs

        self.model = None

    def create_vector_db(
        self,
        vocab_in_path: Path,
        vocab_out_path: Path | None,
        collection_name: str = "my_collection",
        force: bool = False,
    ) -> None:
        collection_df = prepare_data.add_vocab_embeddings(
            vocab=prepare_data.parse_vocab(
                vocab_path=vocab_in_path,
                use_altLabels=self.use_altLabels,
            ),
            model_name=self.embedding_model_name,
            embedding_dimensions=self.embedding_dimensions,
            batch_size=self.embedding_batch_size,
        )

        if vocab_out_path:
            collection_df.write_ipc(vocab_out_path)

        self.client.create_collection(
            collection_df=collection_df,
            collection_name=collection_name,
            embedding_dimensions=self.embedding_dimensions,
            hnsw_index_name="hnsw_index",
            hnsw_metric="cosine",
            force=force,
        )

    def _prepare_train_data(
        self,
        texts: list[str],
        doc_ids: list[int],
        collection_name: str,
    ) -> pl.DataFrame:
        candidates_dfs = []

        for text, doc_id in zip(texts, doc_ids):
            candidates_dfs.append(
                self.generate_candidates(
                    text=text,
                    doc_id=doc_id,
                    collection_name=collection_name,
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
        )

    def _read_long_document_format(
        self,
        path_to_document_file: Path,
        path_to_index_file: Path,
    ) -> pl.DataFrame:
        documents = (
            pl.read_csv(
                path_to_document_file,
                has_header=False,
                separator="\t",
            )
            .with_row_index()
            .with_columns(pl.col("column_2").str.split(" "))
            .explode("column_2")
        )

        index = pl.read_ipc(path_to_index_file)

        return (
            documents.join(
                other=index,
                left_on="index",
                right_on="location",
                how="inner",
            )
            .select(["column_1", "column_2", "idn"])
            .rename({"column_1": "text", "column_2": "label_id"})
        )

    def prepare_train_from_docs(
        self,
        path_to_document_file: Path,
        path_to_index_file: Path,
        collection_name: str = "my_collection",
    ) -> pl.DataFrame:
        try:
            documents = self._read_long_document_format(
                path_to_document_file,
                path_to_index_file,
            )
        except FileNotFoundError:
            return
        except PermissionError:
            return

        train_texts = documents.get_column("text").to_list()
        train_doc_ids = documents.get_column("idn").to_list()
        gold_label_ids = [
            label_id.split("/")[4][:-1]
            for label_id in documents.get_column("label_id").to_list()
        ]
        gold_doc_ids = documents.get_column("idn").to_list()

        train_candidates = self._prepare_train_data(
            texts=train_texts,
            doc_ids=train_doc_ids,
            collection_name=collection_name,
        )

        gold_standard = pl.DataFrame(
            {
                "doc_id": gold_doc_ids,
                "label_id": gold_label_ids,
            }
        )

        return (
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
                    "gold",
                ]
            )
        )

    def prepare_train(
        self,
        train_texts: list[str],
        train_doc_ids: list[str],
        collection_name: str,
        gold_doc_ids: list[str],
        gold_label_ids: list[str],
    ) -> pl.DataFrame:
        train_candidates = self._prepare_train_data(
            texts=train_texts,
            doc_ids=train_doc_ids,
            collection_name=collection_name,
        )

        gold_standard = pl.DataFrame(
            {
                "doc_id": gold_doc_ids,
                "label_id": gold_label_ids,
            }
        )

        return (
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
                    "gold",
                ]
            )
        )

    def train(self, train_data: pl.DataFrame) -> None:
        xgb_matrix = train_data.select(
            [
                "score",
                "occurrences",
                "min_cosine_similarity",
                "max_cosine_similarity",
                "first_occurence",
                "last_occurence",
                "spread",
                "is_prefLabel",
            ]
        )

        matrix = xgb.DMatrix(xgb_matrix.to_pandas(), train_data.to_pandas()["gold"])

        model = xgb.train(
            params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": self.train_shrinkage,
                "max_depth": self.train_interaction_depth,
                "subsample": self.train_subsample,
                "verbosity": self.train_verbosity,
                "nthread": self.train_jobs,
            },
            dtrain=matrix,
            num_boost_round=self.train_rounds,
            evals=[(matrix, "train")],
        )

        self.model = model

    def generate_candidates(
        self,
        text: str,
        doc_id: int,
        collection_name: str,
    ) -> pl.DataFrame:
        embedding_generator = EmbeddingGenerator(
            model_name=self.embedding_model_name,
            embedding_dimensions=self.embedding_dimensions,
        )

        text_chunks = self.chunker.chunk_text(text)

        query_df = pl.DataFrame(
            {
                "id": [i + 1 for i in range(len(text_chunks))],
                "doc_id": [doc_id for _ in range(len(text_chunks))],
                "chunk_position": [i + 1 for i in range(len(text_chunks))],
                "n_chunks": [len(text_chunks) for _ in range(len(text_chunks))],
                "embeddings": embedding_generator.generate_embeddings(
                    texts=text_chunks,
                    batch_size=self.embedding_batch_size,
                    task="retrieval.passage",
                ),
            }
        )

        return self.client.vector_search(
            query_df=query_df,
            collection_name=collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=self.query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

    def generate_candidates_batch(
        self,
        texts: list[str],
        doc_ids: list[int],
        collection_name: str,
    ):
        embedding_generator = EmbeddingGenerator(
            model_name=self.embedding_model_name,
            embedding_dimensions=self.embedding_dimensions,
        )

        text_chunks = []
        for text in texts:
            text_chunks.append(self.chunker.chunk_text(text))

        query_dfs = []
        id_count = 1
        for doc_id, chunks in zip(doc_ids, text_chunks):
            query_dfs.append(
                pl.DataFrame(
                    {
                        "id": [i + id_count for i in range(len(chunks))],
                        "doc_id": [doc_id for _ in range(len(chunks))],
                        "chunk_position": [i + 1 for i in range(len(chunks))],
                        "n_chunks": [len(chunks) for _ in range(len(chunks))],
                        "embeddings": embedding_generator.generate_embeddings(
                            texts=chunks,
                            batch_size=self.embedding_batch_size,
                            task="retrieval.passage",
                        ),
                    }
                )
            )

            id_count += len(chunks)

        return self.client.vector_search(
            query_df=pl.concat(query_dfs),
            collection_name=collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=self.query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

    def predict(self, candidates: pl.DataFrame) -> pl.DataFrame:
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
                ]
            )
        )

        return (
            candidates.with_columns(
                pl.Series(self.model.predict(matrix)).alias("score")
            )
            .select(["doc_id", "label_id", "score"])
            .sort(["doc_id", "score"], descending=[False, True])
            .group_by("doc_id")
            .agg(pl.all().head(self.query_top_k))
            .explode(["label_id", "score"])
        )

    def load(self, input_path: Path) -> None:
        if not self.model:
            try:
                self.model = pickle.load(open(input_path, "rb"))
            except FileNotFoundError:
                return
            except PermissionError:
                return

    def save(self, output_path: Path, force: bool = False) -> None:
        if output_path.exists() and not force:
            return
        else:
            try:
                pickle.dump(self.model, open(output_path, "wb"))
            except FileNotFoundError:
                return
            except PermissionError:
                return
