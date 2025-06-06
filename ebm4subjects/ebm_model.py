from pathlib import Path
import pickle

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
        candidates = candidates.with_columns(pl.lit(True).alias("suggested"))
        gold_standard = gold_standard.with_columns(pl.lit(True).alias("gold"))

        comparison = candidates.join(
            other=gold_standard,
            on=["doc_id", "label_id"],
            how="outer",
        ).with_columns(
            pl.col("suggested").fill_null(False),
            pl.col("gold").fill_null(False),
        )

        return comparison

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

        train_data = (
            self._compare_to_gold_standard(train_candidates, gold_standard)
            .with_columns(pl.when(pl.col("gold")).then(1).otherwise(0).alias("gold"))
            .filter(pl.col("doc_id").is_not_null())
            .select(
                [
                    "score",
                    "occurrences",
                    "first_occurence",
                    "last_occurence",
                    "spread",
                    "is_prefLabel",
                    "gold",
                ]
            )
        )

        return train_data

    def train(self, train_data: pl.DataFrame) -> xgb.Booster:
        xgb_matrix = train_data.select(
            [
                "score",
                "occurrences",
                "first_occurence",
                "last_occurence",
                "spread",
                "is_prefLabel",
            ]
        )

        dtrain = xgb.DMatrix(xgb_matrix.to_pandas(), train_data.to_pandas()["gold"])

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
            dtrain=dtrain,
            num_boost_round=self.train_rounds,
            evals=[(dtrain, "train")],
        )

        return model

    def generate_candidates(
        self,
        text: str,
        doc_id: int,
        collection_name: str,
    ) -> pl.DataFrame:
        text_chunks = self.chunker.chunk_text(text)

        embedding_generator = EmbeddingGenerator(
            model_name=self.embedding_model_name,
            embedding_dimensions=self.embedding_dimensions,
        )

        query_df = pl.DataFrame(
            {
                "id": [i + 1 for i in range(len(text_chunks))],
                "doc_id": [doc_id for i in range(len(text_chunks))],
                "chunk_position": [i + 1 for i in range(len(text_chunks))],
                "n_chunks": [len(text_chunks) for _ in range(len(text_chunks))],
                "embeddings": embedding_generator.generate_embeddings(
                    texts=text_chunks,
                    batch_size=self.embedding_batch_size,
                    task="retrieval.passage",
                ),
            }
        )

        candidates_df = self.client.vector_search(
            query_df=query_df,
            collection_name=collection_name,
            embedding_dimensions=self.embedding_dimensions,
            n_jobs=self.query_jobs,
            n_hits=self.max_query_hits,
            chunk_size=1024,
            top_k=self.query_top_k,
            hnsw_metric_function="array_cosine_distance",
        )

        return candidates_df

    def predict(self):
        pass

    def load(self, input_path: Path) -> xgb.Booster:
        return pickle.load(input_path)

    def save(self, model:xgb.Booster, output_path: Path) -> None:
        pickle.dump(model, open(output_path, "wb"))
