from threading import Thread

import duckdb
import polars as pl


class Duckdb_client:
    def __init__(
        self,
        db_path: str,
        config: dict = {},
    ) -> None:
        self.connection = duckdb.connect(
            database=db_path,
            config=config,
        )

        self.connection.install_extension("vss")
        self.connection.load_extension("vss")

    def create_collection(
        self,
        collection_df: pl.DataFrame,
        collection_name: str = "my_collection",
        embedding_dimensions: int = 1024,
        hnsw_index_name: str = "hnsw_index",
        hnsw_metric: str = "cosine",
        force: bool = False,
    ):
        replace = ""
        if force:
            replace = "OR REPLACE "

        self.connection.execute(
            f"""CREATE {replace}TABLE {collection_name} (
                id INTEGER,
                doc_id VARCHAR,
                label_text VARCHAR,
                is_prefLabel BOOLEAN,
                embeddings FLOAT[{embedding_dimensions}])"""
        )

        self.connection.execute(
            f"INSERT INTO {collection_name} BY NAME SELECT * FROM collection_df"
        )

        if force:
            self.connection.execute(f"DROP INDEX IF EXISTS {hnsw_index_name}")
        self.connection.execute(
            f"""CREATE INDEX IF NOT EXISTS {hnsw_index_name}
            ON {collection_name}
            USING HNSW (embeddings)
            WITH (metric = '{hnsw_metric}')"""
        )

    def vector_search(
        self,
        query_df: pl.DataFrame,
        collection_name: str,
        embedding_dimensions: int,
        n_jobs: int = 1,
        n_hits: int = 10,
        chunk_size: int = 2048,
        top_k: int = 10,
        hnsw_metric_function: str = "array_cosine_distance",
    ) -> pl.DataFrame:
        query_dfs = [
            query_df.slice(i, chunk_size) for i in range(0, query_df.height, chunk_size)
        ]
        batches = [query_dfs[i : i + n_jobs] for i in range(0, len(query_dfs), n_jobs)]

        self.connection.execute("""CREATE OR REPLACE TABLE results ( 
                                id INTEGER,
                                doc_id VARCHAR,
                                chunk_position INTEGER,
                                n_chunks INTEGER,
                                label_id VARCHAR,
                                is_prefLabel BOOLEAN,
                                score FLOAT)""")

        for batch in batches:
            threads = []
            for df in batch:
                threads.append(
                    Thread(
                        target=self._vss_thread_query,
                        args=(
                            df,
                            collection_name,
                            embedding_dimensions,
                            hnsw_metric_function,
                            n_hits,
                        ),
                    )
                )

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        result_df = self.connection.execute("SELECT * FROM results").pl()

        result_df = (
            result_df.group_by("id")
            .agg(
                doc_id=pl.col("doc_id").first(),
                chunk_position=pl.col("chunk_position").first(),
                n_chunks=pl.col("n_chunks").first(),
                label_id=pl.col("label_id"),
                is_prefLabel=pl.col("is_prefLabel"),
                score=pl.col("score") / pl.col("score").max(),
            )
            .explode(["label_id", "is_prefLabel", "score"])
        )

        result_df = (
            result_df.sort("score", descending=True)
            .group_by(["id", "label_id", "doc_id"])
            .head(1)
        )

        result_df = result_df.group_by(["doc_id", "label_id"]).agg(
            score=pl.col("score").sum(),
            occurrences=pl.col("doc_id").count(),
            min_cosine_similarity=pl.col("score").min(),
            max_cosine_similarity=pl.col("score").max(),
            first_occurence=pl.col("chunk_position").min(),
            last_occurence=pl.col("chunk_position").max(),
            spread=(pl.col("chunk_position").max() - pl.col("chunk_position").min()),
            is_prefLabel=pl.col("is_prefLabel").first(),
            n_chunks=pl.col("n_chunks").first(),
        )

        result_df = (
            result_df.sort("score", descending=True).group_by("doc_id").head(top_k)
        )

        return (
            result_df.with_columns(
                (pl.col("score") / pl.col("n_chunks")),
                (pl.col("occurrences") / pl.col("n_chunks")),
                (pl.col("first_occurence") / pl.col("n_chunks")),
                (pl.col("last_occurence") / pl.col("n_chunks")),
                (pl.col("spread") / pl.col("n_chunks")),
            )
            .drop("n_chunks")
            .sort(["doc_id", "label_id"])
        )

    def _vss_thread_query(
        self,
        queries_df: pl.DataFrame,
        collection_name: str,
        vector_dimensions: int,
        hnsw_metric_function: str = "array_cosine_distance",
        limit: int = 10,
    ):
        thread_connection = self.connection.cursor()

        thread_connection.execute(
            f"""CREATE OR REPLACE TEMP TABLE queries ( 
                query_id INTEGER,
                query_doc_id VARCHAR,
                chunk_position INTEGER,
                n_chunks INTEGER,
                embeddings FLOAT[{vector_dimensions}])"""
        )

        thread_connection.execute(
            "INSERT INTO queries BY NAME SELECT * FROM queries_df"
        )

        thread_connection.execute(
            f"""INSERT INTO results
            SELECT queries.query_id, 
            queries.query_doc_id,
            queries.chunk_position,
            queries.n_chunks,
            doc_id AS label_id,
            is_prefLabel,
            (1 - score) AS score,
            FROM queries, LATERAL (
                SELECT {collection_name}.doc_id,
                {collection_name}.is_prefLabel,
                {hnsw_metric_function}(queries.embeddings, {collection_name}.embeddings) AS score
                FROM {collection_name}
                ORDER BY score
                LIMIT {limit}
            )"""
        )
