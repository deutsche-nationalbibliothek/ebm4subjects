from threading import Thread
from typing import Any

import duckdb
import polars as pl


def vss_thread_query(
    connection: duckdb.DuckDBPyConnection,
    queries: dict[str, Any],
    collection_name: str,
    vector_dimensions: int,
    hnsw_metric_function: str = "array_cosine_distance",
    ef_search: int = 256,
    limit: int = 100,
):
    """
    A helper function for performing the vector search in parallel.

    Args:
        connection (duckdb.DuckDBPyConnection): The DuckDB connection object.
        queries (dict[str, Any]): The queries to be searched against the collection.
        collection_name (str): The name of the collection to search against.
        vector_dimensions (int): The number of dimensions for the
            vector embeddings.
        hnsw_metric_function (str, optional): The metric function to use for the
            HNSW index (default: "array_cosine_distance").
        ef_search (int, optional): The enter factor of search for the HNSW index
        limit (int, optional): The number of hits to return per document
            (default: 100).
    """
    # Create a temporary connection for the thread
    with connection.cursor() as thread_connection:
        # Convert the queries dictionary to a Polars DataFrame
        queries_df = pl.DataFrame(queries)  # noqa: F841

        # Create a temporary table to store the search results
        thread_connection.execute(
            f"""CREATE OR REPLACE TEMP TABLE queries ( 
                    query_id INTEGER,
                    query_doc_id VARCHAR,
                    chunk_position INTEGER,
                    n_chunks INTEGER,
                    embeddings FLOAT[{vector_dimensions}])"""
        )

        # Insert the data into the temporary table
        thread_connection.execute(
            "INSERT INTO queries BY NAME SELECT * FROM queries_df"
        )

        # apply oversearch to reduce sensitivity in MinMax scaling
        limit = max(limit, 100)

        # Set the HNSW index parameters for search
        thread_connection.execute(f"SET hnsw_ef_search = {ef_search}")

        # Perform the vector search
        thread_connection.execute(
            f"""INSERT INTO results
                SELECT queries.query_id, 
                queries.query_doc_id,
                queries.chunk_position,
                queries.n_chunks,
                label_id,
                is_prefLabel,
                (1 - intermed_score) AS score,
                FROM queries, LATERAL (
                    SELECT {collection_name}.label_id,
                    {collection_name}.is_prefLabel,
                    {hnsw_metric_function}(queries.embeddings, {collection_name}.embeddings) AS intermed_score
                    FROM {collection_name}
                    ORDER BY intermed_score
                    LIMIT {limit}
                )"""
        )


class Duckdb_client:
    """
    A class for interacting with a DuckDB database,
    specifically for creating and querying vector search indexes.

    Attributes:
        - db_config (dict): Configuration options for the DuckDB connection.
        - db_path (str): The path to the DuckDB database.
        - hnsw_index_params (dict): Parameters for the HNSW index,
            including the number of clusters (M),
            the enter factor of construction (ef_construction),
            and the enter factor of search (ef_search).
    Methods:
        - create_collection: Creates a new collection in the DuckDB database
        - vector_search: Performs a vector search on the specified collection
    """

    def __init__(
        self,
        db_path: str,
        config: dict[str, Any] | None = None,
        hnsw_index_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the Duckdb_client.

        Args:
            db_path (str): The path to the DuckDB database.
            config (dict, optional): Configuration options for the DuckDB connection
                (default: {"hnsw_enable_experimental_persistence": True, "threads": 1).
            hnsw_index_params (dict, optional): Parameters for the HNSW index
                (default: {"M": 32, "ef_construction": 256, "ef_search": 256}).

        Notes:
            'hnsw_enable_experimental_persistence' needs to be set to 'True' in order
            to store and query the index later
        """
        if hnsw_index_params is None:
            hnsw_index_params = {"M": 32, "ef_construction": 256, "ef_search": 256}
        if config is None:
            config = {"hnsw_enable_experimental_persistence": True, "threads": 1}

        self.db_path = db_path
        self.db_config = config

        # Install and load the vss extension for DuckDB
        with duckdb.connect(database=self.db_path, config=self.db_config) as connection:
            connection.install_extension("vss")
            connection.load_extension("vss")

        self.hnsw_index_params = hnsw_index_params

    def create_collection(
        self,
        collection_df: pl.DataFrame,
        collection_name: str = "my_collection",
        embedding_dimensions: int = 1024,
        hnsw_index_name: str = "hnsw_index",
        hnsw_metric: str = "cosine",
        force: bool = False,
    ):
        """
        Creates a new collection in the DuckDB database and indexes it
        using the HNSW algorithm.

        Args:
            collection_df (pl.DataFrame): The data to be inserted into the collection.
            collection_name (str, optional): The name of the collection
                (default: "my_collection").
            embedding_dimensions (int, optional): The number of dimensions for the
                vector embeddings (default: 1024).
            hnsw_index_name (str, optional): The name of the HNSW index
                (default: "hnsw_index").
            hnsw_metric (str, optional): The metric to be used for the HNSW index
                (default: "cosine")
            force (bool, optional): Whether to replace the existing collection if it
                already exists (default: False).

        Notes:
            If 'hnsw_metric' is changed in this function 'hnsw_metric_function' in
            the vector_search function needs to be changed accordingly in order
            for the index to work properly.
        """
        with duckdb.connect(database=self.db_path, config=self.db_config) as connection:
            # Determine whether to replace the existing collection
            replace = ""
            if force:
                replace = "OR REPLACE "

            # Create the collection table
            connection.execute(
                f"""CREATE {replace}TABLE {collection_name} (
                    id INTEGER,
                    label_id VARCHAR,
                    label_text VARCHAR,
                    is_prefLabel BOOLEAN,
                    embeddings FLOAT[{embedding_dimensions}])"""
            )

            # Insert the data into the collection table
            connection.execute(
                f"INSERT INTO {collection_name} BY NAME SELECT * FROM collection_df"
            )

            # Create the HNSW index
            if force:
                # Drop the existing index if it exists
                connection.execute(f"DROP INDEX IF EXISTS {hnsw_index_name}")
            connection.execute(
                f"""CREATE INDEX IF NOT EXISTS {hnsw_index_name}
                ON {collection_name}
                USING HNSW (embeddings)
                WITH (metric = '{hnsw_metric}', M = {self.hnsw_index_params["M"]}, ef_construction = {self.hnsw_index_params["ef_construction"]})"""
            )

    def vector_search(
        self,
        query_df: pl.DataFrame,
        collection_name: str,
        embedding_dimensions: int,
        n_jobs: int = 1,
        n_hits: int = 100,
        chunk_size: int = 2048,
        top_k: int = 10,
        hnsw_metric_function: str = "array_cosine_distance",
    ) -> pl.DataFrame:
        """
        Performs a vector search on the specified collection using the HNSW index.

        Args:
            query_df (pl.DataFrame): The data to be searched against the collection.
            collection_name (str): The name of the collection to search against.
            embedding_dimensions (int): The number of dimensions for the
                vector embeddings.
            n_jobs (int, optional): The number of jobs to use for
                parallel processing (default: 1).
            n_hits (int, optional): The number of hits to return per document
                (default: 100).
            chunk_size (int, optional): The size of each chunk for
                parallel processing (default: 2048).
            top_k (int, optional): The number of top-k suggestions to return
                per document (default: 10).
            hnsw_metric_function (str, optional): The metric function to use for
                the HNSW index (default:  "array_cosine_distance").

        Returns:
            pl.DataFrame: The result of the vector search.

        Notes:
            If 'hnsw_metric_function' is changed in this function 'hnsw_metric' in
            the create_collection function needs to be changed accordingly in order
            for the index to work properly.
            The argument 'chunk_size' is already set to the optimal value for the
            query processing with DuckDB. Only change it if necessary.
        """
        with duckdb.connect(database=self.db_path, config=self.db_config) as connection:
            # Create a temporary table to store the search results
            connection.execute("""CREATE OR REPLACE TABLE results ( 
                                    id INTEGER,
                                    doc_id VARCHAR,
                                    chunk_position INTEGER,
                                    n_chunks INTEGER,
                                    label_id VARCHAR,
                                    is_prefLabel BOOLEAN,
                                    score FLOAT)""")

            # Split the query data into chunks for parallel processing
            query_dfs = [
                query_df.slice(i, chunk_size) for i in range(0, query_df.height, chunk_size)
            ]
            batches = [query_dfs[i : i + n_jobs] for i in range(0, len(query_dfs), n_jobs)]

            # Perform the vector search in parallel
            for batch in batches:
                threads: list[Thread] = []
                for df in batch:
                    threads.append(
                        Thread(
                            target=vss_thread_query,
                            args=(
                                connection,
                                df.to_dict(as_series=False),
                                collection_name,
                                embedding_dimensions,
                                hnsw_metric_function,
                                self.hnsw_index_params["ef_search"],
                                n_hits,
                            ),
                        )
                    )

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            # Retrieve the search results
            result_df = connection.execute("SELECT * FROM results").pl()

            # Apply MinMax scaling to the 'score' column per 'id'
            # and keep n_hits results
            result_df = (
                result_df.group_by("id")
                .agg(
                    doc_id=pl.col("doc_id").first(),
                    chunk_position=pl.col("chunk_position").first(),
                    n_chunks=pl.col("n_chunks").first(),
                    label_id=pl.col("label_id"),
                    is_prefLabel=pl.col("is_prefLabel"),
                    cosine_similarity=pl.col("score"),
                    max_score=pl.col("score").max(),
                    min_score=pl.col("score").min(),
                    score=pl.col("score"),
                )
                .explode(["label_id", "is_prefLabel", "cosine_similarity", "score"])
                .with_columns(
                    [
                        (
                            (pl.col("score") - pl.col("min_score"))
                            / (pl.col("max_score") - pl.col("min_score") + 1e-9)
                        ).alias("score")
                    ]
                )
                .drop("min_score", "max_score")
                .sort("score", descending=True)
                .group_by("id")
                .head(n_hits)
            )

            # If a label is hit more then once due to altlabels
            # keep only the best hit
            result_df = (
                result_df.sort("score", descending=True)
                .group_by(["id", "label_id", "doc_id"])
                .head(1)
            )

            # Across chunks (queries) aggregate statistics for
            # each tuple 'doc_id', 'label_id'
            result_df = result_df.group_by(["doc_id", "label_id"]).agg(
                score=pl.col("score").sum(),
                occurrences=pl.col("doc_id").count(),
                min_cosine_similarity=pl.col("cosine_similarity").min(),
                max_cosine_similarity=pl.col("cosine_similarity").max(),
                first_occurrence=pl.col("chunk_position").min(),
                last_occurrence=pl.col("chunk_position").max(),
                spread=(pl.col("chunk_position").max() - pl.col("chunk_position").min()),
                is_prefLabel=pl.col("is_prefLabel").first(),
                n_chunks=pl.col("n_chunks").first(),
            )

            # keep only top_k suggestions per document
            result_df = (
                result_df.sort("score", descending=True).group_by("doc_id").head(top_k)
            )

            # Scale the results and return it
            return result_df.with_columns(
                (pl.col("score") / pl.col("n_chunks")),
                (pl.col("occurrences") / pl.col("n_chunks")),
                (pl.col("first_occurrence") / pl.col("n_chunks")),
                (pl.col("last_occurrence") / pl.col("n_chunks")),
                (pl.col("spread") / pl.col("n_chunks")),
            ).sort(["doc_id", "label_id"])
