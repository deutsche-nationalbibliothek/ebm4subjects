from threading import Thread
import duckdb
import pandas as pd
from tqdm import tqdm


class Duckdb_client:
    def __init__(
        self,
        db_path: str,
        read_only: bool = False,
        config: dict = {},
    ) -> None:
        self.connection = duckdb.connect(
            database=db_path,
            read_only=read_only,
            config=config,
        )

        self.connection.install_extension("vss")
        self.connection.load_extension("vss")
        self.connection.install_extension("fts")
        self.connection.load_extension("fts")

    def create_collection(
        self,
        collection_df: pd.DataFrame,
        collection_name: str = "gnd",
        vector_dimensions: int = 1024,
        hnsw_index_name: str = "hnsw_index",
        hnsw_metric: str = "l2sq",
        fts_config: dict = {},
        force: bool = False,
    ):
        if isinstance(force, str):
            force = force == "true" or force == "True"

        replace = ""
        if force:
            replace = "OR REPLACE "

        print(f"Creating/overwriting collection {collection_name}")
        self.connection.execute(
            f"""CREATE {replace}TABLE {collection_name} (
                id INTEGER,
                idn VARCHAR,
                label_text VARCHAR,
                is_prefLabel BOOLEAN,
                embeddings FLOAT[{vector_dimensions}])"""
        )

        print(f"Inserting vocabulary into {collection_name}")
        self.connection.execute(
            f"INSERT INTO {collection_name} BY NAME SELECT * FROM collection_df"
        )

        print("Creating/overwriting HNSW-index")
        if force:
            self.connection.execute(f"DROP INDEX IF EXISTS {hnsw_index_name}")
        self.connection.execute(
            f"""CREATE INDEX IF NOT EXISTS {hnsw_index_name}
            ON {collection_name}
            USING HNSW (embeddings)
            WITH (metric = '{hnsw_metric}')"""
        )

        print("Creating/overwriting FTS-index")
        self.connection.execute(f"""PRAGMA create_fts_index(
                                '{collection_name}', 
                                'id', 
                                'label_text',
                                overwrite={int(force)},
                                stemmer='{fts_config.get("stemmer", "porter")}',
                                stopwords='{fts_config.get("stopwords", "none")}',
                                ignore='{fts_config.get("ignore", "[^a-z]+")}',
                                strip_accents={fts_config.get("strip_accents", 1)},
                                lower={fts_config.get("lower", 1)})""")

    def vector_search(
        self,
        query_df: pd.DataFrame,
        n_chunks_df: pd.DataFrame,
        collection_name: str,
        vector_dimensions: int,
        n_jobs: int = 1,
        n_hits: int = 10,
        chunk_size: int = 2048,
        top_k: int = 10,
        hnsw_metric_function: str = "array_distance",
    ):
        query_dfs = [
            query_df[i : i + chunk_size]
            for i in range(0, query_df.shape[0], chunk_size)
        ]

        batches = [query_dfs[i : i + n_jobs] for i in range(0, len(query_dfs), n_jobs)]

        result_dfs = []
        for batch_number, batch in enumerate(tqdm(batches, desc="Processing batches")):
            threads = []
            for thread_number, df in enumerate(batch):
                threads.append(
                    Thread(
                        target=self.__vss_thread_query,
                        args=(
                            df,
                            result_dfs,
                            collection_name,
                            vector_dimensions,
                            hnsw_metric_function,
                            n_hits,
                            thread_number
                        ),
                    )
                )

            for thread in threads:
                thread.start()
            for thread in tqdm(threads, desc=f"Processing batch {batch_number + 1}"):
                thread.join()

        result_df = pd.concat(result_dfs, ignore_index=True)

        result_df = (
            result_df.explode(
                ["is_prefLabels", "label_texts", "vss_matches", "vss_scores"]
            )
            .reset_index(drop=True)
            .rename(
                columns={
                    "is_prefLabels": "is_prefLabel",
                    "label_texts": "label_text",
                    "vss_matches": "label_id",
                    "vss_scores": "score",
                }
            )
        )
        result_df["score"] = result_df.groupby("id")["score"].transform(
            lambda x: x / x.max()
        )

        result_df = (
            result_df.sort_values("score", ascending=False)
            .groupby(["id", "label_id", "doc_id"])
            .head(1)
        )

        result_df = (
            result_df.groupby(["doc_id", "label_id"])
            .agg(
                score=("score", "sum"),
                occurrences=("doc_id", "count"),
                first_occurence=("chunk_position", "min"),
                last_occurence=("chunk_position", "max"),
                spread=("chunk_position", lambda x: x.max() - x.min()),
                is_prefLabel=("is_prefLabel", "any"),
            )
            .reset_index()
        )
        result_df = (
            result_df.sort_values(["score"], ascending=False)
            .groupby("doc_id")
            .head(top_k)
        )
        result_df = pd.merge(result_df, n_chunks_df, on="doc_id", how="left")
        result_df["score"] = result_df["score"] / result_df["n_chunks"]
        result_df["first_occurence"] = (
            result_df["first_occurence"] / result_df["n_chunks"]
        )
        result_df["last_occurence"] = (
            result_df["last_occurence"] / result_df["n_chunks"]
        )
        result_df["spread"] = result_df["spread"] / result_df["n_chunks"]
        result_df = result_df.drop("n_chunks", axis=1)

        return result_df

    def hybrid_search(
        self,
        query_df: pd.DataFrame,
        n_chunks_df: pd.DataFrame,
        collection_name: str,
        vector_dimensions: int,
        n_jobs: int = 1,
        n_hits: int = 10,
        alpha: float = 0.5,
        chunk_size: int = 2048,
        top_k: int = 10,
        hnsw_metric_function: str = "array_distance",
    ):
        query_dfs = [
            query_df[i : i + chunk_size]
            for i in range(0, query_df.shape[0], chunk_size)
        ]

        result_dfs = []
        for df in tqdm(query_dfs, desc="Processing batches"):
            result_dfs.append(
                self.hybrid_query(
                    query_df=df,
                    collection_name=collection_name,
                    vector_dimensions=vector_dimensions,
                    n_jobs=n_jobs,
                    n_hits=n_hits,
                    alpha=alpha,
                    hnsw_metric_function=hnsw_metric_function,
                )
            )

        result = pd.concat(result_dfs, ignore_index=True)
        result = (
            result.groupby(["doc_id", "label_id"])
            .agg(
                score=("score", "sum"),
                occurrences=("doc_id", "count"),
                first_occurence=("chunk_position", "min"),
                last_occurence=("chunk_position", "max"),
                spread=("chunk_position", lambda x: x.max() - x.min()),
                is_prefLabel=("is_prefLabel", "any"),
            )
            .reset_index()
        )
        result = (
            result.sort_values(["score"], ascending=False).groupby("doc_id").head(top_k)
        )
        result = pd.merge(result, n_chunks_df, on="doc_id", how="left")
        result["score"] = result["score"] / result["n_chunks"]
        result["first_occurence"] = result["first_occurence"] / result["n_chunks"]
        result["last_occurence"] = result["last_occurence"] / result["n_chunks"]
        result["spread"] = result["spread"] / result["n_chunks"]
        result = result.drop("n_chunks", axis=1)

        return result

    def hybrid_query(
        self,
        query_df: pd.DataFrame,
        collection_name: str,
        vector_dimensions: int,
        n_jobs: int = 1,
        n_hits: int = 10,
        alpha: float = 0.5,
        hnsw_metric_function: str = "array_distance",
    ):
        result_list_fts = []
        result_list_vss = []

        query_ids = query_df["id"].tolist()
        query_doc_ids = query_df["doc_id"].tolist()
        query_chunk_positions = query_df["chunk_position"].tolist()
        query_texts = query_df["text"].tolist()

        chunk_size = max(int(len(query_ids) / n_jobs), 1)
        chunked_query_ids = [
            query_ids[i : i + chunk_size] for i in range(0, len(query_ids), chunk_size)
        ]
        chunked_doc_ids = [
            query_doc_ids[i : i + chunk_size]
            for i in range(0, len(query_doc_ids), chunk_size)
        ]
        chunked_chunk_positions = [
            query_chunk_positions[i : i + chunk_size]
            for i in range(0, len(query_chunk_positions), chunk_size)
        ]
        chunked_query_texts = [
            query_texts[i : i + chunk_size]
            for i in range(0, len(query_texts), chunk_size)
        ]

        threads = [
            Thread(
                target=self.__vss_thread_query,
                args=(
                    query_df,
                    result_list_vss,
                    collection_name,
                    vector_dimensions,
                    hnsw_metric_function,
                    n_hits,
                ),
            )
        ]
        for ids, doc_ids, chunk_positions, texts in zip(
            chunked_query_ids,
            chunked_doc_ids,
            chunked_chunk_positions,
            chunked_query_texts,
        ):
            threads.append(
                Thread(
                    target=self.__fts_thread_query,
                    args=(
                        collection_name,
                        ids,
                        doc_ids,
                        chunk_positions,
                        texts,
                        result_list_fts,
                        n_hits,
                    ),
                )
            )
            pass

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        results_fts = pd.concat(result_list_fts, ignore_index=True)

        results_vss = (
            result_list_vss[0]
            .explode(["is_prefLabels", "label_texts", "vss_matches", "vss_scores"])
            .reset_index(drop=True)
            .rename(
                columns={
                    "is_prefLabels": "is_prefLabel",
                    "label_texts": "label_text",
                    "vss_matches": "match",
                    "vss_scores": "vss_score",
                }
            )
        )

        results_vss["vss_score"] = results_vss.groupby("id")["vss_score"].transform(
            lambda x: x / x.max()
        )
        results_fts["fts_score"] = results_fts.groupby("id")["fts_score"].transform(
            lambda x: x / x.max()
        )

        results = (
            results_vss.join(
                other=results_fts.set_index(
                    [
                        "id",
                        "doc_id",
                        "chunk_position",
                        "match",
                        "is_prefLabel",
                        "label_text",
                    ]
                ),
                on=[
                    "id",
                    "doc_id",
                    "chunk_position",
                    "match",
                    "is_prefLabel",
                    "label_text",
                ],
                how="outer",
            )
            .infer_objects(copy=False)
            .fillna(0)
            .rename(columns={"match": "label_id"})
        )

        results["score"] = results["vss_score"] * alpha + results["fts_score"] * (
            1 - alpha
        )
        results = (
            results.sort_values("score", ascending=False)
            .groupby(["id", "label_id", "doc_id"])
            .head(1)
        )

        return results

    def __vss_thread_query(
        self,
        queries_df: pd.DataFrame,
        results: list,
        collection_name: str,
        vector_dimensions: int,
        hnsw_metric_function: str = "array_distance",
        limit: int = 10,
        thread_number: int = 0,
    ):
        thread_connection = self.connection.cursor()

        thread_connection.execute(
            f"""CREATE OR REPLACE TABLE queries{thread_number} ( 
                id INTEGER,
                doc_id VARCHAR,
                chunk_position INTEGER,
                text VARCHAR,
                embeddings FLOAT[{vector_dimensions}])"""
        )

        thread_connection.execute(
            f"INSERT INTO queries{thread_number} BY NAME SELECT * FROM queries_df"
        )

        result = thread_connection.execute(
            f"""SELECT queries{thread_number}.id, 
            queries{thread_number}.doc_id,
            queries{thread_number}.chunk_position,
            list(idn) AS vss_matches,
            list(is_prefLabel) AS is_prefLabels,
            list(label_text) AS label_texts,
            list(vss_score) AS vss_scores 
            FROM queries{thread_number}, LATERAL (
                SELECT {collection_name}.idn,
                {collection_name}.is_prefLabel,
                {collection_name}.label_text,
                {hnsw_metric_function}(queries{thread_number}.embeddings, {collection_name}.embeddings) AS vss_score
                FROM {collection_name}
                ORDER BY vss_score
                LIMIT {limit}
            )
            GROUP BY id, doc_id, chunk_position"""
        ).df()

        results.append(result)

    def __fts_thread_query(
        self,
        collection_name,
        query_ids: list[int],
        doc_ids: list[str],
        chunk_positions: list[int],
        query_texts: list[str],
        results: list,
        limit: int = 10,
    ):
        thread_connection = self.connection.cursor()

        thread_connection.execute(f"""PREPARE query AS (
                                  WITH scored_docs AS (
                                    SELECT *, fts_main_{collection_name}.match_bm25(id, ?, fields := 'label_text') AS fts_score 
                                    FROM {collection_name}
                                  )
                                  SELECT idn AS match, is_prefLabel, label_text, fts_score
                                  FROM scored_docs
                                  WHERE fts_score IS NOT NULL
                                  ORDER BY fts_score DESC
                                  LIMIT {limit})
                                  """)

        for query_id, doc_id, chunk_position, query_text in zip(
            query_ids, doc_ids, chunk_positions, query_texts
        ):
            query_text = query_text.replace("'", "")
            query_text = query_text.replace('"', "")

            result = thread_connection.execute(f"EXECUTE query('{query_text}')").df()

            result["id"] = query_id
            result["doc_id"] = doc_id
            result["chunk_position"] = chunk_position

            results.append(result)
