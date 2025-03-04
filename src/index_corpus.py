import argparse
import logging

import numpy as np
import pandas as pd
from dvc.api import params_show

from duckdb_client import Duckdb_client
from tqdm import tqdm

params = params_show()
embedding_dim = params["general"]["embedding_dim"]
collection_name = params["vocab_config"]["collection_name"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument(
        "--db_path", type=str, default="gnd.duckdb", help="Path to DuckDB"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="results/test/chunks.txt",
        help="Text chunks to be indexed",
    )
    parser.add_argument(
        "--chunk_index",
        type=str,
        default="results/test/chunk_index.arrow",
        help="Index file containing a mapping of documents and chunks",
    )
    parser.add_argument(
        "--chunk_embeddings",
        type=str,
        default="results/test/chunk_embeddings.npy",
        help="Embeddings file matching the chunks and chunk_index",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for hybrid search"
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of top k results to keep"
    )
    parser.add_argument(
        "--n_hits",
        type=int,
        default=10,
        help="Number of hits to retrieve from Hybrid Search",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=20, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/test/candidates_new.arrow",
        help="Output file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    client = Duckdb_client(
        db_path=args.db_path,
        config={"hnsw_enable_experimental_persistence": True},
    )

    logger.info("Reading chunks...")
    with open(args.chunks, "r", encoding="utf-8") as file:
        chunk_texts = file.readlines()

    logger.info("Reading chunk index...")
    chunk_index = pd.read_feather(args.chunk_index)
    chunk_positions = chunk_index["rel_chunk_position"].tolist()
    doc_ids = chunk_index["doc_id"].tolist()
    n_chunks_df = (
        chunk_index.groupby("doc_id")
        .agg(n_chunks=("rel_chunk_position", "count"))
        .reset_index()
    )

    logger.info("Reading chunk embeddings...")
    chunk_embdeddings = np.load(args.chunk_embeddings)

    input_df = pd.DataFrame()
    input_df["id"] = [i for i in range(len(chunk_texts))]
    input_df["doc_id"] = doc_ids
    input_df["chunk_position"] = chunk_positions
    input_df["text"] = chunk_texts
    input_df["embeddings"] = chunk_embdeddings[0 : len(chunk_texts), :].tolist()

    query_dfs = [input_df[i : i + 2048] for i in range(0, input_df.shape[0], 2048)]

    result_dfs = []
    logger.info("Generating candidates with Hybrid Search...")
    for query_df in tqdm(query_dfs, desc="Processing batches"):
        result_dfs.append(
            client.query_collection(
                query_df=query_df,
                collection_name=collection_name,
                vector_dimensions=embedding_dim,
                n_jobs=args.n_jobs,
                n_hits=args.n_hits,
                alpha=args.alpha,
                hnsw_metric_function="array_cosine_distance",
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
        result.sort_values(["score"], ascending=False)
        .groupby("doc_id")
        .head(args.top_k)
    )
    result = pd.merge(result, n_chunks_df, on="doc_id", how="left")
    result["score"] = result["score"] / result["n_chunks"]
    result["first_occurence"] = result["first_occurence"] / result["n_chunks"]
    result["last_occurence"] = result["last_occurence"] / result["n_chunks"]
    result["spread"] = result["spread"] / result["n_chunks"]
    result = result.drop("n_chunks", axis=1)

    logger.info("Writing Outputs...")
    result.to_feather(args.output)