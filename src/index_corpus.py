import argparse
import logging

import numpy as np
import polars as pl
from dvc.api import params_show

from duckdb_client import Duckdb_client

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
    chunk_texts = open(args.chunks, "r", encoding="utf-8").readlines()

    logger.info("Reading chunk index...")
    chunk_index = pl.read_ipc(args.chunk_index)

    logger.info("Reading chunk embeddings...")
    chunk_embdeddings = np.load(args.chunk_embeddings)

    data = {
        "id": [i for i in range(len(chunk_texts))],
        "doc_id": chunk_index["doc_id"].to_list(),
        "chunk_position": chunk_index["rel_chunk_position"].to_list(),
        "embeddings": chunk_embdeddings[0 : len(chunk_texts), :].tolist(),
    }
    query_df = pl.DataFrame(data).join(
        other=chunk_index.group_by("doc_id").agg(
            pl.col("rel_chunk_position").count().alias("n_chunks")
        ),
        on="doc_id",
        how="left",
    )

    logger.info("Generating candidates with Vector Search...")
    vector_result = client.vector_search(
        query_df=query_df,
        collection_name=collection_name,
        vector_dimensions=embedding_dim,
        n_jobs=args.n_jobs,
        n_hits=args.n_hits,
        chunk_size=2048,
        top_k=args.top_k,
        hnsw_metric_function="array_cosine_distance",
    )

    logger.info("Writing Outputs...")
    vector_result.write_ipc(args.output)
