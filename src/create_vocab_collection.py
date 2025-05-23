import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pyoxigraph
from dvc.api import params_show
from tqdm import tqdm

from duckdb_client import Duckdb_client
from utils import str2bool

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"

params = params_show()
embedding_dim = params["general"]["embedding_dim"]


def parse_vocab(
    ttl_path: Path, use_altLabels: bool = True, phrase: str = None
) -> pd.DataFrame:
    print(f"Parsing vocabulary from {ttl_path}")

    with ttl_path.open("rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, mime_type="text/turtle")
        labels: list[(str, str, bool)] = []

        for identifier, pref_alt, label in tqdm(graph, desc="Processing triples"):
            idn = identifier.value.split("/")[-1]
            is_prefLabel = pref_alt.value == PREF_LABEL_IRI
            is_altLabel = pref_alt.value == ALT_LABEL_IRI
            label_text = label.value if phrase is None else f"{phrase}{label.value}"
            if is_prefLabel:
                labels.append((idn, label_text, True))
            elif is_altLabel and use_altLabels:
                labels.append((idn, label_text, False))

        labels = pd.DataFrame(labels, columns=["idn", "label_text", "is_prefLabel"])

        return labels


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_name",
        help="Collection name",
        type=str,
        default="gnd204k_w_altlabels_no_phrase_jina_1024_retrieval",
    )
    parser.add_argument(
        "--db_path", help="Path to DuckDB", type=str, default="gnd.duckdb"
    )
    parser.add_argument("--overwrite", help="Overwrite", type=str, default=True)
    parser.add_argument(
        "--arrow_in",
        help="Arrow input",
        type=str,
        default="vocab/gnd204k_w_altlabels_w_embeddings.arrow",
    )
    parser.add_argument(
        "--embeddings",
        help="Numpy Array with embeddings",
        type=str,
        default="vocab/embeddings.npy",
    )
    args = parser.parse_args()

    vocab = pd.read_feather(args.arrow_in)

    client = Duckdb_client(
        db_path=args.db_path,
        config={"hnsw_enable_experimental_persistence": True},
    )

    embeddings = np.load(args.embeddings)

    if str2bool(args.overwrite):
        vocab["embeddings"] = pd.Series(embeddings.tolist())
        vocab["id"] = [i for i in range(len(vocab["idn"].tolist()))]

        client.create_collection(
            collection_df=vocab,
            collection_name=args.collection_name,
            vector_dimensions=embedding_dim,
            hnsw_index_name="hnsw_index",
            hnsw_metric="cosine",
            fts_config={
                "stemmer": "german",
                "stopwords": "none",
                "ignore": "[^a-z]+",
                "strip_accents": 1,
                "lower": 1,
            },
            force=args.overwrite,
        )


run()
