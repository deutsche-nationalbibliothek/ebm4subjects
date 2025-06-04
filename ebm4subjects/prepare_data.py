from pathlib import Path

import numpy as np
import pandas as pd
import pyoxigraph

from ebm4subjects.embedding_generator import EmbeddingGenerator

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"


def parse_vocab(ttl_path: Path, use_altLabels: bool = True) -> pd.DataFrame:
    with ttl_path.open("rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)
        labels: list[(str, str, bool)] = []

        for identifier, pref_alt, label, _ in graph:
            idn = identifier.value.split("/")[-1]
            label_text = label.value

            if pref_alt.value == PREF_LABEL_IRI:
                labels.append((idn, label_text, True))
            elif pref_alt.value == ALT_LABEL_IRI and use_altLabels:
                labels.append((idn, label_text, False))

    return pd.DataFrame(labels, columns=["idn", "label_text", "is_prefLabel"])


def create_vocab_embeddings(
    ttl_path: Path,
    vocab_out: Path,
    embeddings_out: Path,
    model_name: str,
    embedding_dimensions: int,
    batch_size: int = 1,
    use_altLabels: bool = True,
):
    vocab = parse_vocab(ttl_path, use_altLabels)
    vocab.to_feather(vocab_out)

    generator = EmbeddingGenerator(model_name, embedding_dimensions)

    embeddings = generator.generate_embeddings(
        vocab["label_text"].tolist(),
        batch_size=batch_size,
        task="retrieval.query",
    )
    np.save(embeddings_out, embeddings)
