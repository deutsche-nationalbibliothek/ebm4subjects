from pathlib import Path

import numpy as np
import polars as pl
import pyoxigraph

from ebm4subjects.embedding_generator import EmbeddingGenerator

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"


def parse_vocab(ttl_path: Path, use_altLabels: bool = True) -> pl.DataFrame:
    with ttl_path.open("rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)
        idns = []
        label_texts = []
        pref_labels = []

        for identifier, pref_alt, label, _ in graph:
            idn = identifier.value.split("/")[-1]
            label_text = label.value

            if pref_alt.value == PREF_LABEL_IRI:
                idns.append(idn)
                label_texts.append(label_text)
                pref_labels.append(True)
            elif pref_alt.value == ALT_LABEL_IRI and use_altLabels:
                idns.append(idn)
                label_texts.append(label_text)
                pref_labels.append(True)

    return pl.DataFrame({"idn": idns, "label_text": label_texts, "is_prefLabel": pref_labels})


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
    vocab.write_ipc(vocab_out)

    generator = EmbeddingGenerator(model_name, embedding_dimensions)

    embeddings = generator.generate_embeddings(
        vocab.get_column("label_text").to_list(),
        batch_size=batch_size,
        task="retrieval.query",
    )
    np.save(embeddings_out, embeddings)
