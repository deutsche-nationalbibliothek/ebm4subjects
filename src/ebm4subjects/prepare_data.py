from pathlib import Path

import polars as pl
import pyoxigraph

from ebm4subjects.embedding_generator import EmbeddingGenerator

PREF_LABEL_URI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_URI = "http://www.w3.org/2004/02/skos/core#altLabel"


def parse_vocab(vocab_path: Path, use_altLabels: bool = True) -> pl.DataFrame:
    with vocab_path.open("rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)
        label_ids = []
        label_texts = []
        pref_labels = []

        for identifier, predicate, label, _ in graph:
            label_id = identifier.value.split("/")[-1]
            label_text = label.value

            if predicate.value == PREF_LABEL_URI:
                label_ids.append(label_id)
                label_texts.append(label_text)
                pref_labels.append(True)
            elif predicate.value == ALT_LABEL_URI and use_altLabels:
                label_ids.append(label_id)
                label_texts.append(label_text)
                pref_labels.append(False)

    return pl.DataFrame(
        {
            "label_id": label_ids,
            "label_text": label_texts,
            "is_prefLabel": pref_labels,
        }
    )


def add_vocab_embeddings(
    vocab: pl.DataFrame,
    generator: EmbeddingGenerator,
    encode_args: dict = None
):

    embeddings = generator.generate_embeddings(
        vocab.get_column("label_text").to_list(),
        **(encode_args if encode_args is not None else {})
    )

    return vocab.with_columns(
        pl.Series(name="embeddings", values=embeddings.tolist()),
        pl.Series(name="id", values=[i for i in range(vocab.height)]),
    )
