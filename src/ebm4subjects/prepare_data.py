import polars as pl
import pyoxigraph
from rdflib.namespace import SKOS

from ebm4subjects.embedding_generator import EmbeddingGenerator


def parse_vocab(vocab_path: str, use_altLabels: bool = True) -> pl.DataFrame:
    with open(vocab_path, "rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)
        label_ids = []
        label_texts = []
        pref_labels = []

        for identifier, predicate, label, _ in graph:
            if predicate.value == str(SKOS.prefLabel):
                label_ids.append(identifier.value)
                label_texts.append(label.value)
                pref_labels.append(True)
            elif predicate.value == str(SKOS.altLabel) and use_altLabels:
                label_ids.append(identifier.value)
                label_texts.append(label.value)
                pref_labels.append(False)

    return pl.DataFrame(
        {
            "label_id": label_ids,
            "label_text": label_texts,
            "is_prefLabel": pref_labels,
        }
    )


def add_vocab_embeddings(
    vocab: pl.DataFrame, generator: EmbeddingGenerator, encode_args: dict = None
):
    embeddings = generator.generate_embeddings(
        vocab.get_column("label_text").to_list(),
        **(encode_args if encode_args is not None else {}),
    )

    return vocab.with_columns(
        pl.Series(name="embeddings", values=embeddings.tolist()),
        pl.Series(name="id", values=[i for i in range(vocab.height)]),
    )
