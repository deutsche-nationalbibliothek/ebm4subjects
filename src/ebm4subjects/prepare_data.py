from pathlib import Path

import polars as pl
import pyoxigraph

from ebm4subjects.embedding_generator import EmbeddingGenerator

PREF_LABEL_URI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_URI = "http://www.w3.org/2004/02/skos/core#altLabel"


def parse_vocab(vocab_path: Path, use_altLabels: bool = True) -> pl.DataFrame:
    with vocab_path.open("rb") as in_file:
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)
        doc_ids = []
        label_texts = []
        pref_labels = []

        for identifier, pref_alt, label, _ in graph:
            doc_id = identifier.value.split("/")[-1]
            label_text = label.value

            if pref_alt.value == PREF_LABEL_URI:
                doc_ids.append(doc_id)
                label_texts.append(label_text)
                pref_labels.append(True)
            elif pref_alt.value == ALT_LABEL_URI and use_altLabels:
                doc_ids.append(doc_id)
                label_texts.append(label_text)
                pref_labels.append(True)

    return pl.DataFrame(
        {
            "doc_id": doc_ids,
            "label_text": label_texts,
            "is_prefLabel": pref_labels,
        }
    )


def add_vocab_embeddings(
    vocab: pl.DataFrame,
    model_name: str,
    embedding_dimensions: int,
    batch_size: int = 1,
):
    generator = EmbeddingGenerator(model_name, embedding_dimensions)

    embeddings = generator.generate_embeddings(
        vocab.get_column("label_text").to_list(),
        batch_size=batch_size,
        task="retrieval.query",
    )

    return vocab.with_columns(
        pl.Series(name="embeddings", values=embeddings.tolist()),
        pl.Series(name="id", values=[i for i in range(vocab.height)]),
    )
