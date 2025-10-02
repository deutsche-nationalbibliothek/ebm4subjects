import polars as pl
import pyoxigraph
from rdflib.namespace import SKOS

from ebm4subjects.embedding_generator import EmbeddingGenerator


def parse_vocab(vocab_path: str, use_altLabels: bool = True) -> pl.DataFrame:
    """
    Parse a SKOS vocabulary file and return a DataFrame containing label
    and label information.

    Args:
        vocab_path (str): Path to the SKOS vocabulary file in Turtle format.
        use_altLabels (bool, optional): Whether to include alternative labels
            in the output (default: True).

    Returns:
        pl.DataFrame: A DataFrame containing label information, including label IDs,
            label texts, and whether each label is a preferred label.
    """
    # Open the vocabulary file
    with open(vocab_path, "rb") as in_file:
        # Parse the RDF graph from the file using PyOXIGraph
        graph = pyoxigraph.parse(input=in_file, format=pyoxigraph.RdfFormat.TURTLE)

        # Initialize lists to store label information
        label_ids = []
        label_texts = []
        pref_labels = []

        # Iterate over the RDF triples in the graph
        for identifier, predicate, label, _ in graph:
            # Check if the current label is as preferred or
            # alternative label and add the label information to the lists
            if predicate.value == str(SKOS.prefLabel):
                label_ids.append(identifier.value)
                label_texts.append(label.value)
                pref_labels.append(True)
            elif predicate.value == str(SKOS.altLabel) and use_altLabels:
                label_ids.append(identifier.value)
                label_texts.append(label.value)
                pref_labels.append(False)

    # Return a DataFrame containing the label information
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
    """
    Adds vocabulary embeddings to the given DataFrame.

    Args:
        vocab (pl.DataFrame): The DataFrame containing the vocabulary.
        generator (EmbeddingGenerator): The generator used to create the embeddings.
        encode_args (dict, optional): Additional arguments for the embedding generator
            (default: None).

    Returns:
        pl.DataFrame: The updated DataFrame with the added embeddings.
    """
    # Generate embeddings using the provided generator and vocabulary
    embeddings = generator.generate_embeddings(
        # Get the text labels from the vocabulary DataFrame
        vocab.get_column("label_text").to_list(),
        # Pass any additional arguments to the generator
        **(encode_args if encode_args is not None else {}),
    )

    # Add the generated embeddings and IDs to the vocabulary DataFrame
    return vocab.with_columns(
        pl.Series(name="embeddings", values=embeddings.tolist()),
        pl.Series(name="id", values=[i for i in range(vocab.height)]),
    )
