import argparse
from pathlib import Path
import pandas as pd
import pyoxigraph
import numpy as np
from tqdm import tqdm
from dvc.api import params_show

from utils import str2bool
from generate_embeddings import generate_embeddings

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
    parser.add_argument("--ttl_file", help="Input Filename/Path", type=str, required=True)
    parser.add_argument("--phrase", help="Phrase", type=str, default="Ein gutes Schlagwort für dieses Dokument lautet: ")
    parser.add_argument("--arrow_out", help="Arrow output", type=str, default=None)
    parser.add_argument("--embeddings_out", help="Output Filename/Path", type=str, required=True)
    parser.add_argument("--use_altLabels", help="Use altLabels", type=str, default=True)
    # parser.add_argument("--labelkind", help="Labelkind", type=list, default=["prefLabel"])
    args = parser.parse_args()
    vocab = parse_vocab(Path(args.ttl_file), use_altLabels=str2bool(args.use_altLabels), phrase=args.phrase)
    vocab.to_feather(args.arrow_out)

    embeddings = generate_embeddings(vocab["label_text"].tolist(), task = "retrieval.query")
    np.save(args.embeddings_out, embeddings)

run()