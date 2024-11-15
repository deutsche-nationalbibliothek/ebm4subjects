import argparse
from tqdm import tqdm
import itertools
import pandas as pd
from pathlib import Path
import pyoxigraph
from pyoxigraph import RdfFormat
import pandas as pd
import torch
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from generate_embeddings import generate_embeddings
from utils import str2bool

PREF_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#prefLabel"
ALT_LABEL_IRI = "http://www.w3.org/2004/02/skos/core#altLabel"

def parse_vocab(ttl_path: Path, use_altLabels: bool = True, phrase: str = None) -> pd.DataFrame:
    print(f"Parsing vocabulary from {ttl_path}")
    with ttl_path.open("rb") as f:
        graph = pyoxigraph.parse(f, RdfFormat.TURTLE)
        labels: list[(str, str, bool)] = []
        # uris: list[str] = []
        # limit = 100
        #itertools.islice(graph, limit)

        for s,p,o,_  in tqdm(graph, desc="Processing triples"):
            uri = s.value
            idn = uri.split('/')[-1]  # extract idn from uri
            is_prefLabel = (p.value == PREF_LABEL_IRI)
            is_altLabel = (p.value == ALT_LABEL_IRI)
            label_text = o.value if phrase is None else f"{phrase}{o.value}"
            if is_prefLabel:
                labels.append((idn, label_text, True))
            elif is_altLabel and use_altLabels:
                labels.append((idn, label_text, False))

            # uris.append(idn)

        labels = pd.DataFrame(labels, columns=["idn", "label_text", "is_prefLabel"])
        # labels.drop_duplicates("idn", keep="first", inplace=True)
        # uris = pd.DataFrame(list(set(uris)), columns=["idn"])

    # df  = pd.merge(uris, labels, on="idn", how="left")
    # gnd_data = pd.read_feather("gnd_data.arrow")
    # df = pd.merge(df, gnd_data, on="idn", how="left")
        return labels
    
def create_collection(
        client: weaviate.Client, 
        collection_name: str,
        overwrite: bool = False,
        TEI_port: str = '8090'):
    print(f"Attempting to create collection {collection_name} in Weaviate")
    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists")
        if overwrite:
            client.collections.delete(collection_name)
            print(f"Old Collection {collection_name} deleted")
        else:
            return collection_name

    client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(
                name="idn",
                description="DNB internal identifier",
                data_type=wvc.config.DataType.TEXT,
                tokenization=wvc.config.Tokenization.WORD,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="label_text",
                description="Label description (pref label or alt label)",
                data_type=wvc.config.DataType.TEXT,
                vectorize_property_name=False,
                tokenization=wvc.config.Tokenization.WORD,
                index_searchable=True,
                index_filterable=False,
            ),
            wvc.config.Property(
                name="is_prefLabel",
                description="Boolean: Label description is a SKOS prefLabel T/F",
                data_type=wvc.config.DataType.BOOL,
                vectorize_property_name=False,
                skip_vectorization=True,
                index_searchable=False,
                index_filterable=True,
            )
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_huggingface(
            # only works if huggingface TEI endpoint is running
            endpoint_url = f"http://host.docker.internal:{TEI_port}",
            vectorize_collection_name = False
        )
    )   
    return collection_name

def insert_vocab(
        client: weaviate.Client, 
        collection_name: str,
        vocab: pd.DataFrame,
        embeddings: torch.Tensor,
        phrase: str = None):
    print(f"Inserting vocabulary into {collection_name}")
    this_collection = client.collections.get(collection_name)
    # with this_collection.batch.fixed_size(
    #     batch_size=100,
    #     concurrent_requests=100
    # ) as batch:
    with this_collection.batch.dynamic() as batch:
        # Loop through the data
        for i, row in tqdm(vocab.iterrows()):

            # Build the object payload
            gnd_entity_obj = {
                "idn": row["idn"],
                "label_text": row["label_text"] if phrase is None else f"{phrase}{row['label_text']}",
                "is_prefLabel": row["is_prefLabel"]
            }

            # Add object to batch queue
            batch.add_object(
                properties=gnd_entity_obj,
                vector=embeddings[i].tolist(),
                # uuid=generate_uuid5(row["idn"]) idn is not unique due to altLabels
                # references=reference_obj  # You can add references here
            )

            # if batch.failed_objects:
            #     print(f"Failed to import {len(batch.failed_objects)} objects")
            # Batcher automatically sends batches

    # Check for failed objects
    if len(this_collection.batch.failed_objects) > 0:
        print(f"Failed to import {len(this_collection.batch.failed_objects)} objects")



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl_file", help="Input Filename/Path", type=str, required=True)
    parser.add_argument("--collection_name", help="Collection Name in Weaviate", type=str, required=True)
    parser.add_argument("--TEI_port", help="Host", type=str, default='8090')
    parser.add_argument("--phrase", help="Phrase", type=str, default="Ein gutes Schlagwort für dieses Dokument lautet: ")
    parser.add_argument("--overwrite", help="Overwrite", type=str, default=True)
    parser.add_argument("--arrow_out", help="Arrow output", type=str, default=None)
    parser.add_argument("--use_altLabels", help="Use altLabels", type=str, default=True)
    # parser.add_argument("--labelkind", help="Labelkind", type=list, default=["prefLabel"])
    args = parser.parse_args()
    vocab = parse_vocab(Path(args.ttl_file), use_altLabels=str2bool(args.use_altLabels), phrase=args.phrase)

    client = weaviate.connect_to_local()
    if str2bool(args.overwrite):
        embeddings = generate_embeddings(vocab["label_text"].tolist())
        create_collection(client, args.collection_name, overwrite=str2bool(args.overwrite), TEI_port=args.TEI_port)
        insert_vocab(client, args.collection_name, vocab, embeddings) 
        # Note: phrase is already passed to parse_vocab
    
    if args.arrow_out is not None:
        vocab.to_feather(args.arrow_out)

    if not client.collections.exists(args.collection_name):
        raise ValueError(f"Collection {args.collection_name} does not exist. Try running with --overwrite=True")
    client.close()



run()