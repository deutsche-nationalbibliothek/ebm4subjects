import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
import pandas as pd
import numpy as np
import requests
from weaviate.classes.query import MetadataQuery
import sys
import argparse
# Create argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')

# Add arguments
parser.add_argument('corpus', type=str, help='Cropus to be indexed')
parser.add_argument('arg2', type=str, help='Description of arg2')

# Parse arguments
args = parser.parse_args()

def index_text(query_text):
  embedding = list(
    np.array(
      requests.post(
        'http://127.0.0.1:{}/embed'.format(host),
        headers={"Content-Type": "application/json"},
        json={'inputs': query}).json()
        ).reshape(-1)
  )

# Access the arguments
corpus = args.cropus
if not FileExistsError(corpus):
  sys.exit("Corpus file does not exist. Exiting...")
arg2 = args.arg2

# Rest of the code
# ...

client = weaviate.connect_to_local()
if not client.is_ready():
  sys.exit("Weaviate client is not ready. Exiting...")

gnd_collection = client.collections.get('Gnd859k_baai_bge_m3')
# Get the data from the tsv.gz corpus file