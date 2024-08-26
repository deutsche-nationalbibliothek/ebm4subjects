import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
import pandas as pd
import numpy as np
import requests
from weaviate.classes.query import MetadataQuery
import sys
import argparse
import multiprocessing
from tqdm import tqdm

# Create argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')

# Add arguments with default values
parser.add_argument('--corpus', type=str, default='corpora/title/test.tsv.gz', help='Corpus to be indexed')
parser.add_argument('--index', type=str, default='corpora/title/test.arrow', help='Index file')
parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for hybrid search')
parser.add_argument('--pref_labels', type=str, default='vocab/gnd_pref_labels.arrow', help='Data frame with preferred labels')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--output', type=str, default='results/test/predictions.arrow', help='Output file')

# Parse arguments
args = parser.parse_args()

# Access the arguments
corpus = args.corpus
index = args.index
alpha = args.alpha
if not FileExistsError(corpus):
    sys.exit("Corpus file does not exist. Exiting...")
pref_labels = args.pref_labels
n_jobs = args.n_jobs
output = args.output

def index_text(text_query, doc_id, wv_collection, alpha, host='8090'):
    embedding = list(
        np.array(
            requests.post(
                'http://127.0.0.1:{}/embed'.format(host),
                headers={"Content-Type": "application/json"},
                json={'inputs': text_query}).json()
        ).reshape(-1)
    )

    response = wv_collection.query.hybrid(
        query=text_query,
        vector=embedding,
        limit=100,
        return_metadata=MetadataQuery(score=True),
        alpha=alpha
    )

    df = pd.DataFrame(
        [
            dict(
                doc_id=doc_id,
                label_id=o.properties['idn'],
                score=o.metadata.score
            )
            for o in response.objects
        ]
    )

    # group by label_id and only keep rows with the highest score per label_id
    if not df.empty:
        df = df.sort_values('score', ascending=False).groupby('label_id').head(1)
        return df.to_dict(orient='records')
    else:
        return {}

# Get the data from the tsv.gz corpus file
df_documents = pd.read_csv(
    corpus,
    compression="gzip",
    sep="\t",
    header=None,
    names=["content", "ground_truth"],
)

df_index = pd.read_feather(index)
df_documents["doc_id"] = df_index.idn

# Define the function to be executed in parallel
def process_document(row, wv_collection):
    if isinstance(row, dict):
        text_query = row['content']
        doc_id = row['doc_id']
    else:
        text_query = row.content
        doc_id = row.doc_id
    return index_text(text_query, doc_id, wv_collection, alpha)

# Define the function to process a batch of documents
def process_batch(batch):
    print("Processing batch of {} documents".format(len(batch)))
    client = weaviate.connect_to_local()
    if not client.is_ready():
      sys.exit("Weaviate client is not ready. Exiting...")
    wv_collection = client.collections.get('Gnd859k_baai_bge_m3')
    result = []
    for row in tqdm(batch, total=len(batch)):
      result.extend(process_document(row, wv_collection))
    client.close()
    return result

# Split df_documents into batches
batch_size = len(df_documents) // n_jobs
batches = [df_documents.iloc[i:i + batch_size].to_dict(orient='records') for i in range(0, len(df_documents), batch_size)]

# Apply the function to each batch in parallel
with multiprocessing.Pool(processes=n_jobs) as pool:
    results = pool.map(process_batch, tqdm(batches, total=len(batches)))

# Concatenate the results into a single DataFrame
df_results = pd.concat([pd.DataFrame(batch_result) for batch_result in results])

# Write results arrow file to 'output'
df_results.to_feather(output)

