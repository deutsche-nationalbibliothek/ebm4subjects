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
import time
import hashlib
import nltk

sentence_tokenizer = nltk.data.load("tokenizers/punkt/german.pickle")
# Create argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')

# Add arguments with default values
parser.add_argument('--corpus', type=str, default='corpora/title/test.tsv.gz', help='Corpus to be indexed')
parser.add_argument('--index', type=str, default='corpora/title/test.arrow', help='Index file')
parser.add_argument('--max_docs', type=int, default=-1, help='Maximum number of documents to index')
parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for hybrid search')
parser.add_argument('--top_k', type=int, default=100, help='Number of top k results to keep')
parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size')
parser.add_argument('--pref_labels', type=str, default='vocab/gnd_pref_labels.arrow', help='Data frame with preferred labels')
parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--output', type=str, default='results/test/predictions.arrow', help='Output file')
parser.add_argument('--task', type=str, default='title', help='Task name')
parser.add_argument('--evalset', type=str, default='test', help='Evaluation set')

# Parse arguments
args = parser.parse_args()

# Access the arguments
corpus = args.corpus
index = args.index
alpha = args.alpha
top_k = args.top_k
chunk_size = args.chunk_size
if not FileExistsError(corpus):
    sys.exit("Corpus file does not exist. Exiting...")
pref_labels = args.pref_labels
n_jobs = args.n_jobs
output = args.output
task = args.task
evalset = args.evalset

def index_chunk(text_query, doc_id, chunk_id, client, alpha, n_hits = 20, host='8090'):
    gnd_collection = client.collections.get('Gnd859k_baai_bge_m3')
    text_collection = client.collections.get(f'{task}_{evalset}_baai_bge_m3')
    embedding_response = None
    # check if doc_id and chunk_id are already vectorzied and stored in text_collection
    text_collection_response = text_collection.query.fetch_objects(
        filters=(
            Filter.by_property('doc_id').equal(doc_id) & 
            Filter.by_property('chunk_id').equal(chunk_id)
        ),
        limit=1,
        include_vector = True
    )
    if not text_collection_response.objects:
        for _ in range(100):
            try:
                embedding_response = requests.post(
                    'http://127.0.0.1:{}/embed'.format(host),
                    headers={"Content-Type": "application/json"},
                    json={'inputs': text_query}).json()
                break
            except Exception as e:
                # print("An error occurred:", str(e))
                # print("Retrying...")
                time.sleep(1)
    else:
        # print("found existing embedding for doc_id:", doc_id, "chunk_id:", chunk_id)
        embedding_response = text_collection_response.objects[0].vector['default']

    
    if embedding_response is None:
        print("Failed to get embedding for text query:", text_query, "doc_id:", doc_id)
        return {}

    embedding = list(
        np.array(
            embedding_response
        ).reshape(-1)
    )
    # store embedding in text_collection
    if not text_collection_response.objects:
        # print("Storing new embedding for doc_id:", doc_id, "chunk_id:", chunk_id)
        text_collection.data.insert(
            properties={
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'chunk_text': text_query,
                'chunking_config': 'sentence_tokenizer'
            },
            vector=embedding
        )

    response = gnd_collection.query.hybrid(
        query=text_query,
        vector=embedding,
        limit=n_hits,
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
        return df
    else:
        return {}

def index_text(text: str, doc_id: str, client, alpha: float, top_k: int = 100, chunk_size: int = 1000, fixed_length_chunking: bool = False):
    # Split the text into chunks of 1000 characters
    limit_n_chunks = 300
    chunks = []
    if not fixed_length_chunking:
        try:
            chunks = sentence_tokenizer.tokenize(text)
            chunks = [chunk for chunk in chunks if len(chunk) >= 10]
            n_chunks = len(chunks)
        except Exception as e:
            print("An error occurred during sentence tokenization:", str(e))
            fixed_length_chunking = True
    if n_chunks >= limit_n_chunks:
        print("Warning: Number of chunks is greater than 300. Switching to fixed length chunking.") 
        fixed_length_chunking = True
    
    if fixed_length_chunking:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    candidates = pd.DataFrame(columns=['doc_id', 'label_id', 'score', 'chunk_position'])
    chunk_position = 0
    n_chunks = len(chunks)
    for chunk in chunks:
        if len(chunk.split()) <= 1:
            continue
        chunk_md5 = hashlib.md5(chunk.encode()).hexdigest()
        chunk_df = index_chunk(chunk, doc_id, chunk_md5, client, alpha)
        if not chunk_df.empty:
            chunk_df['chunk_position'] = chunk_position/n_chunks
            if not candidates.empty:
                candidates = pd.concat([candidates, chunk_df])
            else:
                candidates = chunk_df
        chunk_position += 1

    df = candidates.groupby('label_id').agg(
        score=('score', 'sum'),
        occurrences=('doc_id', 'count'),
        first_occurence=('chunk_position', 'min'),
        last_occurence=('chunk_position', 'max'),
        spread=('chunk_position', lambda x: x.max() - x.min())
    ).reset_index()
    df.columns = ['label_id', 'score', 'occurrences', 'first_occurence', 'last_occurence', 'spread']
    # normalize score by n_chunks
    df['score'] = df['score'] / n_chunks
    # add new column doc_id at as first column, filled with the constand value `doc_id` as passed to the function
    df.insert(0, 'doc_id', doc_id)
    # arrange by score and keep top K
    df = df.sort_values('score', ascending=False).head(top_k)

    return df

df_index = pd.read_feather(index)
if args.max_docs > 0:
    df_index = df_index.head(args.max_docs)

# Get the data from the tsv.gz corpus fil
df_documents = pd.read_csv(
    corpus,
    compression="gzip",
    sep="\t",
    header=None,
    names=["content", "ground_truth"],
    nrows=df_index.shape[0]
)

df_documents["doc_id"] = df_index.idn

# df_documents = df_documents.head(10)

# Define the function to be executed in parallel
def process_document(row, client):
    if isinstance(row, dict):
        text_query = row['content']
        doc_id = row['doc_id']
    else:
        text_query = row.content
        doc_id = row.doc_id
    # print(f"Processing document {doc_id}")
    try:
        return index_text(text_query, doc_id, client, alpha, top_k, chunk_size)
    except Exception as e:
        print("An error occurred in doc_id", doc_id, str(e))
        return None

# Define the function to process a batch of documents
def process_batch(batch):
    # batch_id = multiprocessing.current_process()._identity[0]
    # print(f"Processing batch {batch_id}")
    client = weaviate.connect_to_local()
    print(client.is_ready())
    if not client.is_ready():
      sys.exit("Weaviate client is not ready. Exiting...")
    result = []
    for row in tqdm(batch, total=len(batch)):
      result.append(process_document(row, client))
    client.close()
    return result

# Split df_documents into batches
batch_size = len(df_documents) // n_jobs
batches = [df_documents.iloc[i:i + batch_size].to_dict(orient='records') for i in range(0, len(df_documents), batch_size)]

# Apply the function to each batch in parallel
parallel = True
if parallel:
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(process_batch, batches)
else:
    results = [process_batch(batch) for batch in batches]

# Concatenate the results
df_results = pd.concat([pd.concat(result) for result in results])

# Write results arrow file to 'output'
df_results.to_feather(output)


