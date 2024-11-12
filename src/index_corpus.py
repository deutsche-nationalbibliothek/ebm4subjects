import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter
import torch
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
from dvc.api import params_show
import logging
from functools import partial

# Create argument parser
parser = argparse.ArgumentParser(description='Process command line arguments')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Indexer:

    def __init__(self, alpha, n_hits, top_k, vocab_collection):
        self.alpha = alpha
        self.n_hits = n_hits
        self.top_k = top_k
        self.vocab_collection = vocab_collection

    def index_chunk(
            self,
            text_query: str, 
            embedding: list,
            doc_id: str, 
            client: weaviate.Client, 
    ):
        vocab_collection = client.collections.get(self.vocab_collection)

        response = vocab_collection.query.hybrid(
            query=text_query,
            vector=embedding.numpy(),
            limit=self.n_hits,
            return_metadata=MetadataQuery(score=True),
            alpha=self.alpha
        )

        df = pd.DataFrame(
            [
                dict(
                    doc_id=doc_id,
                    label_id=o.properties['idn'],
                    score=o.metadata.score,
                    is_prefLabel=o.properties['is_prefLabel']
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

    def index_text(self, chunks: list, embeddings: list, doc_id: str, client: weaviate.Client):
        # Split the text into chunks of 1000 characters
        candidates = pd.DataFrame(columns=['doc_id', 'label_id', 'score', 'chunk_position', 'is_prefLabel'])
        n_chunks = len(chunks)
        for i in range(n_chunks):
            # skip 1-word chunks
            #if len(chunks[i].split()) <= 1:
            #    continue
            chunk_df = self.index_chunk(
                text_query = chunks[i],
                embedding = embeddings[i], 
                doc_id = doc_id, 
                client = client)
            if not chunk_df.empty:
                chunk_df['chunk_position'] = i/n_chunks
                if not candidates.empty:
                    candidates = pd.concat([candidates, chunk_df])
                else:
                    candidates = chunk_df

        df = candidates.groupby('label_id').agg(
            score=('score', 'sum'),
            occurrences=('doc_id', 'count'),
            first_occurence=('chunk_position', 'min'),
            last_occurence=('chunk_position', 'max'),
            spread=('chunk_position', lambda x: x.max() - x.min()),
            is_prefLabel=('is_prefLabel', 'any')
        ).reset_index()
        df.columns = ['label_id', 'score', 'occurrences', 'first_occurence', 'last_occurence', 'spread', 'is_prefLabel']
        # normalize score by n_chunks
        df['score'] = df['score'] / n_chunks
        # add new column doc_id at as first column, filled with the constand value `doc_id` as passed to the function
        df.insert(0, 'doc_id', doc_id)
        # arrange by score and keep top K
        df = df.sort_values('score', ascending=False).head(self.top_k)

        return df

# Define the function to be executed in parallel
def process_document(item, client: weaviate.Client, indexer: Indexer):
    if not isinstance(item, dict):
        raise ValueError("Row must be an instance of dict")
    chunks = item['chunks']
    embeddings = item['embeddings']
    doc_id = item['doc_id']

    # print(f"Processing document {doc_id}")
    try:
        return indexer.index_text(chunks=chunks,
                          embeddings=embeddings, 
                          doc_id = doc_id, 
                          client = client)
    except Exception as e:
        print("An error occurred in doc_id", doc_id, str(e))
        return None

# Define the function to process a batch of documents
def process_batch(batch, position, indexer: Indexer):
    # batch_id = multiprocessing.current_process()._identity[0]
    # print(f"Processing batch {batch_id}")
    client = weaviate.connect_to_local()
    if not client.is_ready():
      sys.exit("Weaviate client is not ready. Exiting...")
    result = []
    for item in (pbar := tqdm(batch, total=len(batch), position=position, leave=True)):
        pbar.set_description(f"{position:02}")
        result.append(
            process_document(item, client, indexer))
    client.close()
    return result

if __name__ == '__main__':
    # Add arguments with default values
    parser.add_argument('--chunks', type=str, default='results/test/chunks.txt', help='Text chunks to be indexed')
    parser.add_argument('--chunk_index', type=str, default='results/test/chunk_index.arrow', help='Index file containing a mapping of documents and chunks')
    parser.add_argument('--chunk_embeddings', type=str, default='results/test/embeddings.arrow', help='Embeddings file matching the chunks and chunk_index')
    parser.add_argument('--alpha', type=float, default=0.0, help='Alpha value for hybrid search')
    parser.add_argument('--top_k', type=int, default=100, help='Number of top k results to keep')
    parser.add_argument('--n_hits', type=int, default=20, help='Number of hits to retrieve from Hybrid Search')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
    parser.add_argument('--output', type=str, default='results/test/candidates.arrow', help='Output file')

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load parameters from params.yaml
    params = params_show()

    indexer = Indexer(
        alpha=args.alpha, n_hits=args.n_hits, top_k=args.top_k, 
        vocab_collection=params["vocab_config"]["collection_name"]
        )

    n_jobs = args.n_jobs
    output = args.output


    logger.info("Reading chunks and chunk index")
    # Get the data from previous stages chunk_texts and embed_chunks
    with open(args.chunks, 'r', encoding='utf-8') as file:
        chunks = file.readlines()
    chunk_index = pd.read_feather(args.chunk_index)
    logger.info("Reading torch embeddings")
    chunk_embdeddings = torch.load(args.chunk_embeddings, weights_only=True)
    chunk_embdeddings = chunk_embdeddings.to('cpu')
    docs_w_emb = []

    # Iterate over unique doc_ids in the chunk index
    for doc_id in chunk_index['doc_id'].unique():
        # Get the chunks and embeddings for the current doc_id
        this_docs_chunk_ids = chunk_index[chunk_index['doc_id'] == doc_id]['abs_chunk_position'].tolist()
        start = this_docs_chunk_ids[0]
        end = this_docs_chunk_ids[-1] + 1
        doc_chunks = chunks[start:end]
        doc_embeddings = chunk_embdeddings[start:end, :]
        # Create a dictionary for the current document
        doc_item = {
            'doc_id': doc_id,
            'chunks': doc_chunks,
            'embeddings': doc_embeddings
        }
        # Append the document item to the list
        docs_w_emb.append(doc_item)

    batch_size = int(len(docs_w_emb) / n_jobs) + (len(docs_w_emb) % n_jobs > 0)

    batches = [(
        docs_w_emb[i:i + batch_size],
        i // batch_size,
        indexer
    ) for i in range(0, len(docs_w_emb), batch_size)]
    
    # Apply the function to each batch in parallel
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.starmap(process_batch, batches)

    # Concatenate the results
    inner_results_list = [pd.concat(result) for result in results if result is not None]
    df_results = pd.concat([result for result in inner_results_list if not result.empty])

    # Write results arrow file to 'output'
    df_results.to_feather(output)


