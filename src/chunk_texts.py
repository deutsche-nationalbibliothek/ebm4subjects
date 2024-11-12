import argparse
import pandas as pd
import nltk
import multiprocessing
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import hashlib
import logging

nltk.download('punkt')

class Chunker:
    def __init__(self, max_chunks, chunk_size, max_sentences_per_doc, fixed_length_chunking):
        self.max_chunks = max_chunks if max_chunks is not None else float('inf')
        self.chunk_size = chunk_size if chunk_size is not None else float('inf')
        self.max_sentences_per_doc = max_sentences_per_doc if max_sentences_per_doc is not None else float('inf')
        self.fixed_length_chunking = fixed_length_chunking
        self.tokenizer = nltk.data.load("tokenizers/punkt/german.pickle")

    def __repr__(self):
        return (f"Chunker(max_chunks={self.max_chunks}, chunk_size={self.chunk_size}, "
                f"max_sentences_per_doc={self.max_sentences_per_doc}, fixed_length_chunking={self.fixed_length_chunking})")

    def chunk_text(self, text, doc_id):
        chunks = []
        sentences = self.tokenizer.tokenize(text)
        sentences = sentences[:self.max_sentences_per_doc]
        current_chunk = []
        for sentence in sentences:
            if len('. '.join(current_chunk)) < self.chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append(('. '.join(current_chunk), doc_id))
                current_chunk = [sentence]
                if len(chunks) >= self.max_chunks:
                    break
        if current_chunk:
            chunks.append(('. '.join(current_chunk), doc_id))
        return chunks

def process_batch(chunker, texts, doc_ids, position):
    chunked_texts = []
    with tqdm(total=len(texts), position = position, desc="Chunking texts", leave=False) as pbar:
        pbar.set_description(f"{position:02}")
        for text, doc_id in zip(texts, doc_ids):
            chunked_texts.extend(chunker.chunk_text(text, doc_id))
            pbar.update(1)

    return chunked_texts

if __name__ == '__main__':
    # Add arguments with default values
    parser = argparse.ArgumentParser(description='Process command line arguments')
    parser.add_argument('--corpus', type=str, default='corpora/title/test.tsv.gz', help='Corpus to be indexed')
    parser.add_argument('--index', type=str, default='corpora/title/test.arrow', help='Index file')
    parser.add_argument('--max_docs', type=int, default=-1, help='Maximum number of documents to index')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Maximum character length of one text chunk')
    parser.add_argument('--max_chunks', type=int, default=100, help='Maximum number of chunks per Document')
    parser.add_argument('--max_sentences_per_doc', type=int, default=500, help='Maximum number of sentences per document')
    parser.add_argument('--fixed_length_chunking', type=bool, nargs='?', const=True, default=False, help='Use fixed length chunking')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--chunk_texts_output', type=str, default='results/test/chunks.txt', help='Output file for texxt chunks')
    parser.add_argument('--chunk_index_output', type=str, default='results/test/chunk_index.arrow', help='Chunk index output file')
    args = parser.parse_args()

    chunker = Chunker(
        max_chunks=args.max_chunks if args.max_chunks != -1 else None,
        chunk_size=args.chunk_size,
        max_sentences_per_doc=args.max_sentences_per_doc,
        fixed_length_chunking=args.fixed_length_chunking
    )

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Reading index and documents")
    index = pd.read_feather(args.index)
    documents = pd.read_csv(
        args.corpus,
        compression="gzip",
        sep="\t",
        header=None,
        names=["content", "ground_truth"],
        nrows=args.max_docs
    )

    documents["doc_id"] = index.idn
    if args.max_docs > 0:
        index = index.head(args.max_docs)
        documents = documents.head(args.max_docs)

    # Create a partial function with the chunker instance
    process_batch_partial = partial(process_batch, chunker)
    batch_size = int(len(documents) / args.n_jobs) + (len(documents) % args.n_jobs > 0)

    batches = [(
        documents['content'][i:i + batch_size].tolist(),
        documents['doc_id'][i:i + batch_size].tolist(),
        i // batch_size
    ) for i in range(0, len(documents), batch_size)]
    # Process documents in batches

    logger.info("Starting chunking process with %d jobs", args.n_jobs)
    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        results = pool.starmap(process_batch_partial, batches)

    # Flatten the results and separate chunks and chunk_index
    chunks = [chunk for batch_result in results for chunk, _ in batch_result]
    chunk_index = [(hashlib.md5(chunk.encode('utf-8')).hexdigest(), doc_id) for batch_result in results for chunk, doc_id in batch_result]
    


    # Write chunks to plain text file
    with open(args.chunk_texts_output, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write('\"' + chunk + '\"' + '\n')

    # Write chunk_index to CSV file
    chunk_index_df = pd.DataFrame(chunk_index, columns=['chunk_md5hash', 'doc_id'])
    # Add chunk_position to relative to the document
    chunk_index_df['rel_chunk_position'] = chunk_index_df.groupby('doc_id').cumcount()
    # Add absolute chunk position in corpus
    chunk_index_df['abs_chunk_position'] = range(len(chunk_index_df))
    chunk_index_df.to_feather(args.chunk_index_output)