from concurrent.futures import ProcessPoolExecutor
from math import ceil
from typing import Tuple

import polars as pl

from ebm4subjects.analyzer import EbmAnalyzer


class Chunker:
    def __init__(
        self,
        tokenizer_name: str,
        max_chunks: int | None,
        max_chunk_size: int | None,
        max_sentences: int | None,
    ):
        self.max_chunks = max_chunks if max_chunks else float("inf")
        self.max_chunk_size = max_chunk_size if max_chunk_size else float("inf")
        self.max_sentences = max_sentences if max_sentences else float("inf")

        self.tokenizer = EbmAnalyzer(tokenizer_name)

    def chunk_text(self, text: str) -> list[str]:
        chunks = []
        sentences = self.tokenizer.tokenize_sentences(text)
        sentences = sentences[: self.max_sentences]

        current_chunk = []
        for sentence in sentences:
            if len(" ".join(current_chunk)) < self.max_chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                if len(chunks) == self.max_chunks:
                    break

        if current_chunk and len(chunks) < self.max_chunks:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_batches(
        self, texts: list[str], doc_ids: list[str], chunking_jobs: int, query_jobs: int
    ) -> Tuple[list[str], list[str]]:
        text_chunks = []
        chunk_index = []

        num_batches = chunking_jobs
        chunking_batch_size = ceil(len(texts) / num_batches)
        batch_args = [
            (
                doc_ids[i * chunking_batch_size : (i + 1) * chunking_batch_size],
                texts[i * chunking_batch_size : (i + 1) * chunking_batch_size],
            )
            for i in range(num_batches)
        ]

        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=query_jobs) as executor:
            results = list(executor.map(self._chunk_batch, batch_args))

        for batch_chunks, batch_chunk_indices in results:
            text_chunks.extend(batch_chunks)
            chunk_index.extend(batch_chunk_indices)

        return text_chunks, chunk_index

    def _chunk_batch(self, args):
        batch_doc_ids, batch_texts = args

        batch_chunks = []
        batch_chunk_indices = []
        for doc_id, text in zip(batch_doc_ids, batch_texts):
            new_chunks = self.chunk_text(text)
            n_chunks = len(new_chunks)
            chunk_df = pl.DataFrame(
                {
                    "query_doc_id": [doc_id] * n_chunks,
                    "chunk_position": list(range(n_chunks)),
                    "n_chunks": [n_chunks] * n_chunks,
                }
            )
            batch_chunks.extend(new_chunks)
            batch_chunk_indices.append(chunk_df)
        return batch_chunks, batch_chunk_indices
