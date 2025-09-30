from concurrent.futures import ProcessPoolExecutor
from math import ceil
from typing import Tuple

import polars as pl

from ebm4subjects.analyzer import EbmAnalyzer


class Chunker:
    """
    A class for chunking text into smaller sections based on various criteria.

    The Chunker class takes a tokenizer name and optional maximum chunk size,
    maximum number of chunks, and maximum number of sentences as input.
    It uses these parameters to chunk text into smaller sections.

    Attributes:
        tokenizer (EbmAnalyzer): The tokenizer used for tokenizing sentences.
        max_chunks (int): The maximum number of chunks to generate.
        max_chunk_size (int): The maximum size of each chunk in characters.
        max_sentences (int): The maximum number of sentences to consider.

    Methods:
        - chunk_text: Chunks a given text into smaller sections
        - chunk_batches: Chunks a list of texts into smaller sections in parallel
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_chunks: int | None,
        max_chunk_size: int | None,
        max_sentences: int | None,
    ):
        """
        Initializes the Chunker.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            max_chunks (int | None): The maximum number of chunks to generate.
            max_chunk_size (int | None): The maximum size of each chunk in characters.
            max_sentences (int | None): The maximum number of sentences to consider.
        """
        self.max_chunks = max_chunks if max_chunks else float("inf")
        self.max_chunk_size = max_chunk_size if max_chunk_size else float("inf")
        self.max_sentences = max_sentences if max_sentences else float("inf")

        self.tokenizer = EbmAnalyzer(tokenizer_name)

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunks a given text into smaller sections based on the maximum chunk size.

        Args:
            text (str): The text to be chunked.

        Returns:
            list[str]: A list of chunked text sections.
        """
        # Initialize an empty list to store the chunks
        chunks = []

        # Tokenize the text into sentences
        sentences = self.tokenizer.tokenize_sentences(text)
        sentences = sentences[: self.max_sentences]

        # Initialize an empty list to store the current chunk
        current_chunk = []

        # Iterate over the sentences
        for sentence in sentences:
            # If the current chunk is not full, add the sentence to it
            if len(" ".join(current_chunk)) < self.max_chunk_size:
                current_chunk.append(sentence)
            # Otherwise, add the current chunk to the list of chunks
            # and start a new chunk
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                if len(chunks) == self.max_chunks:
                    break

        # If the maximum number of chunks is reached, break the loop
        if current_chunk and len(chunks) < self.max_chunks:
            chunks.append(" ".join(current_chunk))

        # Return the chunked text
        return chunks

    def chunk_batches(
        self, texts: list[str], doc_ids: list[str], chunking_jobs: int
    ) -> Tuple[list[str], list[pl.DataFrame]]:
        """
        Chunks a list of texts into smaller sections in parallel
        using multiple processes.

        Args:
            texts (list[str]): A list of texts to be chunked.
            doc_ids (list[str]): A list of document IDs corresponding to the texts.
            chunking_jobs (int): The number of processes to use for chunking.

        Returns:
            tuple[list[str], list[pl.DataFrame]]: A tuple containing the list of
                chunked text sections and the list of chunk indices.
        """
        # Initialize an empty lists to store the chunks and chunk indices
        text_chunks = []
        chunk_index = []

        # Calculate the batch size for each process
        chunking_batch_size = ceil(len(texts) / chunking_jobs)
        # Split the texts and document IDs into batches
        batch_args = [
            (
                doc_ids[i * chunking_batch_size : (i + 1) * chunking_batch_size],
                texts[i * chunking_batch_size : (i + 1) * chunking_batch_size],
            )
            for i in range(chunking_jobs)
        ]

        # Use ProcessPoolExecutor to chunk the batches in parallel
        with ProcessPoolExecutor(max_workers=chunking_jobs) as executor:
            results = list(executor.map(self._chunk_batch, batch_args))

        # Flatten the results into a single list of chunked text sections
        # and a single list of chunk indices
        for batch_chunks, batch_chunk_indices in results:
            text_chunks.extend(batch_chunks)
            chunk_index.extend(batch_chunk_indices)

        # Return the chunked texts and coressponding chunk indices
        return text_chunks, chunk_index

    def _chunk_batch(self, args) -> Tuple[list[str], list[pl.DataFrame]]:
        """
        Chunks a batch of texts into smaller sections.

        Args:
            args (tuple[list[str], list[str]]): A tuple containing the batch
                of document IDs and the batch of texts.

        Returns:
            tuple[list[str], list[pl.DataFrame]]: A tuple containing the list
                of chunked text sections and the list of chunk indices.
        """
        batch_doc_ids, batch_texts = args

        # Initialize empty lists to store the chunks and chunk indices
        batch_chunks = []
        batch_chunk_indices = []

        # Iterate over the texts in the batch
        for doc_id, text in zip(batch_doc_ids, batch_texts):
            # Chunk the text into smaller sections
            new_chunks = self.chunk_text(text)
            n_chunks = len(new_chunks)

            # Create a DataFrame to store the chunk indices
            chunk_df = pl.DataFrame(
                {
                    "query_doc_id": [doc_id] * n_chunks,
                    "chunk_position": list(range(n_chunks)),
                    "n_chunks": [n_chunks] * n_chunks,
                }
            )

            # Add the chunked text sections and chunk indices to the lists
            batch_chunks.extend(new_chunks)
            batch_chunk_indices.append(chunk_df)

        # Return the chunked texts and the list of chunk indices
        return batch_chunks, batch_chunk_indices
