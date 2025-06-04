import nltk


class Chunker:
    def __init__(
        self,
        max_chunks: int | None,
        max_chunk_size: int | None,
        max_sentences: int | None,
    ):
        self.max_chunks = max_chunks if max_chunks else float("inf")
        self.max_chunk_size = max_chunk_size if max_chunk_size else float("inf")
        self.max_sentences = max_sentences if max_sentences else float("inf")

        self.tokenizer = nltk.data.load("tokenizers/punkt/german.pickle")

    def chunk_text(self, text: str, doc_id: int) -> list[(str, int)]:
        chunks = []
        sentences = self.tokenizer.tokenize(text)
        sentences = sentences[: self.max_sentences]

        current_chunk = []
        for sentence in sentences:
            if len(" ".join(current_chunk)) < self.max_chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append((" ".join(current_chunk), doc_id))
                current_chunk = [sentence]
                if len(chunks) == self.max_chunks:
                    break

        if current_chunk and len(chunks) < self.max_chunks:
            chunks.append((" ".join(current_chunk), doc_id))

        return chunks

    def process_batch(self, texts, doc_ids):
        chunked_texts = []
        for text, doc_id in zip(texts, doc_ids):
            chunked_texts.extend(self.chunk_text(text, doc_id))

        return chunked_texts
