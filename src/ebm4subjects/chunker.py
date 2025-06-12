import nltk


class Chunker:
    def __init__(
        self,
        tokenizer: str,
        max_chunks: int | None,
        max_chunk_size: int | None,
        max_sentences: int | None,
    ):
        self.max_chunks = max_chunks if max_chunks else float("inf")
        self.max_chunk_size = max_chunk_size if max_chunk_size else float("inf")
        self.max_sentences = max_sentences if max_sentences else float("inf")

        self.tokenizer = nltk.data.load(tokenizer)

    def chunk_text(self, text: str) -> list[str]:
        chunks = []
        sentences = self.tokenizer.tokenize(text)
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
