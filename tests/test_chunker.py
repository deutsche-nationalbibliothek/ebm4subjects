from ebm4subjects.chunker import ProcessArgs, chunk, split_sentence


class TestSplitSentence:
    """Tests for split_sentence."""

    def test_returns_sentence_unchanged_when_within_limit(self):
        sentence = "short sentence"
        assert split_sentence(sentence, max_chunk_length=50) == [sentence]

    def test_splits_at_space_near_middle(self):
        sentence = "a" * 10 + " " + "b" * 10
        pieces = split_sentence(sentence, max_chunk_length=15)
        assert pieces == ["a" * 10, "b" * 10]

    def test_all_pieces_within_max_length(self):
        sentence = " ".join(["word"] * 50)
        pieces = split_sentence(sentence, max_chunk_length=20)
        assert all(len(piece) <= 20 for piece in pieces)

    def test_recombining_pieces_preserves_words(self):
        sentence = "the quick brown fox jumps over the lazy dog"
        pieces = split_sentence(sentence, max_chunk_length=12)
        assert " ".join(pieces).split() == sentence.split()

    def test_falls_back_to_hard_split_without_spaces(self):
        sentence = "a" * 30
        pieces = split_sentence(sentence, max_chunk_length=10)
        assert pieces == ["a" * 10, "a" * 10, "a" * 10]

    def test_empty_sentence_returns_single_empty_piece(self):
        assert split_sentence("", max_chunk_length=10) == [""]


class TestChunk:
    """Tests for chunk."""

    def test_groups_short_sentences_into_single_chunk(self):
        sentences = ["one", "two", "three"]
        process_args = ProcessArgs(
            max_sentence_count=10, max_chunk_length=100, max_chunk_count=10
        )
        result = chunk(sentences, process_args)
        assert result == ["one two three"]

    def test_starts_new_chunk_once_length_exceeded(self):
        sentences = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
        process_args = ProcessArgs(
            max_sentence_count=10, max_chunk_length=10, max_chunk_count=10
        )
        result = chunk(sentences, process_args)
        assert result == ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]

    def test_respects_max_sentence_count(self):
        sentences = ["one", "two", "three", "four"]
        process_args = ProcessArgs(
            max_sentence_count=2, max_chunk_length=100, max_chunk_count=10
        )
        result = chunk(sentences, process_args)
        assert result == ["one two"]

    def test_respects_max_chunk_count(self):
        sentences = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
        process_args = ProcessArgs(
            max_sentence_count=10, max_chunk_length=10, max_chunk_count=2
        )
        result = chunk(sentences, process_args)
        assert result == ["aaaaaaaaaa", "bbbbbbbbbb"]

    def test_empty_sentence_list_returns_empty_chunks(self):
        process_args = ProcessArgs(
            max_sentence_count=10, max_chunk_length=100, max_chunk_count=10
        )
        assert chunk([], process_args) == []

    def test_oversized_sentence_is_split_before_chunking(self):
        sentence = " ".join(["word"] * 20)
        process_args = ProcessArgs(
            max_sentence_count=10, max_chunk_length=15, max_chunk_count=10
        )
        result = chunk([sentence], process_args)
        # the oversized sentence must be broken up into several chunks
        assert len(result) > 1
        assert sentence not in result
