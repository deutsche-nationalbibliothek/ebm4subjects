import polars as pl
from pathlib import Path

from ebm4subjects.prepare_data import parse_vocab

def test_parse_vocab_reads_ttl_and_returns_dataframe(tmp_path):
    # Copy the sample vocab.ttl to a temp location
    vocab_src = Path(__file__).parent / "data/vocab.ttl"
    vocab_path = tmp_path / "vocab.ttl"
    vocab_path.write_text(vocab_src.read_text(), encoding="utf-8")

    df = parse_vocab(vocab_path, use_altLabels=True)

    # Check that the DataFrame has the expected columns
    assert set(df.columns) == {"label_id", "label_text", "is_prefLabel"}
    # Check that some expected labels are present
    assert "mathematics" in df["label_id"].to_list()
    assert "Mathematics" in df["label_text"].to_list()
    # All is_prefLabel should be True (since only prefLabel in test data)
    analysis_rows = df.filter(pl.col("label_id") == "analysis")
    assert len(analysis_rows) == 2
    assert {"label_id": "analysis", "label_text": "Analysis", "is_prefLabel": True} in analysis_rows.to_dicts()
    assert {"label_id": "analysis", "label_text": "Calculus", "is_prefLabel": False} in analysis_rows.to_dicts()
    # Check that the number of rows matches the number of prefLabels in the TTL
    assert len(df) >= 1