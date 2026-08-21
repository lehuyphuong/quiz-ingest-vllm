from quiz_ingest.ingest.chunking import chunk_text


def test_chunk_text_basic():
    text = "a" * 2500
    chunks = chunk_text(text, source="test", chunk_size=1000, overlap=100)
    assert len(chunks) == 3
    assert chunks[0].id == 0
    assert chunks[1].id == 1
    # overlap: chunk 1 should start 100 chars before chunk 0 ends
    assert chunks[0].text[-50:] in text


def test_chunk_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello world", source="t", chunk_size=10, overlap=10)


def test_chunk_text_empty_pieces_skipped():
    text = "  \n\n  " + "x" * 500
    chunks = chunk_text(text, source="t", chunk_size=1000, overlap=50)
    assert all(c.text.strip() for c in chunks)
