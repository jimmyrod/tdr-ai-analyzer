from app.chunker import TextChunker
from app.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace_and_removes_control_chars():
    raw = "  OBJETO\r\n\r\nContratar\t\tservicio\x00 de   backup.  "

    cleaned = clean_text(raw)

    assert cleaned == "OBJETO\nContratar servicio de backup."


def test_chunker_keeps_overlap_and_metadata():
    text = " ".join(f"palabra{i}" for i in range(80))
    chunker = TextChunker(chunk_size=140, overlap=30)

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert chunks[0].text
    assert chunks[1].start_char < chunks[0].end_char
    assert chunks[-1].end_char == len(text)
