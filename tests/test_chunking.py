from arabic_rag.chunking import chunk_document, split_sentences

TEXT = ("الجملة الأولى في المستند. الجملة الثانية تشرح التفاصيل. "
        "الجملة الثالثة تضيف شرطاً مهماً. الجملة الرابعة تختم الفقرة.")


def test_split_sentences_returns_positions():
    sents = split_sentences(TEXT)
    assert len(sents) == 4
    for start, end, sent in sents:
        assert TEXT[start:end] == sent


def test_chunks_cover_source_and_keep_offsets():
    chunks = chunk_document("doc", TEXT, target_chars=60)
    assert len(chunks) > 1
    for c in chunks:
        assert TEXT[c.start:c.end].strip() == c.text
        assert c.chunk_id.startswith("doc::")


def test_empty_document_yields_no_chunks():
    assert chunk_document("doc", "   ") == []
