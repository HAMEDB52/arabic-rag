from arabic_rag.normalize import light_stem, normalize, stems, strip_diacritics, tokenize


def test_strip_diacritics():
    assert strip_diacritics("الذِّكاءُ الاصطناعيّ") == "الذكاء الاصطناعي"


def test_normalize_unifies_letters_and_digits():
    assert normalize("إجازة أحمد ٢٠٢٤") == "اجازه احمد 2024"
    assert normalize("آلاء") == normalize("الاء")


def test_tokenize_drops_punctuation():
    assert tokenize("ما هي سياسة الإجازات؟") == ["ما", "هي", "سياسه", "الاجازات"]


def test_light_stem_handles_prefix_and_suffix():
    assert light_stem("الإجازات".replace("إ", "ا")) == light_stem("اجازات")
    assert light_stem("والمشتريات") == light_stem("مشتريات")


def test_stems_match_across_definite_article():
    assert set(stems("الإجازة السنوية")) & set(stems("إجازة سنوية"))
