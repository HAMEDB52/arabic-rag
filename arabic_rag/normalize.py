"""تطبيع النص العربي — Arabic text normalization utilities."""
from __future__ import annotations

import re
import unicodedata

# حركات التشكيل والتطويل
DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC_PUNCT = "،؛؟«»ـ"

_ALEF = re.compile(r"[آأإٱٲٳ]")      # آ أ إ ٱ
_YA = re.compile(r"[ىی]")                                  # ى ی
_TA_MARBUTA = re.compile(r"ة")                                  # ة
_WAW = re.compile(r"[ؤ]")                                       # ؤ
_HAMZA_YA = re.compile(r"[ئ]")                                  # ئ
_SPACES = re.compile(r"\s+")

# الأرقام العربية-الهندية -> أرقام لاتينية
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def strip_diacritics(text: str) -> str:
    """إزالة التشكيل والتطويل."""
    return DIACRITICS.sub("", text).replace(TATWEEL, "")


def normalize(text: str, *, keep_diacritics: bool = False) -> str:
    """تطبيع شامل: توحيد الألف والياء والتاء المربوطة، وإزالة التشكيل والمسافات الزائدة.

    يُستخدم للفهرسة والمطابقة فقط — النص الأصلي يُحفظ كما هو للعرض والاستشهاد.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGITS)
    if not keep_diacritics:
        text = strip_diacritics(text)
    text = _ALEF.sub("ا", text)
    text = _YA.sub("ي", text)
    text = _TA_MARBUTA.sub("ه", text)
    text = _WAW.sub("و", text)
    text = _HAMZA_YA.sub("ي", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """تقطيع إلى كلمات بعد التطبيع."""
    normalized = normalize(text)
    for ch in ARABIC_PUNCT:
        normalized = normalized.replace(ch, " ")
    return [t for t in re.split(r"[^\w؀-ۿ]+", normalized) if t]


# قائمة مختصرة لكلمات الوقف الشائعة
STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "ما", "لا", "ان", "أن", "إن", "كان", "كانت", "هو", "هي", "قد",
    "او", "أو", "و", "ثم", "كل", "بين", "عند", "بعد", "قبل", "حيث", "هل",
    "كم", "متى", "ماذا", "خلال", "يجب", "ينبغي",
}
# تُطبَّع كلمات الوقف بنفس دالة النص؛ وإلا فلن تطابق «إلى» صيغتها المطبّعة «الي»
STOPWORDS |= {normalize(w) for w in STOPWORDS}


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


# ---- تجذير خفيف (Light stemming) ----
# السبب: المطابقة الحرفية تفشل بين «الإجازة السنوية» و«إجازة سنوية».
_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "وا", "و", "ف", "ب", "ك", "ل")
_SUFFIXES = ("اتها", "اتهم", "ياتها", "ونها", "ينها", "هما", "كما", "هم", "هن",
             "ها", "ات", "ون", "ين", "ان", "ية", "يه", "تي", "ه", "ة", "ي", "ا")


def light_stem(token: str, min_len: int = 3) -> str:
    """إزالة السوابق واللواحق الشائعة مع الحفاظ على جذع ذي معنى."""
    if len(token) <= min_len:
        return token
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= min_len:
            token = token[len(p):]
            break
    for s in _SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= min_len:
            token = token[: -len(s)]
            break
    return token


def stems(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """مفاتيح المطابقة النهائية المستخدمة في الفهرسة والتقييم."""
    toks = tokenize(text)
    if drop_stopwords:
        filtered = remove_stopwords(toks)
        toks = filtered or toks
    return [light_stem(t) for t in toks]
