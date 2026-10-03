"""تقطيع المستندات إلى مقاطع قابلة للاسترجاع مع تتبّع الموضع الأصلي."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

SENT_SPLIT = re.compile(r"(?<=[.!؟?])\s+|\n{2,}")


@dataclass
class Chunk:
    """مقطع نصي مع مرجعه الأصلي — المرجع هو ما يجعل الاستشهاد ممكناً."""

    chunk_id: str
    doc_id: str
    text: str                       # النص الأصلي كما هو (للعرض والاستشهاد)
    start: int                      # موضع البداية داخل المستند الأصلي
    end: int
    order: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """إرجاع الجمل مع مواضعها (start, end, sentence)."""
    out, pos = [], 0
    for part in SENT_SPLIT.split(text):
        if part is None:
            continue
        stripped = part.strip()
        if not stripped:
            continue
        start = text.find(stripped, pos)
        if start < 0:
            start = pos
        end = start + len(stripped)
        out.append((start, end, stripped))
        pos = end
    return out


def chunk_document(
    doc_id: str,
    text: str,
    *,
    target_chars: int = 380,
    overlap_sentences: int = 1,
    meta: dict | None = None,
) -> list[Chunk]:
    """تقطيع واعٍ بالجُمل: لا يقطع الجملة في منتصفها، مع تداخل يحفظ السياق."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buf: list[tuple[int, int, str]] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        start, end = buf[0][0], buf[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{len(chunks)}",
                doc_id=doc_id,
                text=text[start:end].strip(),
                start=start,
                end=end,
                order=len(chunks),
                meta=dict(meta or {}),
            )
        )
        buf = buf[-overlap_sentences:] if overlap_sentences else []
        size = sum(len(s[2]) for s in buf)

    for sent in sentences:
        if size and size + len(sent[2]) > target_chars:
            flush()
        buf.append(sent)
        size += len(sent[2])
    flush()
    # إزالة أي مقطع مكرر ناتج عن التداخل في النهاية
    seen, unique = set(), []
    for c in chunks:
        key = (c.start, c.end)
        if key in seen:
            continue
        seen.add(key)
        c.order = len(unique)
        c.chunk_id = f"{c.doc_id}::{c.order}"
        unique.append(c)
    return unique
