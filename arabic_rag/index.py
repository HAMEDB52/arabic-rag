"""فهرس هجين: BM25 على الكلمات + TF-IDF على رباعيات الحروف.

السبب في الدمج: النص العربي يعاني من اختلاف الصيغ الصرفية واللواحق،
فالمطابقة على مستوى الحروف تلتقط ما تفوّته مطابقة الكلمات الكاملة.
"""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .normalize import normalize, stems


class BM25:
    """تطبيق BM25 Okapi مباشر — بدون اعتماديات خارجية."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.corpus_size else 0.0
        self.doc_freqs = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.corpus_size, dtype=float)
        if not self.corpus_size:
            return out
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.doc_freqs], dtype=float)
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1))
            out += idf * (tf * (self.k1 + 1)) / np.where(denom == 0, 1, denom)
        return out


class HybridIndex:
    """فهرس يجمع BM25 ومتجهات حرفية، مع حفظ وتحميل من القرص."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25 | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._build()

    def _build(self) -> None:
        if not self.chunks:
            return
        token_corpus = [stems(c.text) for c in self.chunks]
        self._bm25 = BM25(token_corpus)
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform([normalize(c.text) for c in self.chunks])

    def search(self, query: str, top_k: int = 5, alpha: float = 0.6) -> list[tuple[Chunk, float]]:
        """alpha = وزن BM25 مقابل التشابه الحرفي."""
        if not self.chunks:
            return []
        lexical = self._bm25.scores(stems(query))
        char_vec = self._vectorizer.transform([normalize(query)])
        dense = (self._matrix @ char_vec.T).toarray().ravel()
        combined = alpha * _minmax(lexical) + (1 - alpha) * _minmax(dense)
        order = np.argsort(-combined)[:top_k]
        return [(self.chunks[i], float(combined[i])) for i in order if combined[i] > 0]

    # ---- الحفظ والتحميل ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"chunks": [c.to_dict() for c in self.chunks]}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        index = cls()
        index.chunks = [Chunk(**c) for c in payload["chunks"]]
        index._build()
        return index

    def export_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        # قيم متطابقة (مثل فهرس بمقطع واحد): تُعتبر مطابقة تامة إن كانت موجبة
        return np.ones_like(arr) if hi > 0 else np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
