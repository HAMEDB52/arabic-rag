"""خط RAG الكامل: استيعاب المستندات ← استرجاع ← توليد مُستشهَد."""
from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_document
from .generator import Answer, ExtractiveGenerator, Generator
from .index import HybridIndex


class RAGPipeline:
    def __init__(
        self,
        *,
        generator: Generator | None = None,
        top_k: int = 4,
        target_chars: int = 380,
        alpha: float = 0.6,
    ) -> None:
        self.index = HybridIndex()
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k
        self.target_chars = target_chars
        self.alpha = alpha

    # ---- الاستيعاب ----
    def add_text(self, doc_id: str, text: str, meta: dict | None = None) -> list[Chunk]:
        chunks = chunk_document(doc_id, text, target_chars=self.target_chars, meta=meta)
        self.index.add(chunks)
        return chunks

    def add_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        text = _read_any(path)
        return self.add_text(path.stem, text, {"source": str(path)})

    def add_directory(self, directory: str | Path, patterns: tuple[str, ...] = ("*.txt", "*.md", "*.pdf")) -> int:
        count = 0
        for pattern in patterns:
            for path in sorted(Path(directory).rglob(pattern)):
                self.add_file(path)
                count += 1
        return count

    # ---- الاستعلام ----
    def retrieve(self, question: str, top_k: int | None = None) -> list[Chunk]:
        hits = self.index.search(question, top_k=top_k or self.top_k, alpha=self.alpha)
        return [chunk for chunk, _ in hits]

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        contexts = self.retrieve(question, top_k)
        # تمرير أوزان IDF للمولّد الاستخلاصي إن كان يدعمها
        if hasattr(self.generator, "idf") and self.index._bm25 is not None:
            self.generator.idf = self.index._bm25.idf
        return self.generator.generate(question, contexts)


def _read_any(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text(encoding="utf-8")
