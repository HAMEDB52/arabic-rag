"""تقييم الاسترجاع والاستشهاد — لا قيمة لنظام RAG بلا قياس."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalize as norm
from .pipeline import RAGPipeline


@dataclass
class EvalResult:
    n: int
    recall_at_k: float
    mrr: float
    citation_rate: float
    grounded_rate: float
    answer_accuracy: float

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self) -> str:
        return (
            f"عدد الأسئلة: {self.n}\n"
            f"Recall@k      : {self.recall_at_k:.2%}\n"
            f"MRR           : {self.mrr:.3f}\n"
            f"نسبة الاستشهاد : {self.citation_rate:.2%}\n"
            f"نسبة التأصيل   : {self.grounded_rate:.2%}\n"
            f"دقة الإجابة    : {self.answer_accuracy:.2%}"
        )


def load_dataset(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(pipeline: RAGPipeline, dataset: list[dict], top_k: int = 4) -> EvalResult:
    """كل عنصر: {"question": ..., "doc_id": ..., "answer_contains": [...]}"""
    recall = mrr = cited = grounded = correct = 0.0
    for item in dataset:
        chunks = pipeline.retrieve(item["question"], top_k=top_k)
        doc_ids = [c.doc_id for c in chunks]
        if item["doc_id"] in doc_ids:
            recall += 1
            mrr += 1 / (doc_ids.index(item["doc_id"]) + 1)

        answer = pipeline.generator.generate(item["question"], chunks)
        if answer.citations:
            cited += 1
        # التأصيل: هل كل نص مُستشهَد به موجود فعلاً داخل مقطع مُسترجَع؟
        if answer.citations and all(
            any(c.quote in chunk.text for chunk in chunks) for c in answer.citations
        ):
            grounded += 1
        # الدقة من طرف إلى طرف: هل ظهرت الصيغة المرجعية داخل نص الإجابة؟
        expected = item.get("answer_contains") or []
        if expected and any(norm(e) in norm(answer.text) for e in expected):
            correct += 1

    n = len(dataset) or 1
    return EvalResult(len(dataset), recall / n, mrr / n, cited / n, grounded / n, correct / n)
