"""مولّدات الإجابة: افتراضي استخلاصي يعمل بلا إنترنت، واختياري عبر LLM."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import re

from .chunking import Chunk, split_sentences
from .normalize import normalize, stems

# إشارات السؤال الكمّي ومؤشرات الجمل التي تحمل مقداراً
QUANTITY_CUES = {"كم", "عدد", "مدة", "مده", "نسبة", "نسبه", "متى", "مقدار", "قيمة", "قيمه"}
NUMBER_WORDS = (
    "واحد|اثن|ثلاث|اربع|خمس|ست|سبع|ثمان|تسع|عشر|عشرون|عشرين|ثلاثون|ثلاثين|"
    "اربعون|اربعين|خمسون|خمسين|مئة|مائة|مئتي|مائتي|الف|الاف|يوم|ساعة|ساعه|شهر|سنة|سنه"
)
NUMBER_RE = re.compile(r"\d+|" + NUMBER_WORDS)

# كلمات تصف المقدار المطلوب لا موضوع السؤال — تُخفَّض أوزانها في الأسئلة الكمّية
UNIT_TERMS = {"كم", "عدد", "يوم", "ايام", "ساعه", "ساعات", "شهر", "اشهر",
              "سنه", "سنوات", "مده", "نسبه", "مقدار", "قيمه", "ريال"}


@dataclass
class Citation:
    marker: str          # [1]
    chunk_id: str
    doc_id: str
    quote: str           # النص الأصلي المقتبس — يتيح التحقق اليدوي
    start: int
    end: int


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    contexts: list[Chunk]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [c.__dict__ for c in self.citations],
        }


class Generator(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> Answer: ...


class ExtractiveGenerator:
    """يبني الإجابة من جُمل المصدر نفسها.

    ميزته أنه لا يهلوس: كل جملة في الإجابة مأخوذة حرفياً من مقطع مفهرس،
    ولذلك يصلح كخط أساس (baseline) تُقاس عليه جودة أي مولّد توليدي.
    """

    NO_ANSWER = "لا توجد معلومات كافية في المستندات للإجابة على هذا السؤال."

    def __init__(
        self,
        max_sentences: int = 3,
        idf: dict[str, float] | None = None,
        min_overlap: float = 0.08,
    ) -> None:
        self.max_sentences = max_sentences
        # عتبة الامتناع: دون هذا التداخل تُعتبر المستندات غير ذات صلة
        self.min_overlap = min_overlap
        # أوزان IDF من الفهرس — تمنع الكلمات الشائعة من ترجيح جملة غير مناسبة
        self.idf: dict[str, float] = idf or {}

    def _weight(self, term: str, *, damp_units: bool = False) -> float:
        w = self.idf.get(term, 1.0)
        if damp_units and term in UNIT_TERMS:
            w *= 0.3   # «كم عدد أيام الإجازة» يسأل عن الإجازة، لا عن الأيام
        return w

    def generate(self, question: str, contexts: list[Chunk]) -> Answer:
        if not contexts:
            return Answer(question, self.NO_ANSWER, [], [])

        q_terms = set(stems(question))
        wants_quantity = bool(set(stems(question, drop_stopwords=False)) & QUANTITY_CUES)
        q_mass = sum(self._weight(t, damp_units=wants_quantity) for t in q_terms) or 1.0
        scored: list[tuple[float, int, str, Chunk, int, int, float]] = []
        for rank, chunk in enumerate(contexts):
            for s_start, s_end, sent in split_sentences(chunk.text):
                terms = set(stems(sent))
                if not terms:
                    continue
                shared = q_terms & terms
                overlap = sum(self._weight(t, damp_units=wants_quantity) for t in shared) / q_mass
                # تفضيل الجمل المركّزة: القسمة على جذر الطول تحدّ من أثر الجمل الطويلة
                score = overlap / (1 + len(terms) ** 0.5 / 10) - 0.03 * rank
                # مطابقة نوع الإجابة: سؤال كمّي يرجّح الجملة التي تحمل مقداراً
                if wants_quantity and NUMBER_RE.search(normalize(sent)):
                    score += 0.12
                scored.append(
                    (score, rank, sent, chunk, chunk.start + s_start, chunk.start + s_end, overlap)
                )

        scored.sort(key=lambda x: (-x[0], x[1]))
        # الامتناع خير من الهلوسة: بلا تداخل معجمي كافٍ لا تُبنى إجابة
        if not scored or max(s[6] for s in scored) < self.min_overlap:
            return Answer(question, self.NO_ANSWER, [], contexts)
        picked = [s for s in scored if s[0] > 0][: self.max_sentences] or scored[:1]

        parts, citations, seen, used_sents = [], [], {}, set()
        for _, _, sent, chunk, abs_start, abs_end, _ in picked:
            if sent in used_sents:
                continue
            used_sents.add(sent)
            marker = seen.get(chunk.chunk_id)
            if marker is None:
                marker = f"[{len(seen) + 1}]"
                seen[chunk.chunk_id] = marker
                citations.append(
                    Citation(marker, chunk.chunk_id, chunk.doc_id, sent, abs_start, abs_end)
                )
            parts.append(f"{sent} {marker}")
        return Answer(question, " ".join(parts), citations, contexts)


class OpenAIGenerator:
    """مولّد عبر واجهة متوافقة مع OpenAI — يُفعَّل فقط عند توفّر المفتاح.

    يُلزَم النموذج بالاستشهاد بأرقام المقاطع، ويُرفض أي ادعاء بلا مصدر.
    """

    SYSTEM = (
        "أنت مساعد يجيب بالعربية اعتماداً على المقاطع المرقّمة فقط. "
        "ضع رقم المقطع بين قوسين مربعين بعد كل معلومة، مثل [1]. "
        "إذا لم تجد الإجابة في المقاطع فقل: لا توجد معلومات كافية في المستندات."
    )

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY غير متوفر — استخدم ExtractiveGenerator بدلاً منه.")

    def generate(self, question: str, contexts: list[Chunk]) -> Answer:
        from openai import OpenAI  # اعتمادية اختيارية

        blocks = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(contexts))
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"المقاطع:\n{blocks}\n\nالسؤال: {question}"},
            ],
        )
        text = resp.choices[0].message.content or ""
        citations = [
            Citation(f"[{i + 1}]", c.chunk_id, c.doc_id, c.text[:200], c.start, c.end)
            for i, c in enumerate(contexts)
            if f"[{i + 1}]" in text
        ]
        return Answer(question, text, citations, contexts)
