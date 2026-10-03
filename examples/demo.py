"""عرض سريع: استيعاب المستندات، سؤال، استشهاد، ثم تقييم."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_rag import RAGPipeline, evaluate, load_dataset

ROOT = Path(__file__).resolve().parents[1]

pipeline = RAGPipeline(top_k=4)
pipeline.add_directory(ROOT / "data" / "sample_docs")
print(f"الفهرس: {len(pipeline.index)} مقطعاً\n")

for question in [
    "كم عدد أيام الإجازة السنوية؟",
    "ما حد صلاحية مدير الإدارة في المشتريات؟",
    "خلال كم ساعة يجب الإبلاغ عن حادثة أمنية؟",
    "ما هي عاصمة اليابان؟",          # سؤال خارج نطاق المستندات
]:
    answer = pipeline.ask(question)
    print(f"س: {question}")
    print(f"ج: {answer.text}")
    for c in answer.citations:
        print(f"   {c.marker} {c.doc_id} [{c.start}:{c.end}]")
    print()

print(evaluate(pipeline, load_dataset(ROOT / "data" / "eval" / "qa.json"), top_k=4))
