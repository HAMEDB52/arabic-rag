"""واجهة سطر الأوامر: arabic-rag ask / eval / index"""
from __future__ import annotations

import argparse
import json
import sys

from .evaluate import evaluate, load_dataset
from .pipeline import RAGPipeline


def build(docs: str, top_k: int) -> RAGPipeline:
    pipeline = RAGPipeline(top_k=top_k)
    n = pipeline.add_directory(docs)
    print(f"تم استيعاب {n} مستنداً في {len(pipeline.index)} مقطعاً.", file=sys.stderr)
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arabic-rag", description="نظام RAG عربي مع استشهادات")
    parser.add_argument("--docs", default="data/sample_docs", help="مجلد المستندات")
    parser.add_argument("--top-k", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="طرح سؤال")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")

    ev = sub.add_parser("eval", help="تشغيل التقييم")
    ev.add_argument("--dataset", default="data/eval/qa.json")

    sub.add_parser("index", help="عرض إحصاءات الفهرس")

    args = parser.parse_args(argv)
    pipeline = build(args.docs, args.top_k)

    if args.command == "ask":
        answer = pipeline.ask(args.question)
        if args.json:
            print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        else:
            print("\nالإجابة:\n" + answer.text + "\n")
            print("المصادر:")
            for c in answer.citations:
                print(f"  {c.marker} {c.doc_id} (موضع {c.start}-{c.end})")
        return 0

    if args.command == "eval":
        print(evaluate(pipeline, load_dataset(args.dataset), top_k=args.top_k))
        return 0

    print(json.dumps({"chunks": len(pipeline.index),
                      "docs": len({c.doc_id for c in pipeline.index.chunks})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
