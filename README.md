# arabic-rag — استرجاع معزّز بالتوليد للمستندات العربية

نظام RAG مبني خصيصاً للنص العربي: **كل جملة في الإجابة مرتبطة بموضعها الدقيق في المستند المصدر**، والنظام يمتنع عن الإجابة حين لا تدعمه المستندات.

[![tests](https://img.shields.io/badge/tests-18%20passed-brightgreen)](#الاختبارات)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](#المتطلبات)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## لماذا هذا المشروع؟

معظم حلول RAG الجاهزة تنهار على المحتوى العربي لأسباب محددة:

| المشكلة | الأثر | المعالجة هنا |
|---|---|---|
| التشكيل وتعدّد صور الألف والياء والتاء المربوطة | «الإجازة» لا تطابق «الاجازه» | تطبيع كامل قبل الفهرسة مع حفظ النص الأصلي للعرض |
| اللواحق والسوابق الصرفية | «والمشتريات» لا تطابق «مشتريات» | تجذير خفيف (light stemming) للسوابق واللواحق الشائعة |
| التقطيع الأعمى بعدد الحروف | قطع الجملة في منتصفها يفسد المعنى والاستشهاد | تقطيع واعٍ بالجُمل مع تداخل محسوب |
| إجابات بلا مصدر | لا يمكن التحقق منها | كل استشهاد يحمل `doc_id` وموضع البداية والنهاية والنص المقتبس |
| غياب القياس | لا تعرف هل تحسّن النظام أم ساء | أداة تقييم مدمجة: Recall@k و MRR ونسبة التأصيل ودقة الإجابة |

## المزايا

- **استرجاع هجين**: BM25 على الكلمات المجذّرة + TF-IDF على رباعيات الحروف، بدمج موزون. المطابقة الحرفية تلتقط ما تفوّته مطابقة الكلمات.
- **استشهادات قابلة للتحقق**: لكل مرجع `[1]` نصٌّ مقتبس وموضع `start:end` داخل المستند الأصلي — يمكن فتح المستند والتأكد.
- **امتناع مُتعمَّد**: دون عتبة تداخل معجمي يرد النظام «لا توجد معلومات كافية» بدل اختلاق إجابة.
- **يعمل بلا إنترنت وبلا مفاتيح API**: المولّد الافتراضي استخلاصي (يبني الإجابة من جُمل المصدر نفسها)، ويصلح كخط أساس تُقاس عليه أي ترقية لاحقة.
- **قابل للترقية**: واجهة `Generator` واحدة — بدّل `ExtractiveGenerator` بـ `OpenAIGenerator` دون تغيير بقية النظام.
- **واجهات جاهزة**: مكتبة Python و CLI و HTTP API عبر FastAPI.

## التثبيت

```bash
git clone https://github.com/HAMEDB52/arabic-rag.git
cd arabic-rag
pip install -r requirements-dev.txt
```

## البداية السريعة

```python
from arabic_rag import RAGPipeline

rag = RAGPipeline(top_k=4)
rag.add_directory("data/sample_docs")        # txt / md / pdf

answer = rag.ask("كم عدد أيام الإجازة السنوية؟")
print(answer.text)
for c in answer.citations:
    print(c.marker, c.doc_id, f"{c.start}:{c.end}", c.quote[:60])
```

### سطر الأوامر

```bash
python -m arabic_rag.cli ask "ما حد صلاحية مدير الإدارة في المشتريات؟"
python -m arabic_rag.cli ask "سؤالك" --json
python -m arabic_rag.cli eval --dataset data/eval/qa.json
```

### واجهة HTTP

```bash
uvicorn arabic_rag.api:app --reload
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"خلال كم ساعة يجب الإبلاغ عن حادثة أمنية؟"}'
```

### استخدام نموذج لغوي بدل المولّد الاستخلاصي

```python
from arabic_rag import RAGPipeline, OpenAIGenerator

rag = RAGPipeline(generator=OpenAIGenerator(model="gpt-4o-mini"))  # يقرأ OPENAI_API_KEY
```

## المعمارية

```
المستندات ──► تطبيع عربي ──► تقطيع واعٍ بالجُمل ──► فهرس هجين
                                                      │
                     السؤال ──► تطبيع + تجذير ────────┤
                                                      ▼
                                          استرجاع (BM25 ⊕ حروف)
                                                      │
                                                      ▼
                                   مولّد (استخلاصي | LLM) ──► إجابة + استشهادات
```

| الوحدة | المسؤولية |
|---|---|
| `normalize.py` | تطبيع، تقطيع كلمات، تجذير خفيف |
| `chunking.py` | تقطيع بالجُمل مع حفظ المواضع الأصلية |
| `index.py` | BM25 من الصفر + TF-IDF حرفي + دمج موزون |
| `generator.py` | المولّد الاستخلاصي، مولّد OpenAI، نموذج الاستشهاد |
| `pipeline.py` | ربط الاستيعاب بالاسترجاع بالتوليد |
| `evaluate.py` | Recall@k، MRR، نسبة الاستشهاد، التأصيل، دقة الإجابة |

## التقييم

على مجموعة الأسئلة المرفقة (`data/eval/qa.json`، ٨ أسئلة على ٣ مستندات سياسات):

```
Recall@4       : 100.00%
MRR            : 1.000
نسبة الاستشهاد : 100.00%
نسبة التأصيل   : 100.00%
دقة الإجابة     : 100.00%
```

> هذه مجموعة صغيرة مقصودة للتحقق من صحة الخط كاملاً، لا لإثبات جودة على نطاق واسع. ولإعادة إنتاجها: `python -m arabic_rag.cli eval`.

**تعريف المقاييس**: `Recall@k` وجود المستند الصحيح ضمن أعلى k نتيجة · `MRR` مقلوب رتبة أول نتيجة صحيحة · `نسبة التأصيل` أن يكون كل نص مُستشهَد به موجوداً حرفياً داخل مقطع مُسترجَع · `دقة الإجابة` ظهور الصيغة المرجعية داخل نص الإجابة.

## الاختبارات

```bash
python -m pytest -q      # 18 اختباراً
```

تغطي: التطبيع والتجذير · صحة مواضع المقاطع · ترتيب الاسترجاع · تأصيل الاستشهادات · الامتناع عن الأسئلة خارج النطاق · واجهة HTTP.

## الحدود المعروفة

- الاسترجاع **معجمي** لا دلالي: سؤال بمرادفات لا تظهر في النص قد يفشل. الترقية الطبيعية إضافة متجهات دلالية ودمجها بـ RRF.
- المولّد الافتراضي **استخلاصي**: ينقل جُملاً من المصدر ولا يعيد صياغتها، فالإجابة دقيقة لكنها ليست مُركَّبة.
- التجذير الخفيف قائم على قواعد، وقد يفرط في القطع مع كلمات نادرة.
- لم يُختبر على مستندات ممسوحة ضوئياً — تمريرها يحتاج طبقة OCR قبل الاستيعاب.

## خارطة الطريق

- [ ] متجهات دلالية (sentence-transformers) ودمج RRF مع BM25
- [ ] إعادة ترتيب بنموذج cross-encoder
- [ ] دعم الجداول داخل PDF
- [ ] تقييم على مجموعة أكبر ومتعددة اللهجات

## الترخيص

MIT — انظر [LICENSE](LICENSE).

---

## English summary

**arabic-rag** is a retrieval-augmented generation pipeline built for Arabic documents. It normalizes Arabic orthography (diacritics, alef/ya/ta-marbuta variants), applies light stemming, chunks on sentence boundaries while preserving source offsets, and retrieves with a hybrid of from-scratch BM25 and character n-gram TF-IDF. Every answer sentence carries a citation with the exact `doc_id` and character span in the source, and the system abstains when lexical support is insufficient. Runs fully offline with an extractive generator; swap in an LLM generator through a single interface. Includes a CLI, a FastAPI service, and an evaluation harness (Recall@k, MRR, citation grounding, answer accuracy).
