import pytest

from arabic_rag import RAGPipeline

LEAVE = """سياسة الإجازات
يستحق الموظف إجازة سنوية مدفوعة الأجر مدتها واحد وعشرون يوم عمل بعد إتمام سنة كاملة.
يقدَّم طلب الإجازة قبل موعدها بخمسة عشر يوماً على الأقل."""

SECURITY = """سياسة أمن المعلومات
يجب الإبلاغ عن أي حادثة أمنية خلال ساعة واحدة من اكتشافها.
تُراجع صلاحيات الوصول كل ثلاثة أشهر."""


@pytest.fixture
def pipeline() -> RAGPipeline:
    p = RAGPipeline(top_k=3)
    p.add_text("leave", LEAVE)
    p.add_text("security", SECURITY)
    return p


def test_retrieves_correct_document(pipeline):
    chunks = pipeline.retrieve("خلال كم ساعة يجب الإبلاغ عن حادثة أمنية؟")
    assert chunks[0].doc_id == "security"


def test_answer_is_cited_and_grounded(pipeline):
    answer = pipeline.ask("كم مدة الإجازة السنوية؟")
    assert answer.citations, "يجب أن تحتوي كل إجابة على استشهاد"
    for citation in answer.citations:
        assert citation.marker in answer.text
        source = next(c for c in answer.contexts if c.chunk_id == citation.chunk_id)
        assert citation.quote in source.text


def test_unknown_question_is_refused_when_index_empty():
    answer = RAGPipeline().ask("سؤال بلا مستندات")
    assert "لا توجد معلومات كافية" in answer.text
    assert answer.citations == []


def test_citation_offsets_point_at_original_text(pipeline):
    answer = pipeline.ask("متى يقدم طلب الإجازة؟")
    for citation in answer.citations:
        assert citation.end > citation.start


def test_out_of_scope_question_is_refused(pipeline):
    answer = pipeline.ask("ما هي عاصمة اليابان؟")
    assert "لا توجد معلومات كافية" in answer.text
    assert answer.citations == []


@pytest.mark.parametrize("question", ["ما سعر برميل النفط اليوم؟", "ما هي عقوبة السفر إلى المريخ؟"])
def test_function_or_unit_words_alone_do_not_produce_an_answer(pipeline, question):
    # «إلى» و«اليوم» تتطابق معجمياً مع المستندات، لكنها لا تجعل السؤال ضمن النطاق
    assert pipeline.ask(question).citations == []
