from arabic_rag import RAGPipeline, evaluate

DOCS = {
    "hr": "يستحق الموظف إجازة سنوية مدتها واحد وعشرون يوم عمل بعد سنة كاملة من الخدمة.",
    "it": "يجب الإبلاغ عن الحوادث الأمنية خلال ساعة واحدة من اكتشافها.",
}
DATASET = [
    {"question": "كم مدة الإجازة السنوية؟", "doc_id": "hr", "answer_contains": ["واحد وعشرون"]},
    {"question": "خلال كم ساعة يتم الإبلاغ عن حادثة أمنية؟", "doc_id": "it", "answer_contains": ["ساعة واحدة"]},
]


def test_evaluate_reports_all_metrics():
    p = RAGPipeline()
    for doc_id, text in DOCS.items():
        p.add_text(doc_id, text)
    result = evaluate(p, DATASET, top_k=2)
    assert result.n == 2
    assert result.recall_at_k == 1.0
    assert result.citation_rate == 1.0
    assert result.grounded_rate == 1.0
    assert result.answer_accuracy == 1.0
