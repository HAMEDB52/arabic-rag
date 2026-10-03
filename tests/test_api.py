import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from arabic_rag import api  # noqa: E402


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_pipeline", None)
    monkeypatch.setenv("ARABIC_RAG_DOCS", str(tmp_path))
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_then_ask(client):
    r = client.post("/ingest", json={"doc_id": "hr", "text": "الإجازة السنوية واحد وعشرون يوم عمل."})
    assert r.json()["chunks"] >= 1
    data = client.post("/ask", json={"question": "كم مدة الإجازة السنوية؟"}).json()
    assert data["citations"]
