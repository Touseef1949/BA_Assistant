import app


class FakeOpenAIChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["id"]


def test_text_model_factories_use_deepseek_flash(monkeypatch):
    monkeypatch.setattr(app, "OpenAIChat", FakeOpenAIChat)

    worker = app.make_worker_model()
    coordinator = app.make_coordinator_model()

    assert app.TEXT_ANALYSIS_MODEL_ID == "deepseek-v4-flash"
    assert worker.id == "deepseek-v4-flash"
    assert coordinator.id == "deepseek-v4-flash"
    assert worker.kwargs["base_url"] == app.DEEPSEEK_BASE_URL
    assert coordinator.kwargs["base_url"] == app.DEEPSEEK_BASE_URL


def test_simplified_analysis_modes():
    assert app.ANALYSIS_TYPES == ["Standard", "Interactive (Q&A)", "Deep Team"]
    assert "Standard" in app.ANALYSIS_TYPE_INFO


def test_extract_mermaid_code_from_fenced_block():
    markdown = """
Report

```mermaid
flowchart TD
    A[Start] --> B[Finish]
```
"""

    assert app.extract_mermaid_code(markdown).startswith("flowchart TD")


def test_generate_pdf_returns_bytes():
    pdf = app.generate_pdf("Test Project", "# Heading\n\nA short BA report.")

    assert isinstance(pdf, bytes)
    assert len(pdf) > 500


def test_history_path_does_not_include_email(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setattr(app, "safe_secret", lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default)

    path = app._history_path("person@example.com")

    assert str(tmp_path) in path
    assert "person@example.com" not in path
    assert path.endswith(".json")
