import json

import pytest

import app
import core.analyzer
from services import history_store, report_utils


class FakeOpenAIChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["id"]


def test_text_model_factories_use_deepseek_flash(monkeypatch):
    monkeypatch.setattr(core.analyzer, "OpenAIChat", FakeOpenAIChat)

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

    assert report_utils.extract_mermaid_code(markdown).startswith("flowchart TD")


def test_generate_pdf_returns_bytes():
    pdf = report_utils.generate_pdf("Test Project", "# Heading\n\nA short BA report.")

    assert isinstance(pdf, bytes)
    assert len(pdf) > 500


def test_history_path_does_not_include_email(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default

    path = history_store._history_path("person@example.com", safe_secret)

    assert str(tmp_path) in path
    assert "person@example.com" not in path
    assert path.endswith(".json")


def test_save_history_returns_capped_history(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default
    current_history = [{"project": f"Old {index}", "result": "old"} for index in range(25)]

    history = history_store.save_history("Loan Portal", "Standard", "# Report", current_history, safe_secret, email="person@example.com")

    assert len(history) == 20
    assert history[0]["project"] == "Loan Portal"
    assert history[-1]["project"] == "Old 18"


def test_history_path_local_dev_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")

    path = history_store._history_path("person@example.com", lambda name, default="": default)

    assert str(tmp_path) in path
    assert path.endswith(".json")


def test_report_utils_sanitize_pdf_text_and_markdown_lines():
    long_word = "x" * 130
    sanitized = report_utils.sanitize_pdf_text(f"Hello 😀 {long_word}")
    lines = report_utils.markdown_to_pdf_lines(
        "# Heading\n"
        "\n"
        "|---|---|\n"
        "**Bold** and `code`\n"
        "```mermaid\n"
        "flowchart TD\n"
        "A-->B\n"
        "```\n"
    )

    assert "?" in sanitized
    assert "x" * 120 in sanitized
    assert lines == [
        "Heading",
        "",
        "Bold and code",
        "[Mermaid diagram code starts below. Open the Markdown version to view interactive diagrams.]",
        "flowchart TD",
        "A-->B",
    ]


def test_report_utils_mermaid_fallback_and_sanitization():
    extracted = report_utils.extract_mermaid_code("Notes\n\nflowchart TD\nA[Start (MVP)] -->|go/live| B[Finish]")
    fallback = report_utils.extract_mermaid_code("No diagram here")

    assert '["Start (MVP)"]' in extracted
    assert '-->|"go/live"|' in extracted
    assert fallback.startswith("flowchart TD")
    assert report_utils.is_valid_mermaid("bad") is False


def test_history_path_requires_secret_outside_local_dev(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    monkeypatch.delenv("BA_ASSISTANT_LOCAL_DEV", raising=False)

    with pytest.raises(RuntimeError):
        history_store._history_path("person@example.com", lambda name, default="": default)


def test_load_history_logs_invalid_json(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default
    path = history_store._history_path("person@example.com", safe_secret)
    path_obj = tmp_path / path.split("/")[-1]
    path_obj.write_text("{not json", encoding="utf-8")
    calls = []

    loaded = history_store.load_history("person@example.com", safe_secret, lambda *args: calls.append(args))

    assert loaded == []
    assert calls[0][0] == "history_load_failed"


def test_load_history_filters_non_dict_items(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default
    path = history_store._history_path("person@example.com", safe_secret)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([{"project": "A"}, "bad", {"project": "B"}], handle)

    assert history_store.load_history("person@example.com", safe_secret) == [{"project": "A"}, {"project": "B"}]


def test_history_store_empty_and_missing_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default

    with pytest.raises(ValueError):
        history_store._history_path("", safe_secret)
    assert history_store.load_history("", safe_secret) == []
    assert history_store.load_history("missing@example.com", safe_secret) == []


def test_history_store_non_list_file_and_memory_only_save(monkeypatch, tmp_path):
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default
    path = history_store._history_path("person@example.com", safe_secret)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"project": "not a list"}, handle)

    loaded = history_store.load_history("person@example.com", safe_secret)
    history = history_store.save_history("Memory Only", "Standard", "Result", [], safe_secret)

    assert loaded == []
    assert history[0]["project"] == "Memory Only"
