import json

from services import error_logging


def test_jsonl_error_logger_writes_structured_event(monkeypatch, tmp_path):
    log_path = tmp_path / "errors.jsonl"
    monkeypatch.setenv("BA_ASSISTANT_ERROR_LOG", str(log_path))

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        error_logging.log_error("unit_test_failure", exc, {"step": "logging"})

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["level"] == "error"
    assert payload["event"] == "unit_test_failure"
    assert payload["error_type"] == "RuntimeError"
    assert payload["context"] == {"step": "logging"}
