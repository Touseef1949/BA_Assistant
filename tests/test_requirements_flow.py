from types import SimpleNamespace

import pytest

from ui import requirements_flow


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _deps(**overrides):
    reset_calls = []
    defaults = {
        "financial_templates": {
            "none": ("Start from scratch", ""),
            "loan_origination": ("Loan Origination Portal", "loan text"),
            "payment_gateway": ("Payment Gateway Integration", "payment text"),
        },
        "report_structure": "Report sections",
        "extract_pdf_text_fn": lambda uploaded_file: "pdf text",
        "require_runtime_dependencies_fn": lambda vision=False: True,
        "require_api_keys_fn": lambda vision=False: True,
        "analyzer_factory": lambda *args, **kwargs: None,
        "reset_interactive_fn": lambda: reset_calls.append(True),
        "run_paid_gate_fn": lambda email, consume_usage=True: True,
        "parse_questions_fn": lambda raw: [raw],
        "stream_to_markdown_fn": lambda run_callable, placeholder: "result",
        "extract_mermaid_code_fn": lambda result: "flowchart TD",
        "save_history_fn": lambda *args, **kwargs: [],
        "safe_secret_fn": lambda name, default="": default,
    }
    defaults.update(overrides)
    return requirements_flow.RequirementsFlowDependencies(**defaults), reset_calls


def test_template_selector_syncs_selected_template_and_requirements(monkeypatch):
    deps, reset_calls = _deps()

    class FakeStreamlit:
        session_state = {
            "selected_template": "loan_origination",
            "_last_template": "loan_origination",
            "requirements_area": "loan text",
        }

        @staticmethod
        def selectbox(label, labels, index=0):
            assert label == "📋 Choose Template"
            assert labels[index] == "Loan Origination Portal"
            return "Payment Gateway Integration"

        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)

    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_template_selector(deps)

    assert FakeStreamlit.session_state["selected_template"] == "payment_gateway"
    assert FakeStreamlit.session_state["requirements_area"] == "payment text"
    assert reset_calls == [True]


def test_render_upload_area_pdf_extracts_and_reruns(monkeypatch):
    deps, reset_calls = _deps(extract_pdf_text_fn=lambda uploaded_file: "Extracted PDF")

    class Uploaded:
        name = "spec.pdf"
        size = 100
        type = "application/pdf"
        def seek(self, *_args):
            return None

    class FakeStreamlit:
        session_state = {"last_uploaded_signature": "old"}

        @staticmethod
        def file_uploader(*_args, **_kwargs):
            return Uploaded()
        @staticmethod
        def spinner(_msg):
            return _Context()
        @staticmethod
        def success(_msg):
            return None
        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)

    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_upload_area(SimpleNamespace(model_id="m", show_member_responses=False), deps)

    assert FakeStreamlit.session_state["requirements_area"] == "Extracted PDF"
    assert reset_calls == [True]


def test_render_upload_area_duplicate_pdf_shows_info(monkeypatch):
    deps, _reset_calls = _deps()
    seen = []

    class Uploaded:
        name = "spec.pdf"
        size = 100
        type = "application/pdf"
        def seek(self, *_args):
            return None

    class FakeStreamlit:
        session_state = {"last_uploaded_signature": "spec.pdf:100:application/pdf"}

        @staticmethod
        def file_uploader(*_args, **_kwargs):
            return Uploaded()
        @staticmethod
        def info(msg):
            seen.append(msg)

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    requirements_flow.render_upload_area(SimpleNamespace(model_id="m", show_member_responses=False), deps)
    assert seen == ["PDF already extracted into the requirements area."]


def test_render_upload_area_image_extract_branch(monkeypatch):
    extracted_calls = []

    class Analyzer:
        def extract_requirements_from_image(self, image_bytes, mime_type):
            extracted_calls.append((image_bytes, mime_type))
            return "Image extraction"

    deps, reset_calls = _deps(analyzer_factory=lambda *args, **kwargs: Analyzer())

    class Uploaded:
        name = "screen.png"
        size = 10
        type = "image/png"
        def __init__(self):
            self.seek_calls = 0
        def seek(self, *_args):
            self.seek_calls += 1
        def getvalue(self):
            return b"png-bytes"

    uploaded = Uploaded()

    class FakeStreamlit:
        session_state = {"last_uploaded_signature": "old"}

        @staticmethod
        def file_uploader(*_args, **_kwargs):
            return uploaded
        @staticmethod
        def image(*_args, **_kwargs):
            return None
        @staticmethod
        def button(label, type="secondary"):
            assert label == "🔍 Extract Requirements from Image"
            return True
        @staticmethod
        def spinner(_msg):
            return _Context()
        @staticmethod
        def success(_msg):
            return None
        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    monkeypatch.setattr(requirements_flow.PILImage, "open", lambda f: object())

    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_upload_area(SimpleNamespace(model_id="m", show_member_responses=False), deps)

    assert FakeStreamlit.session_state["requirements_area"] == "Image extraction"
    assert extracted_calls == [(b"png-bytes", "image/png")]
    assert reset_calls == [True]


def test_prompt_preview_uses_fallback_when_analyzer_unavailable(monkeypatch):
    deps, _reset_calls = _deps(require_runtime_dependencies_fn=lambda vision=False: False)
    captured = {}

    class FakeStreamlit:
        @staticmethod
        def expander(label, expanded=False):
            assert label == "Prompt preview"
            assert expanded is False
            return _Context()

        @staticmethod
        def code(body, language=None):
            captured["body"] = body
            captured["language"] = language

    config = SimpleNamespace(
        show_prompt_preview=True,
        model_id="model",
        show_member_responses=False,
        project_name="Project",
        analysis_type="Standard",
    )
    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)

    requirements_flow.render_prompt_preview(config, "raw requirements", deps)

    assert captured["body"] == "Report sections\n\nRequirements:\nraw requirements"
    assert captured["language"] == "markdown"


def test_prompt_preview_uses_analyzer_when_available(monkeypatch):
    class Analyzer:
        def compose_prompt(self, requirements_text, project_name, analysis_type):
            return f"PROMPT::{project_name}::{analysis_type}::{requirements_text}"

    deps, _reset_calls = _deps(analyzer_factory=lambda *args, **kwargs: Analyzer())
    captured = {}

    class FakeStreamlit:
        @staticmethod
        def expander(label, expanded=False):
            return _Context()
        @staticmethod
        def code(body, language=None):
            captured["body"] = body
            captured["language"] = language

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    config = SimpleNamespace(show_prompt_preview=True, model_id="m", show_member_responses=False, project_name="P", analysis_type="Standard")
    requirements_flow.render_prompt_preview(config, "raw", deps)
    assert captured["body"] == "PROMPT::P::Standard::raw"


def test_render_interactive_flow_input_warns_when_requirements_missing(monkeypatch):
    deps, _reset_calls = _deps()
    warnings = []

    class FakeStreamlit:
        session_state = {"interactive_stage": "input"}
        @staticmethod
        def markdown(_msg):
            return None
        @staticmethod
        def caption(_msg):
            return None
        @staticmethod
        def button(_label, **_kwargs):
            return True
        @staticmethod
        def warning(msg):
            warnings.append(msg)

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    requirements_flow.render_interactive_flow(SimpleNamespace(model_id="m", show_member_responses=False, project_name="P", analysis_type="Interactive (Q&A)", add_confetti=False), "person@example.com", "   ", deps)
    assert warnings == ["Add requirements or upload a document first."]


def test_render_interactive_flow_generates_questions_and_reruns(monkeypatch):
    class Analyzer:
        def generate_questions(self, requirements_text):
            return "1. What is the SLA?"

    deps, _reset_calls = _deps(analyzer_factory=lambda *args, **kwargs: Analyzer(), parse_questions_fn=lambda raw: ["What is the SLA?"])

    class FakeStreamlit:
        session_state = {"interactive_stage": "input", "interactive_answers": {}}
        @staticmethod
        def markdown(_msg):
            return None
        @staticmethod
        def caption(_msg):
            return None
        @staticmethod
        def button(_label, **_kwargs):
            return True
        @staticmethod
        def spinner(_msg):
            return _Context()
        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_interactive_flow(SimpleNamespace(model_id="m", show_member_responses=False, project_name="P", analysis_type="Interactive (Q&A)", add_confetti=False), "person@example.com", "Need KYC", deps)
    assert FakeStreamlit.session_state["interactive_stage"] == "questions"
    assert FakeStreamlit.session_state["interactive_questions"] == ["What is the SLA?"]


def test_render_interactive_flow_questions_without_questions_reruns(monkeypatch):
    deps, _reset_calls = _deps()

    class FakeStreamlit:
        session_state = {"interactive_stage": "questions", "interactive_questions": []}
        @staticmethod
        def markdown(_msg):
            return None
        @staticmethod
        def caption(_msg):
            return None
        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_interactive_flow(SimpleNamespace(model_id="m", show_member_responses=False, project_name="P", analysis_type="Interactive (Q&A)", add_confetti=False), "person@example.com", "Need KYC", deps)
    assert FakeStreamlit.session_state["interactive_stage"] == "input"


def test_render_interactive_flow_restart_branch(monkeypatch):
    deps, reset_calls = _deps()

    class FakeStreamlit:
        session_state = {"interactive_stage": "questions", "interactive_questions": ["Q1"], "interactive_answers": {}}
        @staticmethod
        def markdown(_msg):
            return None
        @staticmethod
        def caption(_msg):
            return None
        @staticmethod
        def info(_msg):
            return None
        @staticmethod
        def text_input(label, value="", key=None):
            return "Answer"
        @staticmethod
        def columns(_spec):
            return _Context(), _Context()
        @staticmethod
        def button(label, **_kwargs):
            return label == "↩️ Restart Q&A"
        @staticmethod
        def rerun():
            raise RuntimeError("rerun")

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    with pytest.raises(RuntimeError, match="rerun"):
        requirements_flow.render_interactive_flow(SimpleNamespace(model_id="m", show_member_responses=False, project_name="P", analysis_type="Interactive (Q&A)", add_confetti=False), "person@example.com", "Need KYC", deps)
    assert reset_calls == [True]


def test_render_interactive_flow_generate_full_report_success(monkeypatch):
    class Analyzer:
        def run_interactive(self, requirements_text, project_name, qa_transcript, stream=False):
            return {"content": f"REPORT::{project_name}::{qa_transcript}"}

    deps, _reset_calls = _deps(
        analyzer_factory=lambda *args, **kwargs: Analyzer(),
        stream_to_markdown_fn=lambda run_callable, placeholder: run_callable(False)["content"],
        save_history_fn=lambda *args, **kwargs: [{"project": args[0], "result": args[2]}],
    )
    balloons = []
    successes = []

    class FakeStreamlit:
        session_state = {
            "interactive_stage": "questions",
            "interactive_questions": ["Q1"],
            "interactive_answers": {"Q1": "Old"},
            "history": [],
        }
        @staticmethod
        def markdown(_msg):
            return None
        @staticmethod
        def caption(_msg):
            return None
        @staticmethod
        def info(_msg):
            return None
        @staticmethod
        def text_input(label, value="", key=None):
            return "Fresh answer"
        @staticmethod
        def columns(_spec):
            return _Context(), _Context()
        @staticmethod
        def button(label, **_kwargs):
            return label == "✅ Generate Full Report"
        @staticmethod
        def empty():
            return object()
        @staticmethod
        def spinner(_msg):
            return _Context()
        @staticmethod
        def balloons():
            balloons.append(True)
        @staticmethod
        def success(msg):
            successes.append(msg)

    monkeypatch.setattr(requirements_flow, "st", FakeStreamlit)
    config = SimpleNamespace(model_id="m", show_member_responses=False, project_name="Project", analysis_type="Interactive (Q&A)", add_confetti=True)
    requirements_flow.render_interactive_flow(config, "person@example.com", "Need KYC", deps)

    assert FakeStreamlit.session_state["last_result"].startswith("REPORT::Project")
    assert FakeStreamlit.session_state["last_mermaid"] == "flowchart TD"
    assert FakeStreamlit.session_state["history"][0]["project"] == "Project"
    assert FakeStreamlit.session_state["interactive_stage"] == "questions"
    assert balloons == [True]
    assert successes == ["Interactive report generated."]
