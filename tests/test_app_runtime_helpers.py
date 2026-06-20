import os
from types import SimpleNamespace

import pytest

import app


class DummyPlaceholder:
    def __init__(self):
        self.markdowns = []
        self.warnings = []

    def markdown(self, value):
        self.markdowns.append(value)

    def warning(self, value):
        self.warnings.append(value)


def test_bootstrap_environment_sets_missing_values_only(monkeypatch):
    values = {
        "DEEPSEEK_API_KEY": "deepseek-key",
        "GOOGLE_API_KEY": "google-key",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "supabase-key",
        "RAZORPAY_KEY_ID": "rzp_id",
        "RAZORPAY_KEY_SECRET": "rzp_secret",
        "RAZORPAY_WEBHOOK_SECRET": "rzp_webhook",
    }
    monkeypatch.setattr(app, "safe_secret", lambda name, default="": values.get(name, default))
    for key in (
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "SUPABASE_URL",
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET",
        "RAZORPAY_WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SUPABASE_KEY", "already-set")

    app.bootstrap_environment()

    assert os.environ["DEEPSEEK_API_KEY"] == "deepseek-key"
    assert os.environ["GOOGLE_API_KEY"] == "google-key"
    assert os.environ["SUPABASE_KEY"] == "already-set"
    assert os.environ["RAZORPAY_WEBHOOK_SECRET"] == "rzp_webhook"


def test_require_runtime_dependencies_missing_and_warning(monkeypatch):
    events = {"warning": [], "error": [], "info": []}
    fake_st = SimpleNamespace(
        warning=lambda msg: events["warning"].append(msg),
        error=lambda msg: events["error"].append(msg),
        info=lambda msg: events["info"].append(msg),
    )
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "Agent", None)
    monkeypatch.setattr(app, "Team", None)
    monkeypatch.setattr(app, "OpenAIChat", None)
    monkeypatch.setattr(app, "AgnoImage", None)
    monkeypatch.setattr(app, "Gemini", None)
    monkeypatch.setattr(app, "AGNO_IMPORT_ERROR", "agent-missing")
    monkeypatch.setattr(app, "TEAM_IMPORT_ERROR", "team-missing")
    monkeypatch.setattr(app, "OPENAI_CHAT_IMPORT_ERROR", "chat-missing")
    monkeypatch.setattr(app, "AGNO_IMAGE_IMPORT_ERROR", "image-missing")

    assert app.require_runtime_dependencies(vision=True) is False
    assert events["warning"]
    assert "agent-missing" in events["error"][0]
    assert "pip install -r requirements.txt" in events["info"][0]


def test_require_runtime_dependencies_success(monkeypatch):
    fake_st = SimpleNamespace(warning=lambda msg: None, error=lambda msg: None, info=lambda msg: None)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "Agent", object())
    monkeypatch.setattr(app, "Team", object())
    monkeypatch.setattr(app, "OpenAIChat", object())
    monkeypatch.setattr(app, "AgnoImage", object())

    assert app.require_runtime_dependencies(vision=True) is True


def test_require_api_keys_reports_missing(monkeypatch):
    events = {"error": [], "info": []}
    fake_st = SimpleNamespace(error=lambda msg: events["error"].append(msg), info=lambda msg: events["info"].append(msg))
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "safe_secret", lambda name, default="": "")

    assert app.require_api_keys(vision=True) is False
    assert "DEEPSEEK_API_KEY" in events["error"][0]
    assert "GOOGLE_API_KEY" in events["error"][0]

    monkeypatch.setattr(app, "safe_secret", lambda name, default="": "set-key")
    assert app.require_api_keys(vision=True) is True


def test_event_content_stream_to_markdown_and_fallbacks(monkeypatch):
    placeholder = DummyPlaceholder()
    logs = []
    monkeypatch.setattr(app, "log_error", lambda *args, **kwargs: logs.append((args, kwargs)))

    def stream_ok(stream):
        if stream:
            return [SimpleNamespace(content="Hello "), {"delta": "World"}]
        return {"content": "unused"}

    assert app.event_content(SimpleNamespace(text="alpha")) == "alpha"
    assert app.event_content({"message": "beta"}) == "beta"
    assert app.stream_to_markdown(stream_ok, placeholder) == "Hello World"
    assert placeholder.markdowns[-1] == "Hello World"

    placeholder2 = DummyPlaceholder()

    def type_error_then_nonstream(stream):
        if stream:
            raise TypeError("no stream")
        return {"content": "Recovered"}

    assert app.stream_to_markdown(type_error_then_nonstream, placeholder2) == "Recovered"
    assert placeholder2.markdowns[-1] == "Recovered"

    placeholder3 = DummyPlaceholder()

    def exception_then_nonstream(stream):
        if stream:
            raise RuntimeError("boom")
        return {"content": "Fallback worked"}

    assert app.stream_to_markdown(exception_then_nonstream, placeholder3) == "Fallback worked"
    assert placeholder3.warnings
    assert logs


def test_parse_questions_truncate_reset_interactive_and_run_paid_gate(monkeypatch):
    monkeypatch.setitem(app.st.session_state, "interactive_stage", "questions")
    monkeypatch.setitem(app.st.session_state, "interactive_questions", ["Q1"])
    monkeypatch.setitem(app.st.session_state, "interactive_answers", {"Q1": "A1"})
    app.reset_interactive()
    assert app.st.session_state["interactive_stage"] == "input"
    assert app.st.session_state["interactive_questions"] == []
    assert app.st.session_state["interactive_answers"] == {}

    parsed = app.parse_questions("1. First question?\n- Second question?\nnoise")
    assert parsed == ["First question?", "Second question?"]
    assert app.truncate("a  b   c", length=20) == "a b c"

    messages = {"error": [], "caption": []}
    fake_st = SimpleNamespace(error=lambda msg: messages["error"].append(msg), caption=lambda msg: messages["caption"].append(msg))
    monkeypatch.setattr(app, "st", fake_st)

    assert app.run_paid_gate("", consume_usage=True) is False
    assert messages["error"]

    monkeypatch.setattr(app, "gate_analysis", lambda email, consume_usage=True: (True, "Free usage: 1/2.", {}))
    assert app.run_paid_gate("person@example.com", consume_usage=False) is True
    assert messages["caption"][-1] == "Free usage: 1/2."

    monkeypatch.setattr(app, "gate_analysis", lambda email, consume_usage=True: (False, "Blocked", {}))
    assert app.run_paid_gate("person@example.com", consume_usage=True) is False


def test_requirements_flow_dependencies_and_sidebar_config(monkeypatch):
    monkeypatch.setitem(app.st.session_state, "analysis_type", "Standard")
    monkeypatch.setitem(app.st.session_state, "history", [{"time": "now", "project": "Loan Portal"}])

    deps = app.requirements_flow_dependencies()
    assert deps.analyzer_factory is app.RequirementAnalyzer
    assert deps.stream_to_markdown_fn is app.stream_to_markdown

    class Context:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        sidebar = Context()
        session_state = {"analysis_type": "Standard", "history": [{"time": "now", "project": "Loan Portal"}], "requirements_area": "x", "selected_template": "loan_origination", "_last_template": "loan_origination"}

        def __init__(self):
            self.captions = []
            self.markdowns = []

        def markdown(self, msg):
            self.markdowns.append(msg)
        def success(self, msg):
            self.markdowns.append(msg)
        def info(self, msg):
            self.markdowns.append(msg)
        def caption(self, msg):
            self.captions.append(msg)
        def text_input(self, label, key=None):
            return "Project X"
        def radio(self, label, options, index=0, key=None):
            return options[index]
        def expander(self, label, expanded=False):
            return Context()
        def toggle(self, label, value=False, help=None):
            return False
        def selectbox(self, label, options, index=0):
            return options[index]
        def button(self, label, use_container_width=True):
            return False
        def rerun(self):
            raise AssertionError("rerun should not be called")

    fake_st = FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_pricing", lambda email, user=None: None)
    monkeypatch.setattr(app, "render_footer", lambda: None)
    monkeypatch.setattr(app, "get_user", lambda email: {"plan": "free", "analyses_used": 1, "analyses_limit": 2})
    monkeypatch.setattr(app, "create_user", lambda email: {"plan": "free", "analyses_used": 1, "analyses_limit": 2})

    config = app.sidebar_config("person@example.com", {"plan": "free", "analyses_used": 1, "analyses_limit": 2})
    assert config.project_name == "Project X"
    assert config.analysis_type == "Standard"
    assert any("Loan Portal" in c for c in fake_st.captions)
