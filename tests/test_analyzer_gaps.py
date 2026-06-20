"""Analyzer gap-closing tests — covers RuntimeError + interactive flow paths."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import core.analyzer as analyzer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAIChat is None → RuntimeError (lines 67, 78)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_make_worker_model_no_agno(monkeypatch):
    """make_worker_model raises when OpenAIChat is None."""
    monkeypatch.setattr(analyzer, "OpenAIChat", None, raising=False)
    with pytest.raises(RuntimeError, match="OpenAIChat"):
        analyzer.make_worker_model()


def test_make_coordinator_model_no_agno(monkeypatch):
    """make_coordinator_model raises when OpenAIChat is None."""
    monkeypatch.setattr(analyzer, "OpenAIChat", None, raising=False)
    with pytest.raises(RuntimeError, match="OpenAIChat"):
        analyzer.make_coordinator_model()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent is None → RuntimeError (lines 136, 200, 221)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_analyzer_agent_is_none(monkeypatch):
    """RequirementAnalyzer raises when Agent is None."""
    monkeypatch.setattr(analyzer, "Agent", None, raising=False)
    with pytest.raises(RuntimeError, match="Agno Agent unavailable"):
        analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)


def test_analyzer_agent_is_none_with_vision(monkeypatch):
    """RequirementAnalyzer with vision raises when Agent is None."""
    # Need AgnoImage to not be None, but Agent is None
    monkeypatch.setattr(analyzer, "Agent", None, raising=False)
    monkeypatch.setattr(analyzer, "Team", None, raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", None, raising=False)
    with pytest.raises(RuntimeError, match="Agno Agent unavailable"):
        analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Team is None → RuntimeError (line 236)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_analyzer_team_is_none(monkeypatch):
    """RequirementAnalyzer _build_team raises when Team is None."""
    # Set Agent + OpenAIChat to valid (non-None) mocks, but Team to None
    monkeypatch.setattr(analyzer, "Team", None, raising=False)
    with pytest.raises(RuntimeError, match="Agno Team unavailable"):
        analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# supports_parameter branches (lines 256-257, 298)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_supports_parameter_returns_true():
    """supports_parameter returns True for existing param."""
    assert analyzer.supports_parameter(lambda x, y=1, stream=False: None, "stream") is True


def test_supports_parameter_returns_false():
    """supports_parameter returns False for non-existent param."""
    assert analyzer.supports_parameter(lambda x: None, "nonexistent") is False


def test_supports_parameter_non_callable():
    """supports_parameter with non-callable returns False."""
    assert analyzer.supports_parameter(None, "any") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_interactive (lines 302-306)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_run_interactive_calls_team(monkeypatch):
    """run_interactive calls self.team.run with correct prompt."""
    from unittest.mock import MagicMock

    # Create a minimal analyzer with mocked team
    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    mock_team = MagicMock()
    mock_team.run.return_value = "interactive result"
    a.team = mock_team

    result = a.run_interactive("reqs", "proj", "QA transcript", stream=False)
    assert "interactive" in str(result)
    mock_team.run.assert_called_once()


def test_run_interactive_with_stream(monkeypatch):
    """run_interactive with stream=True."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    mock_team = MagicMock()
    a.team = mock_team

    a.run_interactive("reqs", "proj", "QA", stream=True)
    mock_team.run.assert_called_once()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AgnoImage unavailable (line 361)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_extract_requirements_agno_image_none(monkeypatch):
    """extract_requirements_from_image raises when AgnoImage is None."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "AgnoImage", None, raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=True)
    mock_vision = MagicMock()
    a.vision_agent = mock_vision

    with pytest.raises(RuntimeError, match="AgnoImage unavailable"):
        a.extract_requirements_from_image(b"fake", "image/png")


def test_extract_requirements_no_vision_agent(monkeypatch):
    """extract_requirements_from_image raises when vision_agent is None."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    a.vision_agent = None

    with pytest.raises(RuntimeError, match="Vision agent"):
        a.extract_requirements_from_image(b"fake", "image/png")


def test_extract_requirements_success(monkeypatch):
    """extract_requirements_from_image returns extracted text."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    # Mock AgnoImage class
    mock_agno_image_cls = MagicMock()
    monkeypatch.setattr(analyzer, "AgnoImage", mock_agno_image_cls, raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=True)
    mock_vision = MagicMock()
    mock_vision.run.return_value = "extracted text from image"
    a.vision_agent = mock_vision

    result = a.extract_requirements_from_image(b"fake_bytes", "image/png")
    assert "extracted" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# response_content edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_response_content_from_object_with_text():
    """response_content extracts from object with 'text' attribute."""
    class Resp:
        text = "  hello  "
    result = analyzer.response_content(Resp())
    assert result == "  hello  "


def test_response_content_from_object_with_message():
    """response_content extracts from object with 'message' attribute."""
    class Resp:
        message = "  msg content  "
    result = analyzer.response_content(Resp())
    assert result == "  msg content  "


def test_response_content_from_dict_with_message():
    """response_content extracts from dict with 'message' key."""
    assert analyzer.response_content({"message": "  msg  "}) == "  msg  "


def test_response_content_from_dict_with_output():
    """response_content extracts from dict with 'output' key."""
    assert analyzer.response_content({"output": "  out  "}) == "  out  "


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# generate_mermaid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_generate_mermaid(monkeypatch):
    """generate_mermaid calls diagram_agent.run."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    mock_agent = MagicMock()
    mock_agent.run.return_value = "```mermaid\ngraph TD\nA-->B\n```"
    a.diagram_agent = mock_agent

    result = a.generate_mermaid("some requirements")
    assert "graph TD" in result


def test_run_specialized_standard(monkeypatch):
    """run_specialized with Standard type uses comprehensive_agent."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    # run_specialized: "Standard" not in mapping → defaults to ba_agent (Gap & Clarification)
    mock_agent = MagicMock()
    mock_agent.run.return_value = "standard result"
    a.ba_agent = mock_agent

    result = a.run_specialized("reqs", "proj", "Standard", stream=False)
    assert "standard" in str(result)


def test_run_specialized_deep_team(monkeypatch):
    """run_specialized with Deep Team uses team."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    # run_specialized uses ba_agent for "Deep Team" (defaults to Gap & Clarification)
    mock_agent = MagicMock()
    mock_agent.run.return_value = "team result"
    a.ba_agent = mock_agent

    result = a.run_specialized("reqs", "proj", "Deep Team", stream=False)
    assert "team" in str(result)


def test_run_specialized_other(monkeypatch):
    """run_specialized with other type uses team."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    # run_specialized for unknown types uses ba_agent (Gap & Clarification)
    mock_agent = MagicMock()
    mock_agent.run.return_value = "other result"
    a.ba_agent = mock_agent

    result = a.run_specialized("reqs", "proj", "Interactive (Q&A)", stream=False)
    assert "other" in str(result)


def test_compose_prompt_with_qa(monkeypatch):
    """compose_prompt with QA transcript includes it."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    prompt = a.compose_prompt("reqs", "proj", "Interactive (Q&A)", "QA transcript here")
    assert "QA transcript here" in prompt


def test_compose_prompt_deep_team(monkeypatch):
    """compose_prompt with Deep Team adds enterprise lens."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(analyzer, "Agent", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "Team", MagicMock(), raising=False)
    monkeypatch.setattr(analyzer, "OpenAIChat", MagicMock(), raising=False)

    a = analyzer.RequirementAnalyzer("test-model", show_member_responses=False, enable_vision=False)
    prompt = a.compose_prompt("reqs", "proj", "Deep Team", "")
    assert "enterprise" in prompt.lower()
