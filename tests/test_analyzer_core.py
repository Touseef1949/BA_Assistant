from types import SimpleNamespace

import pytest

import core.analyzer as analyzer_mod


class FakeOpenAIChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["id"]


class FakeGemini:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs["id"]


class FakeAgent:
    def __init__(self, name=None, role=None, model=None, instructions=None, markdown=True, retries=None, delay_between_retries=None, exponential_backoff=None):
        self.name = name
        self.role = role
        self.model = model
        self.instructions = instructions or []
        self.markdown = markdown
        self.retries = retries
        self.delay_between_retries = delay_between_retries
        self.exponential_backoff = exponential_backoff

    def run(self, prompt, stream=False, images=None):
        return {
            "content": f"{self.name} handled: {prompt[:40]}",
            "stream": stream,
            "images": images,
        }


class FakeTeam:
    def __init__(self, name=None, model=None, members=None, instructions=None, markdown=True, retries=None, show_members_responses=None, show_member_responses=None):
        self.name = name
        self.model = model
        self.members = members or []
        self.instructions = instructions or []
        self.markdown = markdown
        self.retries = retries
        self.show_members_responses = show_members_responses
        self.show_member_responses = show_member_responses

    def run(self, prompt, stream=False, show_member_responses=None):
        return {
            "content": f"team handled: {prompt[:40]}",
            "stream": stream,
            "show_member_responses": show_member_responses,
        }


class FakeAgnoImage:
    def __init__(self, content, mime_type):
        self.content = content
        self.mime_type = mime_type


def test_requirement_analyzer_builds_agents_team_and_vision_agent(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "OpenAIChat", FakeOpenAIChat)
    monkeypatch.setattr(analyzer_mod, "Gemini", FakeGemini)
    monkeypatch.setattr(analyzer_mod, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer_mod, "Team", FakeTeam)
    monkeypatch.setattr(analyzer_mod, "safe_secret", lambda name, default="": "test-key")

    analyzer = analyzer_mod.RequirementAnalyzer(show_member_responses=True, enable_vision=True)

    assert analyzer.ba_agent.name == "BA Requirements Analyst"
    assert analyzer.ba_agent.retries == 1
    assert analyzer.comprehensive_agent.model.id == analyzer_mod.TEXT_ANALYSIS_MODEL_ID
    assert analyzer.team.retries == 0
    assert analyzer.team.show_members_responses is True
    assert len(analyzer.team.members) == 5
    assert analyzer.vision_agent.model.id == "gemini-3.5-flash"


def test_make_vision_model_falls_back_to_openai_and_errors_without_backends(monkeypatch):
    monkeypatch.setattr(analyzer_mod, "safe_secret", lambda name, default="": "google-key")
    monkeypatch.setattr(analyzer_mod, "Gemini", None)
    monkeypatch.setattr(analyzer_mod, "OpenAIChat", FakeOpenAIChat)

    fallback_model = analyzer_mod.make_vision_model()
    assert fallback_model.id == "gemini-3.5-flash"
    assert fallback_model.kwargs["base_url"] == analyzer_mod.GOOGLE_OPENAI_BASE_URL

    monkeypatch.setattr(analyzer_mod, "OpenAIChat", None)
    with pytest.raises(RuntimeError):
        analyzer_mod.make_vision_model()


def test_response_content_and_supports_parameter_helpers():
    def fn_with_x(x):
        return x

    assert analyzer_mod.supports_parameter(fn_with_x, "x") is True
    assert analyzer_mod.supports_parameter(fn_with_x, "y") is False
    assert analyzer_mod.response_content(None) == ""
    assert analyzer_mod.response_content("hello") == "hello"
    assert analyzer_mod.response_content(SimpleNamespace(text="from-object")) == "from-object"
    assert analyzer_mod.response_content({"message": "from-dict"}) == "from-dict"


def test_compose_prompt_and_run_specialized_paths():
    calls = []

    class FakeRunner:
        def __init__(self, name):
            self.name = name

        def run(self, prompt, stream=False):
            calls.append((self.name, prompt, stream))
            return {"content": f"{self.name}-ok"}

    analyzer = object.__new__(analyzer_mod.RequirementAnalyzer)
    analyzer.product_agent = FakeRunner("product")
    analyzer.architect_agent = FakeRunner("architect")
    analyzer.ba_agent = FakeRunner("ba")

    prompt = analyzer_mod.RequirementAnalyzer.compose_prompt(
        analyzer,
        "Need payment gateway",
        "Payments",
        "Interactive (Q&A)",
        "Q: volume?\nA: 10k/day",
    )
    assert "Clarifying Q&A transcript" in prompt
    assert "authoritative enrichment" in prompt

    result_known = analyzer_mod.RequirementAnalyzer.run_specialized(
        analyzer,
        "Need payment gateway",
        "Payments",
        "Technical Architecture",
        stream=True,
    )
    result_default = analyzer_mod.RequirementAnalyzer.run_specialized(
        analyzer,
        "Need payment gateway",
        "Payments",
        "Unknown Mode",
        stream=False,
    )

    assert result_known["content"] == "architect-ok"
    assert result_default["content"] == "ba-ok"
    assert calls[0][0] == "architect"
    assert "deployment considerations" in calls[0][1]
    assert calls[1][0] == "ba"


def test_generate_questions_extract_image_and_generate_mermaid(monkeypatch):
    class FakeRunner:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def run(self, prompt, stream=False, images=None):
            self.calls.append({"prompt": prompt, "stream": stream, "images": images})
            return self.response

    monkeypatch.setattr(analyzer_mod, "AgnoImage", FakeAgnoImage)

    analyzer = object.__new__(analyzer_mod.RequirementAnalyzer)
    analyzer.ba_agent = FakeRunner({"content": "1. What is the launch market?"})
    analyzer.diagram_agent = FakeRunner({"content": "```mermaid\nflowchart TD\nA-->B\n```"})
    analyzer.vision_agent = FakeRunner({"content": "Structured requirements from image"})

    questions = analyzer_mod.RequirementAnalyzer.generate_questions(analyzer, "raw reqs")
    extracted = analyzer_mod.RequirementAnalyzer.extract_requirements_from_image(analyzer, b"img", "image/png")
    mermaid = analyzer_mod.RequirementAnalyzer.generate_mermaid(analyzer, "raw reqs")

    assert questions == "1. What is the launch market?"
    assert extracted == "Structured requirements from image"
    assert mermaid.startswith("flowchart TD")
    assert analyzer.vision_agent.calls[0]["images"][0].mime_type == "image/png"


def test_extract_requirements_from_image_requires_initialized_vision_agent():
    analyzer = object.__new__(analyzer_mod.RequirementAnalyzer)
    analyzer.vision_agent = None

    with pytest.raises(RuntimeError, match="Vision agent was not initialized"):
        analyzer_mod.RequirementAnalyzer.extract_requirements_from_image(analyzer, b"img", "image/png")
