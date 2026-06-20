import json

import app
from core.analyzer import RequirementAnalyzer, response_content
import payment
from services import history_store


def test_verified_gate_consumes_usage_and_history_persists(monkeypatch, tmp_path):
    email = "verified@example.com"
    store = {
        email: payment._normalize_user(
            {"email": email, "analyses_used": 0, "analyses_limit": 2, "email_verified": True},
            email,
        )
    }
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_users", lambda: store)
    monkeypatch.setenv("BA_ASSISTANT_HISTORY_DIR", str(tmp_path))
    safe_secret = lambda name, default="": "history-secret" if name == "BA_ASSISTANT_AUTH_SECRET" else default
    monkeypatch.setitem(app.st.session_state, "history", [])

    allowed, message, user = payment.gate_analysis(email, consume_usage=True)
    app.st.session_state["history"] = history_store.save_history(
        "Loan Portal",
        "Standard",
        "# Report\n\nGenerated output",
        app.st.session_state.get("history", []),
        safe_secret,
        email=email,
    )
    loaded = history_store.load_history(email, safe_secret)

    assert allowed is True
    assert message == "Free usage: 1/2."
    assert user["analyses_used"] == 1
    assert loaded[0]["project"] == "Loan Portal"
    assert loaded[0]["type"] == "Standard"
    with open(history_store._history_path(email, safe_secret), "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw[0]["result"].startswith("# Report")


def test_standard_analysis_uses_comprehensive_agent(monkeypatch):
    calls = []

    class FakeRunner:
        def __init__(self, name):
            self.name = name

        def run(self, prompt, stream=False, **kwargs):
            calls.append((self.name, stream, prompt))
            return {"content": f"{self.name} response"}

    analyzer = object.__new__(RequirementAnalyzer)
    analyzer.comprehensive_agent = FakeRunner("comprehensive")
    analyzer.team = FakeRunner("team")
    analyzer.show_member_responses = False

    response = RequirementAnalyzer.run_analysis(analyzer, "Need payments", "Payments", "Standard", stream=False)

    assert response_content(response) == "comprehensive response"
    assert calls[0][0] == "comprehensive"
    assert "Need payments" in calls[0][2]


def test_deep_team_analysis_uses_team_runner():
    calls = []

    class FakeTeam:
        def run(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "team response"

    analyzer = object.__new__(RequirementAnalyzer)
    analyzer.comprehensive_agent = object()
    analyzer.team = FakeTeam()
    analyzer.show_member_responses = False

    response = RequirementAnalyzer.run_analysis(analyzer, "Need KYC", "KYC", "Deep Team", stream=True)

    assert response == "team response"
    assert calls[0][1]["stream"] is True
    assert "enterprise implementation lens" in calls[0][0]
