from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app():
    app_test = AppTest.from_file(str(APP_PATH))
    app_test.run(timeout=15)
    return app_test


def test_unauthenticated_report_cta_is_disabled():
    at = run_app()

    buttons = [button for button in at.button if button.label == "Generate BA Report"]
    assert buttons
    assert buttons[0].disabled is True
    assert any("Sign in" in info.value for info in at.info)


def test_sign_in_widgets_render():
    at = run_app()

    text_inputs = [text_input.label for text_input in at.text_input]
    buttons = [button.label for button in at.button]

    assert "Email address" in text_inputs
    assert "Send code" in buttons


def test_requirements_and_template_controls_render():
    at = run_app()

    assert any(selectbox.label == "📋 Choose Template" for selectbox in at.selectbox)
    assert any(text_area.label == "Paste requirements" for text_area in at.text_area)


def test_authenticated_state_reveals_enabled_core_workflow(monkeypatch):
    monkeypatch.setenv("BA_ASSISTANT_AUTH_SECRET", "test-history-secret")
    at = AppTest.from_file(str(APP_PATH))
    at.session_state["auth_verified"] = True
    at.session_state["auth_email"] = "verified@example.com"
    at.session_state["email"] = "verified@example.com"
    at.run(timeout=15)

    buttons = [button for button in at.button if button.label == "Generate BA Report"]
    assert buttons
    assert buttons[0].disabled is False
    assert any("verified@example.com" in md.value for md in at.markdown)
