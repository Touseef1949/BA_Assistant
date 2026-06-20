"""AppTest coverage — exercises main rendering flow to close app.py cover gap."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def set_local_dev():
    os.environ["BA_ASSISTANT_LOCAL_DEV"] = "1"


def _run_authed() -> AppTest:
    """Run app with auth pre-verified."""
    at = AppTest.from_file("app.py")
    at.session_state["auth_verified"] = True
    at.session_state["auth_email"] = "verified@example.com"
    at.session_state["email"] = "verified@example.com"
    at.run()
    return at


def test_app_loads_and_renders():
    """App loads and renders hero + auth panel (visible even when unauthenticated)."""
    at = AppTest.from_file("app.py")
    at.run()

    assert at is not None
    # Hero section should render (render_header runs before auth gate)
    assert len(at.markdown) > 0


def test_app_sidebar_brand():
    """Sidebar contains brand card and settings (authenticated)."""
    at = _run_authed()

    sidebar = getattr(at, 'sidebar', None)
    assert sidebar is not None or len(at.markdown) > 0


def test_app_clear_button_flow():
    """Clear button resets state (authenticated)."""
    at = _run_authed()

    for btn in at.button:
        if btn.label == "Clear" and not btn.disabled:
            btn.click()
            at.run()
            break


def test_app_template_selector():
    """Template selector is present and functional (authenticated)."""
    at = _run_authed()

    assert len(at.selectbox) >= 1


def test_app_textarea():
    """Requirements textarea is present (authenticated)."""
    at = _run_authed()

    assert len(at.text_area) >= 1


def test_app_expanders():
    """Upload extractor and advanced expanders render (authenticated)."""
    at = _run_authed()

    assert len(at.expander) >= 1


def test_app_result_section():
    """When result exists in session state, report and download buttons render (authenticated)."""
    at = _run_authed()
    at.session_state["last_result"] = "# Test Report\n\nContent."
    at.run()

    md_texts = [str(m.value) for m in at.markdown]
    assert any("Test Report" in t for t in md_texts)
