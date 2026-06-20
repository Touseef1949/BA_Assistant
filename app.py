"""BA Assistant v2 - Streamlit + Agno multi-agent requirement analysis app.

Deploy with `streamlit run app.py` after installing requirements.txt and setting
Streamlit secrets or environment variables for DeepSeek, Google, Supabase, and Razorpay.
"""

from __future__ import annotations

import html
import io
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from core.analyzer import (
    AGNO_IMAGE_IMPORT_ERROR,
    AGNO_IMPORT_ERROR,
    GEMINI_IMPORT_ERROR,
    OPENAI_CHAT_IMPORT_ERROR,
    TEAM_IMPORT_ERROR,
    Agent,
    AgnoImage,
    Gemini,
    OpenAIChat,
    RequirementAnalyzer,
    Team,
    make_coordinator_model,
    make_vision_model,
    make_worker_model,
    response_content,
    supports_parameter,
)
from core.config import (
    DEEPSEEK_BASE_URL,
    GOOGLE_OPENAI_BASE_URL,
    PROMPT_INJECTION_GUARD,
    REPORT_STRUCTURE,
    TEXT_ANALYSIS_MODEL_ID,
    safe_secret,
)
from services.error_logging import log_error
from services.history_store import load_history, save_history
from services.report_utils import (
    _safe_write_line,
    extract_mermaid_code,
    generate_pdf,
    is_valid_mermaid,
    markdown_to_pdf_lines,
    sanitize_mermaid_code,
    sanitize_pdf_text,
)
from ui.requirements_flow import (
    RequirementsFlowDependencies,
    render_interactive_flow,
    render_prompt_preview,
    render_template_selector,
    render_upload_area,
)

try:
    import pdfplumber
except Exception:  # pragma: no cover - handled in UI
    pdfplumber = None

try:
    from payment import create_user, gate_analysis, get_user, render_auth_panel, render_pricing, sign_out
except Exception as exc:  # pragma: no cover
    PAYMENT_IMPORT_ERROR = exc

    def get_user(email: str) -> Dict[str, Any]:  # type: ignore[override]
        return {"email": email, "plan": "free", "usage_count": 0, "usage_limit": 2, "status": "local"}

    def create_user(email: str) -> Dict[str, Any]:  # type: ignore[override]
        return get_user(email)

    def render_pricing(email: str, user: Optional[Dict[str, Any]] = None) -> None:  # type: ignore[override]
        st.info("Payment module is unavailable. Install dependencies and verify payment.py.")

    def render_auth_panel() -> Tuple[bool, str, Dict[str, Any]]:  # type: ignore[override]
        st.error("Authentication is unavailable. Please contact the administrator.")
        return False, "", {}

    def sign_out() -> None:  # type: ignore[override]
        return None

    def gate_analysis(email: str, consume_usage: bool = True) -> Tuple[bool, str, Dict[str, Any]]:  # type: ignore[override]
        # Security: fail CLOSED by default. Only allow the local-dev
        # bypass when the operator has explicitly set this environment
        # variable. Otherwise a missing payment module in production
        # would silently disable the paywall.
        if os.environ.get("BA_ASSISTANT_LOCAL_DEV", "").strip().lower() in ("1", "true", "yes"):
            return True, "Payment module unavailable; running in local dev mode.", get_user(email)
        return False, (
            "Payment module is unavailable. Please contact the administrator. "
            "Analysis is paused to protect paid features."
        ), get_user(email)
else:
    PAYMENT_IMPORT_ERROR = None


st.set_page_config(
    page_title="BA Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()


# -----------------------------------------------------------------------------
# Configuration and constants
# -----------------------------------------------------------------------------

APP_URL = "https://touseefshaik.com"


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    analysis_type: str
    model_id: str
    render_mermaid: bool
    mermaid_theme: str
    add_confetti: bool
    show_prompt_preview: bool
    show_member_responses: bool


ANALYSIS_TYPES = [
    "Standard",
    "Interactive (Q&A)",
    "Deep Team",
]

ANALYSIS_TYPE_INFO = {
    "Standard": "Fast complete BA report with requirements, stories, risks, architecture, and Mermaid diagram.",
    "Interactive (Q&A)": "Ask clarifying questions first, then generate the full report from your answers.",
    "Deep Team": "Advanced multi-agent review for complex or compliance-heavy initiatives.",
}

FINANCIAL_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "none": ("Start from scratch", ""),
    "loan_origination": (
        "Loan Origination Portal",
        """Build a digital loan origination system for a fintech lender.
Users should be able to apply for personal loans, upload KYC documents (Aadhaar, PAN),
undergo automated credit assessment, and receive in-principle approval.
The system should integrate with:
- CKYC/Aadhaar e-KYC for identity verification
- Credit bureaus (CIBIL, Experian) for credit scores
- Bank account aggregators for income verification
- NBFC core systems for disbursement
- RBI-compliant audit trail and reporting
Compliance: RBI digital lending guidelines, data localization, consent framework.""",
    ),
    "payment_gateway": (
        "Payment Gateway Integration",
        """Build a payment processing module that routes transactions through multiple
payment gateways (Razorpay, PayU, BillDesk) with smart routing based on
success rates and transaction costs. Support UPI, netbanking, cards, and wallets.
Include: transaction reconciliation, refund handling, settlement reports,
NPCI compliance for UPI, PCI-DSS requirements, and fraud detection rules.""",
    ),
    "kyc_aml": (
        "KYC/AML Compliance System",
        """Build a KYC/AML screening module for customer onboarding.
Requirements: document OCR (Aadhaar, PAN, passport), face match/liveness check,
PEP (Politically Exposed Persons) screening, sanctions list checks (UN, OFAC),
risk scoring engine, case management for compliance officers,
automated STR (Suspicious Transaction Report) generation for FIU-IND.
Compliance: PMLA Act, RBI KYC Master Direction, SEBI KYC requirements.""",
    ),
    "trading_platform": (
        "Retail Trading Platform",
        """Build a retail stock trading platform for Indian markets.
Features: NSE/BSE real-time quotes, order placement (market/limit/SL),
portfolio tracking with P&L, watchlists, technical charts,
funds management, contract notes (SEBI-compliant),
margin trading with RMS (Risk Management System).
Compliance: SEBI broker regulations, exchange margin requirements, DP integration.""",
    ),
}

SAMPLE_REQUIREMENTS = {
    "Lending MVP": FINANCIAL_TEMPLATES["loan_origination"][1],
    "Payments Routing": FINANCIAL_TEMPLATES["payment_gateway"][1],
    "KYC/AML": FINANCIAL_TEMPLATES["kyc_aml"][1],
}

CARD_CSS = """
<style>
:root {
  --accent: #1DB954;
  --accent-dark: #169a45;
  --text: #1A1A1A;
  --muted: #4A4A4A;
  --muted-2: #8A8A8A;
  --border: #E8E8E8;
  --bg: #FFFFFF;
  --bg-2: #FAFAFA;
  --panel-soft: #F5F5F5;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stAppViewContainer"] > .main { padding-top: 2.5rem !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1080px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-2);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: var(--muted); }
[data-testid="stSidebar"] div.stButton > button {
  background: var(--bg) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] div.stButton > button:hover {
  background: var(--panel-soft) !important;
  border-color: var(--accent) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
[data-testid="stSidebarContent"]::-webkit-scrollbar-track { background: var(--panel-soft); }
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb { background: rgba(29,185,84,0.3); }

/* ── Input fields ── */
.stTextInput input, .stTextArea textarea {
  background: var(--bg) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(29,185,84,0.12) !important;
}

/* ── App layout classes ── */
.hero-card {
  padding: 0.35rem 0 0.9rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.9rem;
}
.hero-title {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0;
  color: var(--text);
  margin-bottom: 0.2rem;
}
.hero-subtitle { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 760px; }
.metric-card, .soft-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  background: var(--bg);
  box-shadow: var(--shadow);
}
button[kind="primary"] {
  border-radius: 8px !important;
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #FFFFFF !important;
}
.small-muted { color: var(--muted); font-size: 0.88rem; }
.auth-shell {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  background: var(--bg-2);
  margin-bottom: 1rem;
}
.workflow-band {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  background: var(--bg);
  box-shadow: var(--shadow);
}
.footer {
  text-align:center;
  color: var(--muted);
  padding: 1.2rem 0 0.3rem 0;
  font-size: 0.92rem;
}
.footer a { color: var(--accent); text-decoration:none; font-weight:700; }
.auth-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.4rem 0.9rem;
  background: rgba(29,185,84,0.08);
  border: 1px solid rgba(29,185,84,0.25);
  border-radius: 6px;
  font-size: 0.88rem;
  color: var(--text);
  margin-bottom: 0.6rem;
}
.auth-badge-check { color: var(--accent); font-weight: 700; font-size: 1rem; }
.auth-badge-email { font-weight: 600; color: var(--text); }
.auth-badge-plan { color: var(--muted); margin-left: 0.2rem; }
code { white-space: pre-wrap !important; }

/* ── Expanders ── */
.streamlit-expanderHeader { background: var(--panel-soft); color: var(--text); }
details summary { color: var(--text); }
[data-testid="stExpander"] { background: var(--bg); border-color: var(--border); }

/* ── Combobox / Select overrides ── */
[data-baseweb="select"] {
  background: var(--bg) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
[data-baseweb="select"] [data-baseweb="tag"] {
  background: var(--panel-soft) !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] {
  background: var(--bg) !important;
  border-color: var(--border) !important;
}
[data-baseweb="popover"] li {
  color: var(--text) !important;
  background: var(--bg) !important;
}
[data-baseweb="popover"] li:hover {
  background: var(--panel-soft) !important;
}
[data-baseweb="popover"] li[aria-selected="true"] {
  color: var(--accent) !important;
  background: rgba(29,185,84,0.08) !important;
}

/* ── Universal light-theme overrides (zero dark backgrounds) ── */
h1, h2, h3, h4, h5, h6, p, li, label, span, div { color: var(--text); }
.stAlert { background: var(--panel-soft) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stMetric"] { background: var(--bg) !important; border-color: var(--border) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stTooltip"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stButton button {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  transition: all 0.2s !important;
}
.stButton button:hover { border-color: var(--accent) !important; background: rgba(29,185,84,0.08) !important; }
[data-testid="stBaseButton-secondary"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="secondary"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stSidebar"] button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] button[kind="primary"]:hover { background: #1ED760 !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: var(--panel-soft) !important; border-color: var(--border) !important; }
[data-baseweb="input"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--bg) !important;
}
[data-baseweb="input"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted-2) !important; }

/* ── Mobile ── */
@media (max-width: 640px) {
  .block-container { padding-top: 1.9rem; padding-left: 0.8rem; padding-right: 0.8rem; }
  .hero-title { font-size: 1.55rem; }
  .hero-subtitle { font-size: 0.94rem; }
  [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
}
</style>
"""


# -----------------------------------------------------------------------------
# Runtime and UI utility helpers
# -----------------------------------------------------------------------------


def bootstrap_environment() -> None:
    """Populate os.environ from Streamlit secrets so third-party SDKs can auto-detect keys."""
    for key in (
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET",
        "RAZORPAY_WEBHOOK_SECRET",
    ):
        value = safe_secret(key, "")
        if value and not os.getenv(key):
            os.environ[key] = value


def require_runtime_dependencies(vision: bool = False) -> bool:
    missing = []
    if Agent is None:
        missing.append(f"agno Agent ({AGNO_IMPORT_ERROR})")
    if Team is None:
        missing.append(f"agno Team ({TEAM_IMPORT_ERROR})")
    if OpenAIChat is None:
        missing.append(f"agno OpenAIChat ({OPENAI_CHAT_IMPORT_ERROR})")
    if vision and AgnoImage is None:
        missing.append(f"agno.media.Image ({AGNO_IMAGE_IMPORT_ERROR})")
    if vision and Gemini is None:
        # Gemini has an OpenAI-compatible fallback, so this is a warning, not a hard stop.
        st.warning("Agno Gemini class was not importable; using the OpenAI-compatible Google fallback.")
    if missing:
        st.error("Missing runtime dependencies: " + "; ".join(missing))
        st.info("Install dependencies with `pip install -r requirements.txt`.")
        return False
    return True


def require_api_keys(vision: bool = False) -> bool:
    missing = []
    if not safe_secret("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    if vision and not safe_secret("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if missing:
        st.error("Missing required API key(s): " + ", ".join(missing))
        st.info("Add them to Streamlit secrets or environment variables before running analysis.")
        return False
    return True


def event_content(event: Any) -> str:
    for attr in ("content", "content_delta", "delta", "text", "message"):
        value = getattr(event, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(event, dict):
        for key in ("content", "content_delta", "delta", "text", "message"):
            value = event.get(key)
            if isinstance(value, str):
                return value
    return ""


def stream_to_markdown(run_callable: Callable[[bool], Any], placeholder: Any) -> str:
    """Stream Agno output to a Streamlit markdown placeholder with non-stream fallback."""
    output = ""
    try:
        stream = run_callable(True)
        for event in stream:
            chunk = event_content(event)
            if chunk:
                output += chunk
                placeholder.markdown(output)
        if output.strip():
            return output.strip()
    except TypeError:
        # Some older providers do not support stream=True consistently.
        pass
    except Exception as exc:
        log_error("streaming_analysis_failed", exc, {"fallback": "non_streaming"})
        placeholder.warning(f"Streaming failed; retrying in non-streaming mode. Details: {exc}")

    try:
        response = run_callable(False)
        output = response_content(response).strip()
        placeholder.markdown(output or "No response returned.")
        return output
    except Exception as exc:
        log_error("non_streaming_analysis_failed", exc)
        raise


def parse_questions(raw_text: str) -> List[str]:
    questions: List[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*[-*•]?\s*\d{0,2}[.)-]?\s*", "", line).strip()
        if "?" in cleaned and len(cleaned) > 8:
            questions.append(cleaned)
    if not questions:
        questions = [
            "Which user persona is the primary MVP target, and what outcome must they achieve first?",
            "Which external systems, APIs, or vendors are mandatory for launch?",
            "What compliance obligations, approval flows, and audit records are non-negotiable?",
            "What edge cases, exception paths, or failure states should the team prioritize?",
        ]
    return questions[:5]


def truncate(text: str, length: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[: length - 1] + "…" if len(compact) > length else compact


# -----------------------------------------------------------------------------
# File extraction, PDF export, and Mermaid rendering
# -----------------------------------------------------------------------------


def extract_pdf_text(uploaded_file: Any) -> str:
    if pdfplumber is None:
        return "pdfplumber is not installed. Add `pdfplumber` to requirements.txt."
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip() or "No extractable text found in PDF. Try uploading a text-based PDF or image screenshot."
    except Exception as exc:
        return f"PDF extraction failed: {exc}"


def render_mermaid(mermaid_code: str, theme: str = "neutral", height: int = 560) -> None:
    safe_theme = theme if theme in {"default", "neutral", "dark", "forest", "base"} else "neutral"
    safe_code = html.escape(mermaid_code)
    components.html(
        f"""
        <div style="width:100%; overflow:auto; border:1px solid #e2e8f0; border-radius:16px; padding:16px; background:#fff;" id="mermaid-container">
            <pre class="mermaid" style="display:none;">{safe_code}</pre>
            <div id="mermaid-error" style="display:none; color:#dc2626; padding:12px; background:#fef2f2; border-radius:8px; font-size:0.9rem;">
                ⚠️ Diagram rendering failed. The Mermaid code may have syntax errors. 
                <details><summary>Show code</summary><pre style="white-space:pre-wrap;font-size:0.8rem;">{safe_code}</pre></details>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            (function() {{
                var container = document.getElementById('mermaid-container');
                var pre = container.querySelector('.mermaid');
                var errorDiv = document.getElementById('mermaid-error');
                var code = pre.textContent;
                pre.style.display = 'block';
                try {{
                    mermaid.parse(code);
                    mermaid.initialize({{
                        startOnLoad: true,
                        theme: "{safe_theme}",
                        securityLevel: "loose",
                    }});
                }} catch(e) {{
                    pre.style.display = 'none';
                    errorDiv.style.display = 'block';
                    errorDiv.innerHTML += '<br><small>' + e.message.replace(/</g, '&lt;') + '</small>';
                }}
            }})();
        </script>
        """,
        height=height,
        scrolling=True,
    )


# -----------------------------------------------------------------------------
# Streamlit UI state and rendering
# -----------------------------------------------------------------------------


def init_session_state() -> None:
    defaults = {
        "requirements_area": FINANCIAL_TEMPLATES["loan_origination"][1],
        "selected_template": "loan_origination",
        "_last_template": "loan_origination",
        "last_result": "",
        "last_mermaid": "",
        "history": [],
        "interactive_stage": "input",
        "interactive_questions": [],
        "interactive_answers": {},
        "last_uploaded_signature": "",
        "analysis_type": "Standard",
        "project_name": "Indian Fintech Product",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_interactive() -> None:
    st.session_state["interactive_stage"] = "input"
    st.session_state["interactive_questions"] = []
    st.session_state["interactive_answers"] = {}


def render_footer() -> None:
    st.markdown(
        "<p class='footer'>Built by <a href='https://touseefshaik.com' target='_blank'>Touseef Shaik</a> · touseefshaik.com</p>",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">BA Assistant</div>
          <p class="hero-subtitle">Paste rough requirements and generate a structured BA report with scope, user stories, risks, architecture notes, and diagrams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config(email: str = "", user: Optional[Dict[str, Any]] = None) -> AppConfig:
    with st.sidebar:
        st.markdown("### BA Assistant")
        if email:
            user = user or get_user(email) or create_user(email)
            plan = str(user.get("plan", "free")).title()
            used = user.get("analyses_used", user.get("usage_count", 0))
            limit = user.get("analyses_limit", user.get("usage_limit", 2))
            st.success("Verified")
            st.caption(f"{email} · {plan} · {used}/{limit if int(limit) < 10_000 else 'Unlimited'}")
            render_pricing(email, user)
            if st.button("Sign out", use_container_width=True):
                sign_out()
                st.rerun()
        else:
            st.info("Sign in from the main page to generate reports.")

        st.markdown("---")
        st.markdown("### Settings")
        project_name = st.text_input("Project Name", key="project_name")
        visible_modes = ["Standard", "Interactive (Q&A)"]
        mode_index = visible_modes.index(st.session_state.get("analysis_type", "Standard")) if st.session_state.get("analysis_type", "Standard") in visible_modes else 0
        analysis_type = st.radio("Report mode", visible_modes, index=mode_index, key="analysis_type")
        st.caption(ANALYSIS_TYPE_INFO.get(analysis_type, ""))

        with st.expander("Advanced", expanded=False):
            deep_team = st.toggle("Use Deep Team mode", value=False, help="Runs the five-agent review path for complex initiatives.")
            if deep_team:
                analysis_type = "Deep Team"
            render_mermaid_flag = st.toggle("Render Mermaid diagrams", value=True)
            mermaid_theme = st.selectbox("Mermaid Theme", ["neutral", "default", "dark", "forest", "base"], index=0)
            add_confetti = st.toggle("Confetti after report", value=False)
            show_prompt_preview = st.toggle("Show prompt preview", value=False)
            show_member_responses = st.toggle("Show member responses", value=False)

        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        for label, sample in SAMPLE_REQUIREMENTS.items():
            if st.button(label, use_container_width=True):
                st.session_state["requirements_area"] = sample
                st.session_state["selected_template"] = "none"
                reset_interactive()
                st.rerun()

        st.markdown("---")
        history = st.session_state.get("history", [])
        if history:
            st.markdown("---")
            st.markdown("### Recent reports")
            for item in history[:5]:
                st.caption(f"{item['time']} · {item['project']}")

        st.markdown("---")
        st.caption("Privacy note: requirements are sent to configured model providers only when you run analysis. Image extraction uses Gemini only when you explicitly upload an image and click extract.")
        render_footer()

    config = AppConfig(
        project_name=project_name,
        analysis_type=analysis_type,
        model_id=TEXT_ANALYSIS_MODEL_ID,
        render_mermaid=render_mermaid_flag,
        mermaid_theme=mermaid_theme,
        add_confetti=add_confetti,
        show_prompt_preview=show_prompt_preview,
        show_member_responses=show_member_responses,
    )
    return config


def run_paid_gate(email: str, consume_usage: bool = True) -> bool:
    if not email:
        st.error("Sign in with a verified email before running analysis.")
        return False
    allowed, message, _user = gate_analysis(email, consume_usage=consume_usage)
    if not allowed:
        st.error(message)
        return False
    if message:
        st.caption(message)
    return True


def requirements_flow_dependencies() -> RequirementsFlowDependencies:
    return RequirementsFlowDependencies(
        financial_templates=FINANCIAL_TEMPLATES,
        report_structure=REPORT_STRUCTURE,
        extract_pdf_text_fn=extract_pdf_text,
        require_runtime_dependencies_fn=require_runtime_dependencies,
        require_api_keys_fn=require_api_keys,
        analyzer_factory=RequirementAnalyzer,
        reset_interactive_fn=reset_interactive,
        run_paid_gate_fn=run_paid_gate,
        parse_questions_fn=parse_questions,
        stream_to_markdown_fn=stream_to_markdown,
        extract_mermaid_code_fn=extract_mermaid_code,
        save_history_fn=save_history,
        safe_secret_fn=safe_secret,
    )


def render_downloads(config: AppConfig, result: str) -> None:
    if not result:
        return
    filename_base = re.sub(r"[^A-Za-z0-9_-]+", "_", config.project_name or "ba_report").strip("_").lower() or "ba_report"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Download MD",
            data=result.encode("utf-8"),
            file_name=f"{filename_base}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download TXT",
            data=re.sub(r"`{3}.*?`{3}", "", result, flags=re.DOTALL).encode("utf-8"),
            file_name=f"{filename_base}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col3:
        try:
            pdf_bytes = generate_pdf(config.project_name, result)
        except Exception as exc:
            log_error("pdf_export_failed", exc, {"project_name": config.project_name})
            pdf_bytes = b""
        if pdf_bytes:
            st.download_button(
                "\U0001f4c4 Export as PDF",
                data=pdf_bytes,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF unavailable — download Markdown instead")


def main() -> None:
    bootstrap_environment()
    init_session_state()
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    render_header()

    if PAYMENT_IMPORT_ERROR is not None:
        st.warning(f"payment.py import fallback active: {PAYMENT_IMPORT_ERROR}")

    verified, email, user = render_auth_panel()
    if not verified:
        st.markdown("<p class='small-muted'>Sign in once to unlock BA report generation and saved report history.</p>", unsafe_allow_html=True)

    if verified and not st.session_state.get("_history_loaded_for") == email:
        st.session_state["history"] = load_history(email, safe_secret, log_error)
        st.session_state["_history_loaded_for"] = email

    config = sidebar_config(email if verified else "", user if verified else None)
    flow_deps = requirements_flow_dependencies()

    st.markdown("### Requirements")
    render_template_selector(flow_deps)
    requirements_text = st.text_area(
        "Paste requirements",
        key="requirements_area",
        height=240,
        placeholder="Paste raw requirements, PRD notes, emails, meeting transcripts, or extracted document text here.",
        label_visibility="collapsed",
    )

    with st.expander("Upload or extract from a document", expanded=False):
        render_upload_area(config, flow_deps)

    render_prompt_preview(config, requirements_text, flow_deps)

    col_analyze, col_clear = st.columns([3, 1])
    with col_analyze:
        analyze_clicked = st.button(
            "Generate BA Report",
            type="primary",
            use_container_width=True,
            disabled=(not verified or config.analysis_type == "Interactive (Q&A)"),
        )
    with col_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

    if not verified:
        st.info("Sign in above to enable report generation. You can still paste requirements while you wait for the verification code.")

    if clear_clicked:
        st.session_state["requirements_area"] = ""
        st.session_state["last_result"] = ""
        st.session_state["last_mermaid"] = ""
        reset_interactive()
        st.rerun()

    if config.analysis_type == "Interactive (Q&A)":
        render_interactive_flow(config, email, requirements_text, flow_deps)

    if analyze_clicked and config.analysis_type != "Interactive (Q&A)":
        if not requirements_text.strip():
            st.warning("Add requirements or upload a document first.")
        elif not run_paid_gate(email, consume_usage=True):
            pass
        elif not require_runtime_dependencies(False) or not require_api_keys(False):
            pass
        else:
            placeholder = st.empty()
            with st.spinner("Generating BA report..."):
                analyzer = RequirementAnalyzer(config.model_id, config.show_member_responses, enable_vision=False)
                result = stream_to_markdown(
                    lambda stream: analyzer.run_analysis(requirements_text, config.project_name, config.analysis_type, stream=stream),
                    placeholder,
                )
            st.session_state["last_result"] = result
            st.session_state["last_mermaid"] = extract_mermaid_code(result)
            st.session_state["history"] = save_history(
                config.project_name,
                config.analysis_type,
                result,
                st.session_state.get("history", []),
                safe_secret,
                email=email,
            )
            if config.add_confetti:
                st.balloons()
            st.success("Report generated.")

    result = st.session_state.get("last_result", "")
    if result:
        st.markdown("---")
        st.markdown("### Report")
        st.markdown(result)
        render_downloads(config, result)

        with st.expander("Diagram", expanded=False):
            diagram_clicked = st.button("Generate or refresh diagram", use_container_width=True)
            if diagram_clicked:
                if not requirements_text.strip():
                    st.warning("Add requirements before generating a diagram.")
                elif not run_paid_gate(email, consume_usage=False):
                    pass
                elif not require_runtime_dependencies(False) or not require_api_keys(False):
                    pass
                else:
                    with st.spinner("Generating Mermaid diagram..."):
                        analyzer = RequirementAnalyzer(config.model_id, config.show_member_responses, enable_vision=False)
                        st.session_state["last_mermaid"] = analyzer.generate_mermaid(requirements_text)

            mermaid = st.session_state.get("last_mermaid") or extract_mermaid_code(result)
            if mermaid:
                if config.render_mermaid:
                    render_mermaid(mermaid, theme=config.mermaid_theme)
                st.code(f"```mermaid\n{mermaid}\n```", language="markdown")
                st.download_button(
                    "Download Mermaid",
                    data=mermaid.encode("utf-8"),
                    file_name="ba_assistant_diagram.mmd",
                    mime="text/plain",
                    use_container_width=True,
                )

        with st.expander("Report history", expanded=False):
            history = st.session_state.get("history", [])
            if not history:
                st.info("No reports generated for this account yet.")
            for item in history:
                with st.container():
                    st.caption(f"{item['time']} · {item['project']} · {item['type']}")
                    st.download_button(
                        "Download this report",
                        data=item["result"].encode("utf-8"),
                        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', item['project']).lower()}_{item['time'].replace(':', '').replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"history_download_{item['time']}_{item['project']}",
                    )
    else:
        st.info("Your generated report will appear below the requirements form.")

    render_footer()


if __name__ == "__main__":
    main()
