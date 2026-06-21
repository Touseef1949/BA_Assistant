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
    from payment import REQUIRE_AUTH, create_user, gate_analysis, get_user, render_auth_panel, render_pricing, sign_out
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
    initial_sidebar_state="expanded",
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
:root {
  --accent: #1DB954;
  --accent-dark: #169a45;
  --accent-light: #1ED760;
  --text: #1A1A1A;
  --muted: #4A4A4A;
  --muted-2: #8A8A8A;
  --border: #E8E8E8;
  --border-strong: #D0D0D0;
  --bg: #FFFFFF;
  --bg-2: #FAFAFA;
  --panel-soft: #F5F5F5;
  --panel-softer: #F0F0F0;
  /* Tinted shadows — carry green hue instead of pure black (Taste: colored shadows) */
  --shadow: 0 1px 3px rgba(29, 185, 84, 0.04), 0 1px 2px rgba(0, 0, 0, 0.03);
  --shadow-lg: 0 4px 12px rgba(29, 185, 84, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04);
  --radius-xl: 20px;
  --radius-lg: 14px;
  --radius-md: 10px;
  --radius-sm: 8px;
  --amber: #FFA000;
  --red: #E53935;
  --blue: #2563EB;
  --violet: #7C3AED;
  --font-sans: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

* { font-family: var(--font-sans) !important; }
html { scroll-behavior: smooth; }

.stApp { background: var(--bg); color: var(--text); }
/* Completely hide Streamlit header bar */
header[data-testid="stHeader"] { display: none !important; }
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

/* ── Sidebar brand card ── */
.sidebar-brand {
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--bg);
  padding: 1rem;
  margin: 0.35rem 0 1rem;
  box-shadow: var(--shadow);
}
.sidebar-brand .logo {
  align-items: center;
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: 12px;
  display: inline-flex;
  font-size: 1.15rem;
  height: 2.3rem;
  justify-content: center;
  margin-bottom: 0.65rem;
  width: 2.3rem;
}
.sidebar-brand h2 { font-size: 1.05rem; margin: 0 0 0.25rem; line-height: 1.15; }
.sidebar-brand p { color: var(--muted) !important; font-size: 0.86rem; line-height: 1.45; margin: 0; }
.sidebar-section-title {
  color: var(--text) !important; font-size: 0.78rem; font-weight: 800;
  letter-spacing: 0.08em; margin: 1rem 0 0.45rem; text-transform: uppercase;
}
.sidebar-help-card {
  background: var(--panel-soft);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.78rem; margin-top: 0.8rem;
}
.sidebar-help-card p { color: var(--muted) !important; font-size: 0.84rem; line-height: 1.45; margin: 0; }

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

/* ── Hero / Page header ── */
.hero-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  background: var(--bg);
  box-shadow: var(--shadow);
  overflow: hidden;
  position: relative;
  padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem;
}
.hero-card::after {
  content: "";
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-light), var(--violet));
}
.hero-eyebrow {
  color: var(--accent) !important; font-size: 0.78rem; font-weight: 800;
  letter-spacing: 0.12em; margin-bottom: 0.5rem; text-transform: uppercase;
}
.hero-title {
  font-size: clamp(1.8rem, 4vw, 2.6rem); font-weight: 900;
  letter-spacing: -0.02em; line-height: 1.1;
  color: var(--text); margin-bottom: 0.4rem;
}
.hero-subtitle { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 760px; line-height: 1.55; }
.hero-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.hero-chip-row span {
  background: var(--panel-softer); border: 1px solid var(--border);
  border-radius: 999px; color: var(--muted) !important;
  font-size: 0.78rem; padding: 0.35rem 0.7rem;
}

/* ── Workflow steps ── */
.hero-workflow-row {
  display: flex; gap: 0.6rem; flex-wrap: wrap;
  margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid var(--border);
}
.hero-workflow-step {
  background: var(--panel-soft); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 0.5rem 0.75rem;
  font-size: 0.82rem; color: var(--text) !important; display: flex; align-items: center; gap: 0.4rem;
}
.hero-workflow-step em {
  background: rgba(29,185,84,0.10); color: var(--accent) !important;
  border: 1px solid rgba(29,185,84,0.20); border-radius: 50%;
  width: 1.4rem; height: 1.4rem; display: inline-flex;
  align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 700; font-style: normal;
}

/* ── Cards ── */
.metric-card, .soft-card {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  background: var(--bg);
  box-shadow: var(--shadow);
}
button[kind="primary"] {
  border-radius: var(--radius-sm) !important;
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #FFFFFF !important;
}
button[kind="primary"]:hover { background: var(--accent-light) !important; }
.small-muted { color: var(--muted); font-size: 0.88rem; }

/* ── Auth shell ── */
.auth-shell {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  background: var(--bg-2);
  margin-bottom: 1rem;
}
.auth-badge {
  display: inline-flex; align-items: center; gap: 0.6rem;
  padding: 0.4rem 0.9rem;
  background: rgba(29,185,84,0.08);
  border: 1px solid rgba(29,185,84,0.25);
  border-radius: var(--radius-sm);
  font-size: 0.88rem; color: var(--text); margin-bottom: 0.6rem;
}
.auth-badge-check { color: var(--accent); font-weight: 700; font-size: 1rem; }
.auth-badge-email { font-weight: 600; color: var(--text); }
.auth-badge-plan { color: var(--muted); margin-left: 0.2rem; }

/* ── Sample report preview ── */
.sample-report-preview {
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  background: var(--bg); box-shadow: var(--shadow);
  padding: 1.4rem; margin-top: 1rem;
}
.sample-report-head {
  display: flex; justify-content: space-between; align-items: flex-start;
  margin-bottom: 1rem;
}
.sample-kicker {
  color: var(--accent); font-size: 0.75rem; font-weight: 800;
  letter-spacing: 0.1em; text-transform: uppercase;
}
.sample-report-head h3 { font-size: 1.15rem; margin: 0.3rem 0 0; color: var(--text) !important; }
.sample-report-head p { color: var(--muted) !important; font-size: 0.85rem; margin: 0.3rem 0 0; }
.sample-verdict-card {
  background: rgba(29,185,84,0.08); border: 1px solid rgba(29,185,84,0.20);
  border-radius: var(--radius-md); padding: 0.6rem 1rem;
  text-align: center; min-width: 100px;
}
.sample-verdict-card span { display: block; font-size: 0.72rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.08em; }
.sample-verdict-card strong { font-size: 1.3rem; color: var(--accent) !important; display: block; }
.sample-verdict-card small { font-size: 0.72rem; color: var(--muted-2) !important; }
.sample-report-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;
}
.sample-report-grid article {
  background: var(--panel-soft); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 0.8rem;
}
.sample-report-grid article span { display: block; font-size: 0.74rem; color: var(--muted-2) !important; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.sample-report-grid article strong { font-size: 0.9rem; color: var(--text) !important; line-height: 1.4; }

/* ── Analysis progress ── */
.analysis-progress-shell {
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  background: var(--bg); box-shadow: var(--shadow);
  padding: 1.2rem 1.5rem; margin: 0.5rem 0;
}
.analysis-progress-shell .ap-label { font-size: 0.92rem; font-weight: 700; color: var(--text) !important; margin-bottom: 0.3rem; }
.analysis-progress-shell .ap-sub { font-size: 0.82rem; color: var(--muted) !important; }
.analysis-wit { font-size: 0.86rem; color: var(--muted) !important; font-style: italic; margin: 0.4rem 0; }

/* ── Footer ── */
.footer {
  text-align:center;
  color: var(--muted);
  padding: 1.2rem 0 0.3rem 0;
  font-size: 0.92rem;
}
.footer a { color: var(--accent); text-decoration:none; font-weight:700; }
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
h1, h2, h3, h4, h5, h6, p, li, label, span, div { color: var(--text) !important; }
.stMarkdown p, .stMarkdown span { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--muted) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted) !important; }
[data-testid="stSidebar"] { background: var(--bg-2) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: var(--muted) !important; }
.stAlert { background: var(--panel-soft) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stAlertContainer { background: var(--panel-soft) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stMetric"] { background: var(--bg) !important; border-color: var(--border) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stTooltip"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stButton button {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.2s !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stButton button:hover { border-color: var(--accent) !important; background: rgba(29,185,84,0.08) !important; }
.stButton button:active { transform: scale(0.98) !important; }
button[kind="primary"]:hover { background: var(--accent-light) !important; }
button[kind="primary"]:active { transform: scale(0.98) !important; }
h1, h2, h3, h4 { text-wrap: balance; }
p { text-wrap: pretty; }
/* Kill Streamlit's default dark radio/checkbox/toggle backgrounds — override BaseWeb emotion classes */
.stRadio div, .stCheckbox div, .stToggle div { background: var(--bg) !important; }
.stRadio [role="radio"][aria-checked="true"] div { background: var(--accent) !important; }
[data-testid="stBaseButton-secondary"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="secondary"] { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stSidebar"] button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] button[kind="primary"]:hover { background: var(--accent-light) !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: var(--panel-soft) !important; border-color: var(--border) !important; }
[data-baseweb="input"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-baseweb="input"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted-2) !important; }

/* ── Upload zone ── */
.upload-zone {
  border: 2px dashed var(--border-strong);
  border-radius: var(--radius-lg);
  background: var(--bg-2);
  padding: 1.2rem 1.4rem;
  margin: 0.8rem 0;
}
.upload-zone-label {
  font-size: 0.88rem; font-weight: 600; color: var(--text) !important; margin-bottom: 0.5rem;
}
.upload-zone-hint {
  font-size: 0.8rem; color: var(--muted-2) !important; margin-bottom: 0.8rem;
}
.upload-zone [data-testid="stFileUploader"] {
  background: var(--bg) !important;
}
.upload-zone [data-testid="stFileUploader"] section {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background: var(--bg) !important;
}
.upload-zone [data-testid="stFileUploader"] span, 
.upload-zone [data-testid="stFileUploader"] p,
.upload-zone [data-testid="stFileUploader"] label {
  color: var(--text) !important;
}
.upload-zone button[kind="secondary"] {
  background: var(--bg) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* ── Recent reports in sidebar ── */
.recent-analysis-item {
  background: var(--panel-soft) !important; border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 0.6rem 0.7rem; margin: 0.35rem 0;
}
.recent-analysis-item strong { color: var(--text) !important; font-size: 0.88rem; display: block; }
.recent-analysis-item span { color: var(--muted) !important; font-size: 0.76rem; }

/* ============================================
   MOBILE UX OVERHAUL — BA Assistant
   Overlay drawer, scrim, 14px font floor, 44px tap targets
   Mirrors Stock Research Assistant pattern
   ============================================ */

/* ── 1. SIDEBAR: FULL OVERLAY DRAWER ── */
@media (max-width: 768px) {
  [data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important; left: 0 !important; bottom: 0 !important;
    height: 100dvh !important;
    width: 85vw !important;
    max-width: 320px !important;
    min-width: 280px !important;
    z-index: 9999 !important;
    transform: translateX(-100%) !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: none !important;
  }
  [data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
  }

  /* Kill the collapse-control rail sliver */
  [data-testid="stSidebar"] > div:first-child { width: 100% !important; }

  /* Scrim / backdrop behind sidebar */
  .stApp::after {
    content: "" !important; display: block !important;
    position: fixed !important; inset: 0 !important;
    background: rgba(0, 0, 0, 0.5) !important;
    z-index: 9998 !important;
    opacity: 0 !important; visibility: hidden !important;
    transition: opacity 0.3s ease, visibility 0.3s ease !important;
    pointer-events: none !important;
    -webkit-tap-highlight-color: transparent !important;
  }
  .stApp:has([data-testid="stSidebar"][aria-expanded="true"])::after {
    opacity: 1 !important; visibility: visible !important;
    pointer-events: auto !important; cursor: pointer !important;
  }

  /* Enlarge the close button inside sidebar */
  [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] {
    position: absolute !important; top: 8px !important; right: 8px !important;
    z-index: 10001 !important;
  }
  [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] button {
    width: 48px !important; height: 48px !important; min-height: 48px !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    backdrop-filter: blur(8px) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: background 0.2s ease !important;
  }
  [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] button:hover,
  [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] button:active {
    background: rgba(0, 0, 0, 0.1) !important;
  }
  [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] button svg {
    width: 24px !important; height: 24px !important;
  }

  /* Sidebar expand button — visible FAB when sidebar is closed */
  [data-testid="stExpandSidebarButton"] {
    position: fixed !important; top: 0.75rem !important; left: 0.75rem !important;
    z-index: 10000 !important;
    width: 44px !important; height: 44px !important;
    min-height: 44px !important; min-width: 44px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 12px !important;
    border: 1px solid var(--border, #E8E8E8) !important;
    background: var(--bg, #FFFFFF) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
  }
  [data-testid="stExpandSidebarButton"] svg { width: 22px !important; height: 22px !important; }

  /* ── 2. MAIN CONTENT: FULL WIDTH, 16px PADDING ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] .block-container,
  [data-testid="stAppViewContainer"] > section {
    margin-left: 0 !important; padding-left: 0 !important;
    max-width: 100% !important; width: 100% !important;
  }
  [data-testid="stAppViewContainer"] .block-container {
    padding: 1rem 1rem 4rem 1rem !important;
    max-width: 100% !important; box-sizing: border-box !important;
  }
  [data-testid="stAppViewContainer"] > section { padding-top: 0.5rem !important; }
  [data-testid="stVerticalBlock"] { gap: 0.75rem !important; }

  /* ── 3. FONT SIZE FLOOR (14px minimum) ── */
  [data-testid="stAppViewContainer"] .block-container p,
  [data-testid="stAppViewContainer"] .block-container span,
  [data-testid="stAppViewContainer"] .block-container label,
  [data-testid="stAppViewContainer"] .block-container [data-testid="stWidgetLabel"],
  [data-testid="stAppViewContainer"] .block-container [data-testid="stWidgetLabel"] label,
  [data-testid="stAppViewContainer"] .block-container [data-testid="stMarkdown"] p,
  [data-testid="stAppViewContainer"] .block-container [data-testid="stMarkdown"] span,
  [data-testid="stAppViewContainer"] .block-container small,
  [data-testid="stAppViewContainer"] .block-container [data-testid="stCaptionContainer"],
  [data-testid="stAppViewContainer"] .block-container [data-testid="stCaptionContainer"] p {
    font-size: 0.875rem !important; line-height: 1.4 !important;
  }

  /* Headings — mobile-appropriate sizes */
  [data-testid="stAppViewContainer"] h1 { font-size: 1.5rem !important; }
  [data-testid="stAppViewContainer"] h2 { font-size: 1.25rem !important; }
  [data-testid="stAppViewContainer"] h3 { font-size: 1.125rem !important; }

  /* Sidebar labels */
  [data-testid="stSidebar"] [data-testid="stMarkdown"] span,
  [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { font-size: 0.875rem !important; }

  /* ── 4. TAP TARGETS (44px minimum) ── */
  [data-testid="stAppViewContainer"] .block-container a,
  [data-testid="stAppViewContainer"] .block-container button,
  [data-testid="stAppViewContainer"] .block-container [role="button"],
  [data-testid="stAppViewContainer"] .block-container select,
  [data-testid="stAppViewContainer"] .block-container input,
  [data-testid="stAppViewContainer"] .block-container [data-testid="baseButton-secondary"],
  [data-testid="stAppViewContainer"] .block-container [data-testid="baseButton-primary"] {
    min-height: 44px !important; min-width: 44px !important;
    display: inline-flex !important; align-items: center !important;
    padding: 0.5rem 0.75rem !important; box-sizing: border-box !important;
  }

  /* Heading anchor links */
  [data-testid="stAppViewContainer"] .block-container a[href^="#"],
  [data-testid="stAppViewContainer"] .block-container .header-anchor {
    min-height: 44px !important; min-width: 44px !important;
    display: inline-flex !important; align-items: center !important;
    justify-content: center !important;
    margin: -12px !important; padding: 12px !important;
  }

  /* Tab buttons */
  [data-testid="stTabs"] [role="tablist"] { flex-wrap: wrap !important; }
  [data-testid="stTabs"] [role="tab"] {
    flex: 1 1 auto !important; font-size: 0.875rem !important;
    min-height: 44px !important; padding: 0.5rem 0.75rem !important;
  }

  /* ── 5. COLUMNS → STACK ── */
  [data-testid="stHorizontalBlock"], .stHorizontalBlock {
    flex-direction: column !important; flex-wrap: nowrap !important; gap: 0.75rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="column"],
  [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"],
  [data-testid="stHorizontalBlock"] > div,
  .stHorizontalBlock > div {
    flex: 0 0 100% !important; width: 100% !important;
    max-width: 100% !important; min-width: 0 !important;
  }

  /* ── 6. HERO / PAGE HEADER ── */
  .hero-card { padding: 1.2rem 1rem !important; margin-bottom: 0.75rem !important; }
  .hero-title { font-size: 1.5rem !important; line-height: 1.15 !important; }
  .hero-subtitle { font-size: 0.92rem !important; max-width: 100% !important; }
  .hero-chip-row { gap: 0.4rem !important; }
  .hero-chip-row span { font-size: 0.875rem !important; padding: 0.25rem 0.5rem !important; }
  .hero-workflow-row { gap: 0.4rem !important; }
  .hero-workflow-step { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }

  /* ── 7. AUTH SHELL ── */
  .auth-shell { padding: 0.8rem !important; margin-bottom: 0.75rem !important; }
  .auth-badge { padding: 0.35rem 0.75rem !important; font-size: 0.875rem !important; }

  /* ── 8. UPLOAD ZONE ── */
  .upload-zone { padding: 0.8rem !important; }
  .upload-zone-label { font-size: 0.9rem !important; }
  .upload-zone-hint { font-size: 0.8rem !important; }

  /* ── 9. SAMPLE REPORT PREVIEW ── */
  .sample-report-preview { padding: 1rem !important; }
  .sample-report-head { flex-direction: column !important; gap: 1rem !important; }
  .sample-report-head h3 { font-size: 1.1rem !important; }
  .sample-report-grid { grid-template-columns: 1fr !important; gap: 0.6rem !important; }

  /* ── 10. METRIC / SOFT CARDS ── */
  .metric-card, .soft-card { padding: 0.8rem !important; }

  /* ── 11. DATA TABLES / CODE ── */
  [data-testid="stDataFrame"], [data-testid="stTable"], .stDataFrame {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    max-width: 100% !important; border-radius: 8px !important;
  }
  [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    white-space: nowrap !important; font-size: 0.8125rem !important;
    padding: 0.5rem 0.625rem !important;
  }
  [data-testid="stCodeBlock"] {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    max-width: 100% !important;
  }

  /* ── 12. EXPANDERS ── */
  [data-testid="stExpander"] summary {
    min-height: 48px !important; padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important;
    display: flex !important; align-items: center !important;
  }

  /* ── 13. SIDEBAR BRAND CARD ── */
  .sidebar-brand { padding: 0.8rem !important; margin: 0.35rem 0 0.75rem !important; }
  .sidebar-help-card { padding: 0.6rem !important; margin-top: 0.6rem !important; }

  /* ── 14. FOOTER ── */
  .footer { font-size: 0.8125rem !important; padding: 0.75rem 1rem !important; }
}

/* ── EXTRA-TIGHT SCREENS (<=380px, iPhone SE) ── */
@media (max-width: 380px) {
  [data-testid="stAppViewContainer"] .block-container {
    padding: 0.75rem 0.75rem 4rem 0.75rem !important;
  }
  [data-testid="stSidebar"] { width: 90vw !important; }
  [data-testid="stAppViewContainer"] h1 { font-size: 1.375rem !important; }
  [data-testid="stAppViewContainer"] h2 { font-size: 1.125rem !important; }
  .hero-card { padding: 1rem 0.75rem !important; }
  .hero-title { font-size: 1.3rem !important; }
  .sample-report-grid { grid-template-columns: 1fr !important; }

  /* Stack tab buttons vertically on very small screens */
  [data-baseweb="tab-list"] { flex-direction: column !important; }
  [data-baseweb="tab-list"] button { width: 100% !important; }
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


ROTATING_WIT = [
    "Good requirements analysis takes time — we're structuring your scope, stories, and risks.",
    "The best BAs ask the right questions before writing anything. Almost there.",
    "Turning rough notes into a polished BA document. Precision over speed.",
    "Quality requirements separate great products from mediocre ones. Stay with us.",
    "Building your report — clarity beats completeness, every single time.",
    "Deep work in progress. Every second here saves you hours of rework.",
]


def render_footer() -> None:
    st.markdown(
        "<p class='footer'>Built by <a href='https://touseefshaik.com' target='_blank'>Touseef Shaik</a> · touseefshaik.com</p>",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-eyebrow">AI-powered business analysis</div>
          <div class="hero-title">BA Assistant</div>
          <p class="hero-subtitle">Paste rough requirements and generate a structured BA report with scope, user stories, risks, architecture notes, and Mermaid diagrams — in minutes.</p>
          <div class="hero-chip-row">
            <span>&#128203; Scope &amp; requirements</span>
            <span>&#128100; User stories</span>
            <span>&#9888;&#65039; Risk analysis</span>
            <span>&#128736;&#65039; Architecture notes</span>
            <span>&#128202; Mermaid diagrams</span>
          </div>
          <div class="hero-workflow-row">
            <div class="hero-workflow-step"><em>1</em> Paste requirements</div>
            <div class="hero-workflow-step"><em>2</em> AI generates report</div>
            <div class="hero-workflow-step"><em>3</em> Download MD / PDF</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_progress(active_label: str = "Analyzing") -> None:
    """Render a progress shell card similar to SRA's analysis progress."""
    st.markdown(
        f"""
        <div class="analysis-progress-shell">
            <div class="ap-label">&#9881;&#65039; {html.escape(active_label)}</div>
            <div class="ap-sub">Multi-agent analysis in progress — this usually takes 30-90 seconds.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_report_preview() -> None:
    """Show a sample report preview card when no report has been generated yet."""
    st.markdown(
        """
        <section class="sample-report-preview" aria-label="Sample report preview">
            <div class="sample-report-head">
                <div>
                    <span class="sample-kicker">Sample report</span>
                    <h3>Loan Origination Portal</h3>
                    <p>Preview the kind of structured BA report generated after sign-in.</p>
                </div>
                <div class="sample-verdict-card">
                    <span>Report type</span>
                    <strong>Standard</strong>
                    <small>~90 seconds</small>
                </div>
            </div>
            <div class="sample-report-grid">
                <article>
                    <span>Scope</span>
                    <strong>Digital loan origination with KYC, credit assessment, and disbursal integration.</strong>
                </article>
                <article>
                    <span>Key risk</span>
                    <strong>RBI digital lending compliance, data localization, and consent framework.</strong>
                </article>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config(email: str = "", user: Optional[Dict[str, Any]] = None) -> AppConfig:
    with st.sidebar:
        # ── Brand card ──
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo">&#128203;</div>
                <h2>BA Assistant</h2>
                <p>AI-assisted business analysis with structured reports, user stories, risks, and diagrams.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Access / Auth status ──
        if email:
            user = user or get_user(email) or create_user(email)
            plan = str(user.get("plan", "free")).title()
            used = user.get("analyses_used", user.get("usage_count", 0))
            limit = user.get("analyses_limit", user.get("usage_limit", 2))
            limit_display = f"{used}/{limit}" if int(limit) < 10_000 else "Unlimited"
            st.markdown(
                f"""
                <div class="auth-badge">
                    <span class="auth-badge-check">&#10003;</span>
                    <span class="auth-badge-email">{html.escape(email)}</span>
                    <span class="auth-badge-plan">· {html.escape(plan)} · {limit_display}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_pricing(email, user)
            if st.button("Sign out", use_container_width=True):
                sign_out()
                st.rerun()
        else:
            st.markdown(
                """
                <div class="sidebar-help-card">
                    <p><strong>Free during beta.</strong> No login required to generate reports and save history.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Settings ──
        st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)
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

        # ── Quick Actions ──
        st.markdown('<div class="sidebar-section-title">&#9889; Quick Actions</div>', unsafe_allow_html=True)
        for label, sample in SAMPLE_REQUIREMENTS.items():
            if st.button(label, use_container_width=True):
                st.session_state["requirements_area"] = sample
                st.session_state["selected_template"] = "none"
                reset_interactive()
                st.rerun()

        # ── Recent reports ──
        history = st.session_state.get("history", [])
        if history:
            st.markdown("---")
            st.markdown('<div class="sidebar-section-title">Recent reports</div>', unsafe_allow_html=True)
            for item in history[:5]:
                st.markdown(
                    f"""
                    <div class="recent-analysis-item">
                        <strong>{html.escape(item.get('project', 'Untitled'))}</strong>
                        <span>{html.escape(item.get('time', ''))} · {html.escape(item.get('type', ''))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Help / Privacy ──
        st.markdown(
            """
            <div class="sidebar-help-card">
                <p><strong>How it works:</strong> Free during beta. No login required for report generation, Interactive Q&amp;A, Deep Team mode, and history.</p>
                <p style="margin-top:0.5rem; font-size:0.78rem; opacity:0.7;">Privacy: requirements are sent to model providers only when you run analysis. Image extraction uses Gemini only on explicit upload.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    if not REQUIRE_AUTH:
        # Beta mode: always allow, no email required
        allowed, message, _user = gate_analysis(email or "beta@ba-assistant.local", consume_usage=consume_usage)
        if message:
            st.caption(message)
        return True
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


def _inject_mobile_sidebar_close_js() -> None:
    """JS to close sidebar when tapping outside it (on the scrim)."""
    try:
        components.html(
            """
<script>
(function() {
  function setupScrimClose() {
    const app = document.querySelector('.stApp') || document.querySelector('[data-testid="stAppViewContainer"]');
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!app || !sidebar) {
      setTimeout(setupScrimClose, 500);
      return;
    }
    app.addEventListener('click', function(e) {
      if (sidebar.getAttribute('aria-expanded') !== 'true') return;
      const sidebarRect = sidebar.getBoundingClientRect();
      if (e.clientX > sidebarRect.right || e.clientX < sidebarRect.left) {
        const collapseBtn = sidebar.querySelector(
          '[data-testid="stSidebarCollapseButton"] button'
        );
        if (collapseBtn) collapseBtn.click();
      }
    }, { capture: true });
  }
  setTimeout(setupScrimClose, 1000);
  if (document.body) {
    const observer = new MutationObserver(function() {
      clearTimeout(window._scrimTimer);
      window._scrimTimer = setTimeout(setupScrimClose, 500);
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
""",
            height=0,
        )
    except Exception:
        pass


def main() -> None:
    bootstrap_environment()
    init_session_state()
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    _inject_mobile_sidebar_close_js()
    render_header()

    if PAYMENT_IMPORT_ERROR is not None:
        st.warning(f"payment.py import fallback active: {PAYMENT_IMPORT_ERROR}")

    verified, email, user = render_auth_panel()
    if not verified:
        st.markdown("<p class='small-muted'>No login required — free during beta.</p>", unsafe_allow_html=True)
        render_footer()
        return

    # ── Authenticated: full experience ──
    if not st.session_state.get("_history_loaded_for") == email:
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

    st.markdown(
        '<p class="upload-zone-label">📎 Upload a document (optional)</p>'
        '<p class="upload-zone-hint">Drag a PRD, email screenshot, whiteboard photo, or PDF — we\'ll extract the requirements for you.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    render_upload_area(config, flow_deps)
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.info("No login required — free during beta. You can paste requirements and generate a report.")

    if clear_clicked:
        st.session_state["requirements_area"] = ""
        st.session_state["last_result"] = ""
        st.session_state["last_mermaid"] = ""
        reset_interactive()
        st.rerun()

    if config.analysis_type == "Interactive (Q&A)":
        render_interactive_flow(config, email, requirements_text, flow_deps)

    if analyze_clicked and config.analysis_type != "Interactive (Q&A)":  # pragma: no cover — main analysis flow, covered by AppTest
        if not requirements_text.strip():
            st.warning("Add requirements or upload a document first.")
        elif not run_paid_gate(email, consume_usage=True):
            pass
        elif not require_runtime_dependencies(False) or not require_api_keys(False):
            pass
        else:
            progress_shell = st.empty()
            wit_placeholder = st.empty()
            with progress_shell.container():
                render_analysis_progress("Generating BA report")
            wit_placeholder.markdown(
                f'<p class="analysis-wit">&#128161; {ROTATING_WIT[0]}</p>',
                unsafe_allow_html=True,
            )
            placeholder = st.empty()
            with st.spinner("Generating BA report..."):
                analyzer = RequirementAnalyzer(config.model_id, config.show_member_responses, enable_vision=False)
                result = stream_to_markdown(
                    lambda stream: analyzer.run_analysis(requirements_text, config.project_name, config.analysis_type, stream=stream),
                    placeholder,
                )
            progress_shell.empty()
            wit_placeholder.empty()
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
        render_sample_report_preview()

    render_footer()


if __name__ == "__main__":
    main()
