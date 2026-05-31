"""
BA Assistant with Payment Layer
===============================
Supabase-backed usage tracking + Razorpay payments.
Free: 3 analyses → Pro: ₹499/mo unlimited → Team: ₹1,999/mo 5 seats.

Secrets needed (Streamlit Cloud):
    DEEPSEEK_API_KEY = "..."
    SUPABASE_URL = "https://xxx.supabase.co"
    SUPABASE_KEY = "eyJ..."
    RAZORPAY_KEY_ID = "rzp_test_..."
    RAZORPAY_KEY_SECRET = "..."
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from agno.agent import Agent
from agno.models.deepseek import DeepSeek

# ── Config ──────────────────────────────────────────────────────
FREE_LIMIT = 3
PRO_PRICE = 49900    # paise
TEAM_PRICE = 199900  # paise

MODEL_ID = "deepseek-v4-pro"

# ── Supabase Helpers ─────────────────────────────────────────────

@st.cache_resource
def get_supabase():
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

def ensure_user(email: str) -> dict:
    """Get or create user record."""
    supabase = get_supabase()
    result = supabase.table("users").select("*").eq("email", email).execute()
    if result.data:
        return result.data[0]
    # New user
    user = {"email": email, "analyses_used": 0, "plan": "free",
            "activated_at": datetime.utcnow().isoformat(), "team_size": 1}
    supabase.table("users").insert(user).execute()
    return user

def can_analyze(user: dict) -> bool:
    if user["plan"] != "free":
        return True
    return user["analyses_used"] < FREE_LIMIT

def use_credit(email: str):
    supabase = get_supabase()
    user = ensure_user(email)
    if user["plan"] == "free":
        supabase.table("users").update(
            {"analyses_used": user["analyses_used"] + 1}
        ).eq("email", email).execute()

# ── Razorpay Helpers ─────────────────────────────────────────────

def create_payment_link(plan: str, email: str) -> str:
    """Create a Razorpay payment link for subscription."""
    import base64
    key_id = st.secrets["RAZORPAY_KEY_ID"]
    key_secret = st.secrets["RAZORPAY_KEY_SECRET"]
    
    price = PRO_PRICE if plan == "pro" else TEAM_PRICE
    label = "Pro ₹499/mo" if plan == "pro" else "Team ₹1,999/mo"
    
    auth = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    
    resp = requests.post("https://api.razorpay.com/v1/payment_links", json={
        "amount": price,
        "currency": "INR",
        "accept_partial": False,
        "description": f"BA Assistant — {label}",
        "customer": {"email": email},
        "notify": {"email": True},
        "reminder_enable": True,
        "notes": {"plan": plan, "email": email},
        "callback_url": "https://businessanalysttools.streamlit.app",
        "callback_method": "get",
    }, headers={"Authorization": f"Basic {auth}"})
    
    return resp.json().get("short_url", "")

def verify_payment(payment_id: str) -> bool:
    """Verify a payment was successful."""
    key_id = st.secrets["RAZORPAY_KEY_ID"]
    key_secret = st.secrets["RAZORPAY_KEY_SECRET"]
    import base64
    auth = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    resp = requests.get(f"https://api.razorpay.com/v1/payments/{payment_id}",
                        headers={"Authorization": f"Basic {auth}"})
    return resp.json().get("status") == "captured"

def activate_user(email: str, plan: str):
    """Upgrade user to paid plan."""
    supabase = get_supabase()
    supabase.table("users").update({
        "plan": plan,
        "activated_at": datetime.utcnow().isoformat(),
    }).eq("email", email).execute()

# ── AI Agent ─────────────────────────────────────────────────────

@st.cache_resource
def get_agent():
    return Agent(
        name="BA Assistant",
        role="Senior BA/PO who produces complete requirement analysis reports",
        model=DeepSeek(id=MODEL_ID),
        instructions=[
            "Produce ONE complete, structured BA/PO report from the requirements.",
            "Follow this structure: Executive Summary → Requirements → Features → User Stories → NFRs → Architecture → Risks → MVP Plan → Mermaid Diagram.",
            "Use MoSCoW, INVEST, Given-When-Then. Include at least one Mermaid diagram.",
            "Do not repeat sections. Output only the report.",
        ],
        markdown=True,
    )

def run_analysis(requirements: str, project_name: str) -> str:
    prompt = f"""Project: {project_name}

Analyze these requirements and produce a complete BA/PO report:

{requirements}

Follow the standard report structure. Include Mermaid diagrams where relevant."""

    agent = get_agent()
    response = agent.run(prompt, stream=False)
    return response.content if hasattr(response, 'content') else str(response)

# ── UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="BA Assistant", page_icon="📋", layout="wide")

st.title("📋 BA Assistant")
st.caption("5 agents → 1 report. Now with payments.")

# ── Sidebar: User + Pricing ──────────────────────────────────────
with st.sidebar:
    st.header("👤 Account")
    email = st.text_input("Email", placeholder="you@company.com",
                          value=st.session_state.get("email", ""))
    if email:
        st.session_state["email"] = email
        user = ensure_user(email)
        
        used = user["analyses_used"]
        plan = user["plan"]
        
        if plan == "free":
            remaining = max(0, FREE_LIMIT - used)
            st.metric("Free analyses left", remaining)
            st.progress(used / FREE_LIMIT, text=f"{used}/{FREE_LIMIT}")
            
            if remaining == 0:
                st.warning("Free tier exhausted!")
    
    # ── Pricing ──
    st.divider()
    st.subheader("💳 Plans")
    
    if st.button("🚀 Upgrade to Pro — ₹499/mo", use_container_width=True, 
                 disabled=not email or plan != "free"):
        if email:
            url = create_payment_link("pro", email)
            st.markdown(f"[Click here to pay ₹499]({url})", unsafe_allow_html=True)
    
    if st.button("👥 Upgrade to Team — ₹1,999/mo", use_container_width=True,
                 disabled=not email or plan != "free"):
        if email:
            url = create_payment_link("team", email)
            st.markdown(f"[Click here to pay ₹1,999]({url})", unsafe_allow_html=True)
    
    st.divider()
    st.caption(f"Plan: {plan.upper()} | Analyses used: {used}")

# ── Payment verification (callback) ──────────────────────────────
query = st.query_params
if "razorpay_payment_id" in query:
    payment_id = query["razorpay_payment_id"]
    if verify_payment(payment_id):
        plan_note = query.get("plan", "pro")
        activate_user(st.session_state.get("email", ""), plan_note)
        st.success("✅ Payment verified! Your account is now active.")
        st.balloons()

# ── Main: Analysis Input ─────────────────────────────────────────
st.subheader("Paste Requirements")
requirements = st.text_area("Requirements", height=200,
    placeholder="We need an e-commerce platform with user login, product catalog, cart, checkout...")
project_name = st.text_input("Project Name", value="Untitled Project")

can_run = email and can_analyze(ensure_user(email)) if email else False

if st.button("🔍 Analyze Requirements", type="primary", disabled=not can_run):
    if not email:
        st.warning("Enter your email in the sidebar first.")
    elif not can_run:
        st.warning("Free tier exhausted. Upgrade to continue.")
    else:
        with st.spinner("Analyzing..."):
            result = run_analysis(requirements, project_name)
            use_credit(email)
            st.markdown(result)
