"""Payment layer for BA Assistant — Supabase + Razorpay integration."""

import hashlib
import hmac
import json
import os
import time
from typing import Optional

import requests
import streamlit as st
from supabase import create_client, Client

# =============================================================================
# Config
# =============================================================================

FREE_TIER_LIMIT = 3
PLANS = {
    "free": {"name": "Free", "analyses": 3, "price": 0},
    "pro": {"name": "Pro", "analyses": -1, "price": 49900},     # paise
    "team": {"name": "Team", "analyses": -1, "price": 199900},   # paise
}

# =============================================================================
# Supabase
# =============================================================================

@st.cache_resource
def get_supabase():
    """Cached Supabase client. Returns None if not configured."""
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
    key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
    if not url or not key:
        return None
    return create_client(url, key)


def _supabase_ready() -> bool:
    return get_supabase() is not None


def get_user(email: str) -> Optional[dict]:
    supabase = get_supabase()
    if not supabase:
        return {"email": email, "plan": "free", "analyses_used": 0, "analyses_limit": FREE_TIER_LIMIT}
    result = supabase.table("users").select("*").eq("email", email).execute()
    return result.data[0] if result.data else None


def create_user(email: str) -> dict:
    supabase = get_supabase()
    user = {
        "email": email,
        "plan": "free",
        "analyses_used": 0,
        "analyses_limit": FREE_TIER_LIMIT,
    }
    if not supabase:
        return user
    user["created_at"] = "now()"
    result = supabase.table("users").insert(user).execute()
    return result.data[0] if result.data else user


def increment_usage(email: str) -> dict:
    supabase = get_supabase()
    if not supabase:
        return {"email": email, "analyses_used": 0}
    user = get_user(email)
    if not user:
        user = create_user(email)
    new_count = user["analyses_used"] + 1
    supabase.table("users").update({"analyses_used": new_count}).eq("email", email).execute()
    user["analyses_used"] = new_count
    return user


def user_can_analyze(email: str) -> bool:
    if not _supabase_ready():
        return True  # no paywall without Supabase
    user = get_user(email)
    if not user:
        return True
    return user["plan"] != "free" or user["analyses_used"] < user["analyses_limit"]


def activate_pro(email: str, plan: str) -> None:
    supabase = get_supabase()
    if not supabase:
        return
    limit = 999999
    supabase.table("users").update({
        "plan": plan,
        "analyses_limit": limit,
        "analyses_used": 0,
    }).eq("email", email).execute()


# =============================================================================
# Razorpay
# =============================================================================

def get_razorpay_creds():
    key_id = st.secrets.get("RAZORPAY_KEY_ID", os.getenv("RAZORPAY_KEY_ID", ""))
    key_secret = st.secrets.get("RAZORPAY_KEY_SECRET", os.getenv("RAZORPAY_KEY_SECRET", ""))
    return key_id, key_secret


def create_payment_link(email: str, plan: str) -> Optional[str]:
    """Create a Razorpay payment link for the given plan."""
    key_id, key_secret = get_razorpay_creds()
    if not key_id or not key_secret:
        return None

    plan_info = PLANS.get(plan, PLANS["pro"])
    amount = plan_info["price"]

    payload = {
        "amount": amount,
        "currency": "INR",
        "accept_partial": False,
        "description": f"BA Assistant — {plan_info['name']} Plan",
        "customer": {"email": email},
        "notify": {"email": True, "sms": False},
        "reminder_enable": True,
        "notes": {"plan": plan, "email": email},
        "callback_url": "",
        "callback_method": "get",
    }

    resp = requests.post(
        "https://api.razorpay.com/v1/payment_links",
        auth=(key_id, key_secret),
        json=payload,
        timeout=15,
    )

    if resp.status_code == 200:
        data = resp.json()
        return data.get("short_url")
    return None


def verify_payment(payment_id: str) -> bool:
    """Verify a Razorpay payment."""
    key_id, key_secret = get_razorpay_creds()
    if not key_id or not key_secret:
        return False

    resp = requests.get(
        f"https://api.razorpay.com/v1/payments/{payment_id}",
        auth=(key_id, key_secret),
        timeout=10,
    )

    if resp.status_code == 200:
        data = resp.json()
        return data.get("status") == "captured"
    return False


# =============================================================================
# Streamlit UI Components
# =============================================================================

def render_pricing(email: str):
    """Render pricing cards and handle upgrades."""
    user = get_user(email)
    current_plan = user["plan"] if user else "free"
    usage = user["analyses_used"] if user else 0
    limit = user["analyses_limit"] if user else FREE_TIER_LIMIT

    st.sidebar.divider()
    st.sidebar.subheader("💳 Your Plan")

    if current_plan == "free":
        remaining = max(0, limit - usage)
        st.sidebar.metric("Free Analyses Left", remaining)
        if remaining == 0:
            st.sidebar.warning("Free limit reached!")

        st.sidebar.divider()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✨ Upgrade Pro", use_container_width=True, type="primary"):
                link = create_payment_link(email, "pro")
                if link:
                    st.session_state["payment_link"] = link
                    st.session_state["show_payment"] = True
                else:
                    st.error("Razorpay not configured")
        with col2:
            if st.button("👥 Team", use_container_width=True):
                link = create_payment_link(email, "team")
                if link:
                    st.session_state["payment_link"] = link
                    st.session_state["show_payment"] = True
                else:
                    st.error("Razorpay not configured")
    else:
        st.sidebar.success(f"🎉 {current_plan.upper()} Plan — Unlimited")
        st.sidebar.caption(f"Analyses used: {usage}")

    # Payment modal
    if st.session_state.get("show_payment"):
        link = st.session_state.get("payment_link", "")
        st.sidebar.markdown(f"""
        ### Complete Payment
        [Click here to pay →]({link})

        After payment, paste your **Payment ID** below:
        """)
        payment_id = st.sidebar.text_input("Payment ID", key="payment_id_input")
        if payment_id and st.sidebar.button("Verify Payment", type="primary"):
            if verify_payment(payment_id):
                plan = "pro" if "499" in link else "team"
                activate_pro(email, plan)
                st.session_state["show_payment"] = False
                st.sidebar.success("✅ Payment verified! Refreshing...")
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("Payment not found or failed")


def gate_analysis(email: str) -> bool:
    """Returns True if user can run analysis, False if paywall needed."""
    if not email or "@" not in email:
        st.warning("Enter your email to track usage.")
        return False

    if user_can_analyze(email):
        increment_usage(email)
        return True

    st.error("🚫 Free limit reached. Upgrade to Pro for unlimited analyses.")
    return False
