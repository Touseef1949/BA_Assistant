"""Payment and usage gate for BA Assistant v2.

This module keeps Supabase + Razorpay concerns out of app.py. It is intentionally
fault-tolerant: if Supabase/Razorpay are not configured, the app can still run in
local/session mode for development.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - dependency installed in production
    Client = None  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]


def _is_local_dev() -> bool:
    """True only when the operator has explicitly opted into local-dev mode.

    In production (HF Spaces / Streamlit Cloud) this must return False so
    that Supabase or Razorpay failures are surfaced, not silently swallowed.
    """
    flag = os.environ.get("BA_ASSISTANT_LOCAL_DEV", "").strip().lower()
    return flag in ("1", "true", "yes")


FREE_USAGE_LIMIT = 2
PAID_USAGE_LIMIT = 10_000
USERS_TABLE = "users"


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def _safe_secret(name: str, default: str = "") -> str:
    try:
        raw = st.secrets.get(name, default)
        value = str(raw).strip() if raw is not None else default
    except Exception:
        value = default
    return value or os.getenv(name, default).strip()


@st.cache_resource(show_spinner=False)
def _supabase() -> Optional[Any]:
    url = _safe_secret("SUPABASE_URL")
    key = _safe_secret("SUPABASE_KEY")
    if not url or not key or create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def _razorpay_auth_header() -> Dict[str, str]:
    key_id = _safe_secret("RAZORPAY_KEY_ID")
    key_secret = _safe_secret("RAZORPAY_KEY_SECRET")
    token = base64.b64encode(f"{key_id}:{key_secret}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_users() -> Dict[str, Dict[str, Any]]:
    if "_ba_local_users" not in st.session_state:
        st.session_state["_ba_local_users"] = {}
    return st.session_state["_ba_local_users"]


def _default_user(email: str) -> Dict[str, Any]:
    return {
        "email": email.strip().lower(),
        "plan": "free",
        "status": "active",
        "usage_count": 0,
        "usage_limit": FREE_USAGE_LIMIT,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _normalize_user(user: Optional[Dict[str, Any]], email: str) -> Dict[str, Any]:
    normalized = _default_user(email)
    if user:
        normalized.update({k: v for k, v in user.items() if v is not None})
    normalized["email"] = str(normalized.get("email") or email).strip().lower()
    normalized["plan"] = str(normalized.get("plan") or "free").lower()
    normalized["status"] = str(normalized.get("status") or "active").lower()
    normalized["usage_count"] = int(normalized.get("usage_count") or 0)
    normalized["usage_limit"] = int(normalized.get("usage_limit") or (PAID_USAGE_LIMIT if normalized["plan"] != "free" else FREE_USAGE_LIMIT))
    return normalized


# -----------------------------------------------------------------------------
# User persistence
# -----------------------------------------------------------------------------


def get_user(email: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if not email:
        return {}

    sb = _supabase()
    if sb is not None:
        try:
            result = sb.table(USERS_TABLE).select("*").eq("email", email).limit(1).execute()
            rows = getattr(result, "data", None) or []
            if rows:
                return _normalize_user(rows[0], email)
        except Exception as exc:
            logger.exception("Supabase get_user failed for %s: %s", email, exc)
            if not _is_local_dev():
                # Production: surface the error. Do NOT silently fall back to
                # session-local storage, because that lets users reset their
                # quota by opening a new session.
                raise

    local = _local_users()
    if email in local:
        return _normalize_user(local[email], email)
    return {}


def create_user(email: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if not email:
        return {}

    existing = get_user(email)
    if existing:
        return existing

    user = _default_user(email)
    sb = _supabase()
    if sb is not None:
        try:
            result = sb.table(USERS_TABLE).insert(user).execute()
            rows = getattr(result, "data", None) or []
            if rows:
                return _normalize_user(rows[0], email)
        except Exception as exc:
            logger.exception("Supabase create_user failed for %s: %s", email, exc)
            if not _is_local_dev():
                raise

    _local_users()[email] = user
    return user


def _update_user(email: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    email = email.strip().lower()
    fields = {**fields, "updated_at": _now_iso()}
    sb = _supabase()
    if sb is not None:
        try:
            result = sb.table(USERS_TABLE).update(fields).eq("email", email).execute()
            rows = getattr(result, "data", None) or []
            if rows:
                return _normalize_user(rows[0], email)
        except Exception as exc:
            logger.exception("Supabase _update_user failed for %s: %s", email, exc)
            if not _is_local_dev():
                raise

    local = _local_users()
    user = _normalize_user(local.get(email), email)
    user.update(fields)
    local[email] = user
    return _normalize_user(user, email)


def _increment_usage(email: str, user: Dict[str, Any]) -> Dict[str, Any]:
    count = int(user.get("usage_count") or 0) + 1
    return _update_user(email, {"usage_count": count})


# -----------------------------------------------------------------------------
# Usage gate
# -----------------------------------------------------------------------------


def gate_analysis(email: str, consume_usage: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (allowed, message, user). Increments usage when consume_usage=True.

    Free users get FREE_USAGE_LIMIT analyses. Paid/active users are effectively
    unlimited for normal SaaS usage.
    """
    email = email.strip().lower()
    if not email:
        return False, "Enter your email before running analysis.", {}

    user = get_user(email) or create_user(email)
    plan = str(user.get("plan", "free")).lower()
    status = str(user.get("status", "active")).lower()
    usage_count = int(user.get("usage_count") or 0)
    usage_limit = int(user.get("usage_limit") or (PAID_USAGE_LIMIT if plan != "free" else FREE_USAGE_LIMIT))

    paid_active = plan in {"pro", "paid", "premium", "team", "enterprise"} and status in {"active", "paid", "authenticated", "trialing"}
    if paid_active:
        if consume_usage:
            user = _increment_usage(email, user)
        return True, "Paid plan active.", user

    if usage_count >= usage_limit:
        return False, f"Free limit reached ({usage_count}/{usage_limit}). Upgrade to continue.", user

    if consume_usage:
        user = _increment_usage(email, user)
        usage_count = int(user.get("usage_count") or usage_count + 1)
    return True, f"Free usage: {usage_count}/{usage_limit}.", user


# -----------------------------------------------------------------------------
# Razorpay order, pricing, webhook, and subscription cancellation
# -----------------------------------------------------------------------------


def create_razorpay_order(email: str, amount_in_inr: int = 499, notes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    key_id = _safe_secret("RAZORPAY_KEY_ID")
    key_secret = _safe_secret("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        return {"error": "Razorpay credentials are not configured."}

    payload = {
        "amount": int(amount_in_inr * 100),
        "currency": "INR",
        "receipt": f"ba-assistant-{email}-{int(datetime.now().timestamp())}"[:40],
        "payment_capture": 1,
        "notes": {"email": email, **(notes or {})},
    }
    try:
        response = requests.post(
            "https://api.razorpay.com/v1/orders",
            headers=_razorpay_auth_header(),
            data=json.dumps(payload),
            timeout=20,
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def render_pricing(email: str, user: Optional[Dict[str, Any]] = None) -> None:
    user = user or get_user(email) or create_user(email)
    plan = str(user.get("plan", "free")).title()
    usage = int(user.get("usage_count") or 0)
    limit = int(user.get("usage_limit") or FREE_USAGE_LIMIT)

    with st.expander("💳 Pricing", expanded=False):
        st.markdown(
            f"""
            **Current plan:** {plan}  
            **Usage:** {usage}/{limit if limit < PAID_USAGE_LIMIT else 'Unlimited'}

            **Pro — ₹499/month**
            - Unlimited BA reports within fair use
            - PDF export
            - Fintech templates
            - Image/PDF extraction
            - Mermaid diagrams
            """
        )
        key_id = _safe_secret("RAZORPAY_KEY_ID")
        if not key_id:
            st.caption("Add Razorpay credentials to enable live checkout.")
            return

        if st.button("Upgrade with Razorpay", use_container_width=True, key="razorpay_upgrade_btn"):
            order = create_razorpay_order(email=email, amount_in_inr=499, notes={"plan": "pro"})
            if "error" in order:
                st.error(order["error"])
                return
            order_id = order.get("id", "")
            components.html(
                f"""
                <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
                <button id="rzp-button" style="padding:10px 14px;border:0;border-radius:10px;background:#7c3aed;color:white;font-weight:700;width:100%;cursor:pointer;">Pay ₹499</button>
                <script>
                var options = {{
                    "key": "{key_id}",
                    "amount": "49900",
                    "currency": "INR",
                    "name": "BA Assistant",
                    "description": "BA Assistant Pro Monthly",
                    "order_id": "{order_id}",
                    "prefill": {{"email": "{email}"}},
                    "notes": {{"email": "{email}", "plan": "pro"}},
                    "theme": {{"color": "#7c3aed"}}
                }};
                var rzp = new Razorpay(options);
                document.getElementById('rzp-button').onclick = function(e){{ rzp.open(); e.preventDefault(); }}
                </script>
                """,
                height=90,
            )
            st.info("After payment, webhook activation will update your plan automatically. Refresh if needed.")


def verify_razorpay_webhook(payload: bytes | str, signature: str, secret: Optional[str] = None) -> bool:
    """Verify Razorpay webhook HMAC SHA256 signature."""
    webhook_secret = secret or _safe_secret("RAZORPAY_WEBHOOK_SECRET") or _safe_secret("RAZORPAY_KEY_SECRET")
    if not webhook_secret or not signature:
        return False
    body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    digest = hmac.new(webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


def process_razorpay_webhook(payload: bytes | str, signature: str) -> Tuple[bool, str]:
    """Verify and process Razorpay payment/subscription events.

    This helper can be called from a webhook endpoint outside Streamlit, or from a
    lightweight Streamlit-compatible request handler if one is added later.
    """
    if not verify_razorpay_webhook(payload, signature):
        return False, "Invalid Razorpay webhook signature."

    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"Invalid webhook JSON: {exc}"

    event_name = event.get("event", "")
    entity = (
        event.get("payload", {}).get("payment", {}).get("entity")
        or event.get("payload", {}).get("subscription", {}).get("entity")
        or {}
    )
    notes = entity.get("notes") or {}
    email = (notes.get("email") or entity.get("email") or entity.get("contact") or "").strip().lower()
    if not email:
        return False, "Webhook verified, but no customer email was found in notes/entity."

    if event_name in {"payment.captured", "order.paid", "subscription.activated", "subscription.charged"}:
        _update_user(
            email,
            {
                "plan": "pro",
                "status": "active",
                "usage_limit": PAID_USAGE_LIMIT,
                "razorpay_payment_id": entity.get("id"),
                "razorpay_subscription_id": entity.get("subscription_id") or entity.get("id"),
            },
        )
        return True, f"Activated Pro for {email}."

    if event_name in {"payment.failed", "subscription.cancelled", "subscription.halted"}:
        _update_user(email, {"status": "inactive", "plan": "free", "usage_limit": FREE_USAGE_LIMIT})
        return True, f"Marked account inactive/free for {email}."

    return True, f"Webhook verified; ignored event {event_name}."


def cancel_subscription(email: str, subscription_id: Optional[str] = None) -> Tuple[bool, str]:
    email = email.strip().lower()
    user = get_user(email)
    subscription_id = subscription_id or str(user.get("razorpay_subscription_id") or "")
    if not subscription_id:
        _update_user(email, {"plan": "free", "status": "cancelled", "usage_limit": FREE_USAGE_LIMIT})
        return True, "No Razorpay subscription ID found; account marked cancelled locally."

    key_id = _safe_secret("RAZORPAY_KEY_ID")
    key_secret = _safe_secret("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        _update_user(email, {"plan": "free", "status": "cancelled", "usage_limit": FREE_USAGE_LIMIT})
        return True, "Razorpay credentials missing; account marked cancelled locally."

    try:
        response = requests.post(
            f"https://api.razorpay.com/v1/subscriptions/{subscription_id}/cancel",
            headers=_razorpay_auth_header(),
            data=json.dumps({"cancel_at_cycle_end": 1}),
            timeout=20,
        )
        if response.status_code >= 400:
            return False, response.text
        _update_user(email, {"plan": "free", "status": "cancelled", "usage_limit": FREE_USAGE_LIMIT})
        return True, "Subscription cancellation requested."
    except Exception as exc:
        return False, str(exc)
