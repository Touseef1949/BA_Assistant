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
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from services.error_logging import log_error

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
OTP_TTL_MINUTES = 10
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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


def _local_otps() -> Dict[str, Dict[str, Any]]:
    if "_ba_local_otps" not in st.session_state:
        st.session_state["_ba_local_otps"] = {}
    return st.session_state["_ba_local_otps"]


def _default_user(email: str) -> Dict[str, Any]:
    limit = FREE_USAGE_LIMIT
    return {
        "email": email.strip().lower(),
        "plan": "free",
        "status": "active",
        "analyses_used": 0,
        "analyses_limit": limit,
        "usage_count": 0,
        "usage_limit": limit,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _normalize_user(user: Optional[Dict[str, Any]], email: str) -> Dict[str, Any]:
    normalized = _default_user(email)
    source = user or {}
    if user:
        normalized.update({k: v for k, v in user.items() if v is not None})
    normalized["email"] = str(normalized.get("email") or email).strip().lower()
    normalized["plan"] = str(normalized.get("plan") or "free").lower()
    normalized["status"] = str(normalized.get("status") or "active").lower()
    old_used = normalized.get("usage_count")
    old_limit = normalized.get("usage_limit")
    default_limit = PAID_USAGE_LIMIT if normalized["plan"] != "free" else FREE_USAGE_LIMIT
    normalized["analyses_used"] = int(source.get("analyses_used") if source.get("analyses_used") is not None else (old_used or 0))
    normalized["analyses_limit"] = int(source.get("analyses_limit") if source.get("analyses_limit") is not None else (old_limit or default_limit))
    normalized["usage_count"] = normalized["analyses_used"]
    normalized["usage_limit"] = normalized["analyses_limit"]
    normalized["email_verified"] = bool(normalized.get("email_verified") or normalized.get("verified_at"))
    return normalized


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match((email or "").strip().lower()))


def _otp_digest(email: str, otp: str) -> str:
    secret = _safe_secret("BA_ASSISTANT_AUTH_SECRET") or _safe_secret("SUPABASE_KEY") or "ba-assistant-local-dev"
    return hmac.new(secret.encode("utf-8"), f"{email}:{otp}".encode("utf-8"), hashlib.sha256).hexdigest()


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
            log_error("supabase_get_user_failed", exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
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
            log_error("supabase_create_user_failed", exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
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
            if "email_verified" in fields and "email_verified" in str(exc):
                compat_fields = {k: v for k, v in fields.items() if k != "email_verified"}
                try:
                    result = sb.table(USERS_TABLE).update(compat_fields).eq("email", email).execute()
                    rows = getattr(result, "data", None) or []
                    if rows:
                        return _normalize_user(rows[0], email)
                except Exception as compat_exc:
                    logger.exception("Supabase _update_user compatibility retry failed for %s: %s", email, compat_exc)
                    log_error("supabase_update_user_failed", compat_exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
                    if not _is_local_dev():
                        raise
            logger.exception("Supabase _update_user failed for %s: %s", email, exc)
            log_error("supabase_update_user_failed", exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
            if not _is_local_dev():
                raise

    local = _local_users()
    user = _normalize_user(local.get(email), email)
    user.update(fields)
    local[email] = user
    return _normalize_user(user, email)


def _increment_usage(email: str, user: Dict[str, Any]) -> Dict[str, Any]:
    count = int(user.get("analyses_used") or user.get("usage_count") or 0) + 1
    return _update_user(email, {"analyses_used": count, "usage_count": count})


def request_login_otp(email: str) -> Tuple[bool, str]:
    """Request an email OTP.

    Production uses Supabase Auth email OTP. Session-local OTPs are available
    only when BA_ASSISTANT_LOCAL_DEV is explicitly enabled.
    """
    email = (email or "").strip().lower()
    if not is_valid_email(email):
        return False, "Enter a valid business email address."

    sb = _supabase()
    if sb is not None:
        try:
            auth = getattr(sb, "auth", None)
            if auth is None or not hasattr(auth, "sign_in_with_otp"):
                return False, "Supabase Auth OTP is not available in this deployment."
            # Match SRA pattern — pass only email, no options
            auth.sign_in_with_otp({"email": email})
            create_user(email)
            return True, "Verification code sent. Check your email."
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)[:200]
            logger.exception("Supabase OTP request failed for %s: %s", email, exc)
            log_error("supabase_otp_request_failed", exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
            if not _is_local_dev():
                # Surface the actual error type so users aren't stuck with a generic message
                if "AuthApiError" in exc_type or "Auth" in exc_type:
                    return False, f"Email verification is being set up. Try again in a moment. ({exc_type})"
                if "timeout" in exc_msg.lower() or "Timeout" in exc_type:
                    return False, "Email service timed out. Please try again."
                return False, f"Could not send verification code. ({exc_type})"
            return False, f"OTP error: {exc_msg}"

    if not _is_local_dev():
        return False, "Email login is not configured. Set Supabase secrets before deployment."

    otp = f"{secrets.randbelow(1_000_000):06d}"
    _local_otps()[email] = {
        "digest": _otp_digest(email, otp),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)).isoformat(),
    }
    create_user(email)
    st.session_state["_ba_local_last_otp"] = otp
    return True, f"Local development code: {otp}"


def verify_login_otp(email: str, otp: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify OTP and return the normalized user."""
    email = (email or "").strip().lower()
    otp = (otp or "").strip()
    if not is_valid_email(email):
        return False, "Enter a valid email address.", {}
    if not re.fullmatch(r"\d{6,8}", otp):
        return False, "Enter the verification code from your email.", {}

    sb = _supabase()
    if sb is not None:
        auth = getattr(sb, "auth", None)
        if auth is None:
            return False, "Supabase Auth is not available.", {}
        # Supabase email template determines OTP type — try every valid type.
        # Also try camelCase verifyOtp (JS-style) as a fallback.
        otp_types = ("email", "signup", "magiclink", "recovery")
        verify_fn = getattr(auth, "verify_otp", None) or getattr(auth, "verifyOtp", None)
        if verify_fn is None:
            return False, "Supabase Auth OTP verification is not available.", {}
        last_exc = None
        for verify_type in otp_types:
            try:
                verify_fn({"email": email, "token": otp, "type": verify_type})
                user = create_user(email)
                user = _update_user(email, {"email_verified": True, "verified_at": _now_iso()})
                return True, "Signed in.", user
            except Exception as exc:
                last_exc = exc
                continue
        # All types failed — surface the last error with Supabase's actual message
        if last_exc is not None:
            exc_type = type(last_exc).__name__
            # AuthApiError/AuthError have .message with Supabase's reason
            supabase_msg = getattr(last_exc, "message", "") or str(last_exc)[:200]
            exc_msg = supabase_msg[:200] if supabase_msg else str(last_exc)[:200]
            logger.exception("Supabase OTP verify failed for %s: %s", email, last_exc)
            log_error("supabase_otp_verify_failed", last_exc, {"email_domain": email.split("@")[-1] if "@" in email else ""})
        else:
            exc_type = "Unknown"
            exc_msg = "No OTP types succeeded"
        if not _is_local_dev():
            return False, f"Invalid or expired verification code. ({exc_type}: {exc_msg})", {}
        return False, f"OTP verify error: {exc_msg}", {}

    if not _is_local_dev():
        return False, "Email login is not configured. Set Supabase secrets before deployment.", {}

    record = _local_otps().get(email)
    if not record:
        return False, "Request a new verification code.", {}
    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        _local_otps().pop(email, None)
        return False, "Verification code expired. Request a new one.", {}
    if not hmac.compare_digest(record["digest"], _otp_digest(email, otp)):
        return False, "Invalid verification code.", {}
    _local_otps().pop(email, None)
    user = create_user(email)
    user = _update_user(email, {"email_verified": True, "verified_at": _now_iso()})
    return True, "Signed in.", user


def render_auth_panel() -> Tuple[bool, str, Dict[str, Any]]:
    """Render main-content OTP auth and return (verified, email, user)."""
    session = st.session_state
    session.setdefault("auth_code_sent", False)
    session.setdefault("auth_code_sent_email", "")
    verified_email = str(session.get("auth_email") or "").strip().lower()
    if session.get("auth_verified") and is_valid_email(verified_email):
        user = get_user(verified_email) or create_user(verified_email)
        user = _normalize_user(user, verified_email)
        plan_label = str(user.get("plan") or "free").capitalize()
        usage_used = int(user.get("analyses_used") or user.get("usage_count") or 0)
        usage_limit = int(user.get("analyses_limit") or user.get("usage_limit") or 2)
        badge_html = (
            '<div class="auth-badge">'
            '  <span class="auth-badge-check">&#10003;</span>'
            f'  <span class="auth-badge-email">{verified_email}</span>'
            f'  <span class="auth-badge-plan">{plan_label} &middot; {usage_used}/{usage_limit} reports</span>'
            '</div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        return True, verified_email, user

    st.markdown("### Sign in to generate reports")
    email = st.text_input("Email address", value=session.get("auth_pending_email", ""), placeholder="you@company.com", key="auth_pending_email")
    clean_email = str(email or "").strip().lower()

    if session.get("auth_code_sent_email") and clean_email != session.get("auth_code_sent_email"):
        session["auth_code_sent"] = False
        session["auth_code_sent_email"] = ""
        session.pop("auth_otp_code", None)

    if st.button("Send code", type="secondary", use_container_width=True, key="auth_send_code"):
        ok, message = request_login_otp(clean_email)
        if ok:
            session["auth_code_sent"] = True
            session["auth_code_sent_email"] = clean_email
            st.success(message)
        else:
            st.error(message)

    if session.get("auth_code_sent") and session.get("auth_code_sent_email") == clean_email:
        st.caption("Enter the verification code from your email to continue.")
        code = st.text_input("Verification code", value="", max_chars=8, key="auth_otp_code")
        if st.button("Verify & continue", type="primary", use_container_width=True, key="auth_verify_code"):
            ok, message, user = verify_login_otp(clean_email, code)
            if ok:
                session["auth_verified"] = True
                session["auth_email"] = clean_email
                session["email"] = clean_email
                session["auth_code_sent"] = False
                session["auth_code_sent_email"] = ""
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    return False, "", {}


def sign_out() -> None:
    for key in ("auth_verified", "auth_email", "auth_pending_email", "auth_otp_code", "auth_code_sent", "auth_code_sent_email"):
        st.session_state.pop(key, None)


# -----------------------------------------------------------------------------
# Usage gate
# -----------------------------------------------------------------------------


def gate_analysis(email: str, consume_usage: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (allowed, message, user). Increments usage when consume_usage=True.

    Free users get FREE_USAGE_LIMIT analyses. Paid/active users are effectively
    unlimited for normal SaaS usage.
    """
    email = email.strip().lower()
    if not is_valid_email(email):
        return False, "Sign in with a verified email before running analysis.", {}

    user = get_user(email) or create_user(email)
    if not bool(user.get("email_verified") or user.get("verified_at")):
        return False, "Verify your email before generating a BA report.", user
    plan = str(user.get("plan", "free")).lower()
    status = str(user.get("status", "active")).lower()
    usage_count = int(user.get("analyses_used") or user.get("usage_count") or 0)
    usage_limit = int(user.get("analyses_limit") or user.get("usage_limit") or (PAID_USAGE_LIMIT if plan != "free" else FREE_USAGE_LIMIT))

    paid_active = plan in {"pro", "paid", "premium", "team", "enterprise"} and status in {"active", "paid", "authenticated", "trialing"}
    if paid_active:
        if consume_usage:
            user = _increment_usage(email, user)
        return True, "Paid plan active.", user

    if usage_count >= usage_limit:
        return False, f"Free limit reached ({usage_count}/{usage_limit}). Upgrade to continue.", user

    if consume_usage:
        user = _increment_usage(email, user)
        usage_count = int(user.get("analyses_used") or user.get("usage_count") or usage_count + 1)
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
    usage = int(user.get("analyses_used") or user.get("usage_count") or 0)
    limit = int(user.get("analyses_limit") or user.get("usage_limit") or FREE_USAGE_LIMIT)

    with st.expander("Pricing and usage", expanded=False):
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
                "analyses_limit": PAID_USAGE_LIMIT,
                "usage_limit": PAID_USAGE_LIMIT,
                "razorpay_payment_id": entity.get("id"),
                "razorpay_subscription_id": entity.get("subscription_id") or entity.get("id"),
            },
        )
        return True, f"Activated Pro for {email}."

    if event_name in {"payment.failed", "subscription.cancelled", "subscription.halted"}:
        _update_user(email, {"status": "inactive", "plan": "free", "analyses_limit": FREE_USAGE_LIMIT, "usage_limit": FREE_USAGE_LIMIT})
        return True, f"Marked account inactive/free for {email}."

    return True, f"Webhook verified; ignored event {event_name}."


def cancel_subscription(email: str, subscription_id: Optional[str] = None) -> Tuple[bool, str]:
    email = email.strip().lower()
    user = get_user(email)
    subscription_id = subscription_id or str(user.get("razorpay_subscription_id") or "")
    if not subscription_id:
        _update_user(email, {"plan": "free", "status": "cancelled", "analyses_limit": FREE_USAGE_LIMIT, "usage_limit": FREE_USAGE_LIMIT})
        return True, "No Razorpay subscription ID found; account marked cancelled locally."

    key_id = _safe_secret("RAZORPAY_KEY_ID")
    key_secret = _safe_secret("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        _update_user(email, {"plan": "free", "status": "cancelled", "analyses_limit": FREE_USAGE_LIMIT, "usage_limit": FREE_USAGE_LIMIT})
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
        _update_user(email, {"plan": "free", "status": "cancelled", "analyses_limit": FREE_USAGE_LIMIT, "usage_limit": FREE_USAGE_LIMIT})
        return True, "Subscription cancellation requested."
    except Exception as exc:
        return False, str(exc)
