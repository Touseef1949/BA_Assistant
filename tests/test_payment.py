import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import payment


@pytest.fixture(autouse=True)
def clear_payment_state(monkeypatch):
    payment.st.session_state.clear()
    monkeypatch.delenv("BA_ASSISTANT_LOCAL_DEV", raising=False)
    yield
    payment.st.session_state.clear()


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


def test_normalize_user_prefers_analyses_fields():
    user = payment._normalize_user(
        {
            "email": "USER@example.com",
            "plan": "free",
            "usage_count": 7,
            "usage_limit": 8,
            "analyses_used": 3,
            "analyses_limit": 4,
            "verified_at": "2026-01-01T00:00:00+00:00",
        },
        "user@example.com",
    )

    assert user["email"] == "user@example.com"
    assert user["analyses_used"] == 3
    assert user["analyses_limit"] == 4
    assert user["usage_count"] == 3
    assert user["usage_limit"] == 4
    assert user["email_verified"] is True


def test_normalize_user_reads_legacy_usage_fields():
    user = payment._normalize_user(
        {"email": "legacy@example.com", "usage_count": 1, "usage_limit": 2},
        "legacy@example.com",
    )

    assert user["analyses_used"] == 1
    assert user["analyses_limit"] == 2


def test_invalid_email_is_rejected():
    assert payment.is_valid_email("not-an-email") is False
    assert payment.is_valid_email("person@example.com") is True


def test_request_login_otp_rejects_invalid_email():
    ok, message = payment.request_login_otp("bad-email")

    assert ok is False
    assert "valid business email" in message


def test_request_login_otp_requires_supabase_auth_support(monkeypatch):
    monkeypatch.setattr(payment, "_supabase", lambda: SimpleNamespace(auth=object()))

    ok, message = payment.request_login_otp("user@example.com")

    assert ok is False
    assert "not available" in message


def test_request_login_otp_requires_config_in_production(monkeypatch):
    monkeypatch.setattr(payment, "_supabase", lambda: None)

    ok, message = payment.request_login_otp("user@example.com")

    assert ok is False
    assert "not configured" in message


def test_request_login_otp_local_dev_generates_code(monkeypatch):
    otps = {}
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_otps", lambda: otps)
    monkeypatch.setattr(payment, "create_user", lambda email: {"email": email})
    monkeypatch.setattr(payment.secrets, "randbelow", lambda _: 123456)

    ok, message = payment.request_login_otp("user@example.com")

    assert ok is True
    assert message == "Local development code: 123456"
    assert payment.st.session_state["_ba_local_last_otp"] == "123456"
    assert otps["user@example.com"]["digest"] == payment._otp_digest("user@example.com", "123456")


def test_request_login_otp_supabase_success(monkeypatch):
    calls = []

    class FakeAuth:
        def sign_in_with_otp(self, payload):
            calls.append(payload)

    monkeypatch.setattr(payment, "_supabase", lambda: SimpleNamespace(auth=FakeAuth()))
    monkeypatch.setattr(payment, "create_user", lambda email: {"email": email})

    ok, message = payment.request_login_otp("user@example.com")

    assert ok is True
    assert message == "Verification code sent. Check your email."
    assert calls == [{"email": "user@example.com", "options": {"should_create_user": True}}]


def test_verify_login_otp_rejects_invalid_email():
    ok, message, user = payment.verify_login_otp("bad-email", "123456")

    assert ok is False
    assert "valid email" in message
    assert user == {}


def test_verify_login_otp_rejects_invalid_format():
    ok, message, user = payment.verify_login_otp("user@example.com", "abc")

    assert ok is False
    assert "6-digit" in message
    assert user == {}


def test_verify_login_otp_requires_config_in_production(monkeypatch):
    monkeypatch.setattr(payment, "_supabase", lambda: None)

    ok, message, user = payment.verify_login_otp("user@example.com", "123456")

    assert ok is False
    assert "not configured" in message
    assert user == {}


def test_verify_login_otp_requires_existing_local_code(monkeypatch):
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_otps", lambda: {})

    ok, message, user = payment.verify_login_otp("user@example.com", "123456")

    assert ok is False
    assert "Request a new verification code" in message
    assert user == {}


def test_verify_login_otp_rejects_expired_code(monkeypatch):
    email = "user@example.com"
    otps = {
        email: {
            "digest": payment._otp_digest(email, "123456"),
            "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        }
    }
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_otps", lambda: otps)

    ok, message, user = payment.verify_login_otp(email, "123456")

    assert ok is False
    assert "expired" in message
    assert user == {}
    assert email not in otps


def test_verify_login_otp_rejects_wrong_code(monkeypatch):
    email = "user@example.com"
    otps = {
        email: {
            "digest": payment._otp_digest(email, "123456"),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        }
    }
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_otps", lambda: otps)

    ok, message, user = payment.verify_login_otp(email, "654321")

    assert ok is False
    assert "Invalid verification code" in message
    assert user == {}
    assert email in otps


def test_verify_login_otp_supabase_success(monkeypatch):
    calls = []

    class FakeAuth:
        def verify_otp(self, payload):
            calls.append(payload)

    updated = {"email": "user@example.com", "email_verified": True}
    monkeypatch.setattr(payment, "_supabase", lambda: SimpleNamespace(auth=FakeAuth()))
    monkeypatch.setattr(payment, "create_user", lambda email: {"email": email})
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: {**updated, **fields})

    ok, message, user = payment.verify_login_otp("user@example.com", "123456")

    assert ok is True
    assert message == "Signed in."
    assert user["email_verified"] is True
    assert calls == [{"email": "user@example.com", "token": "123456", "type": "email"}]


def test_local_otp_verification_marks_user_verified(monkeypatch):
    email = "otp@example.com"
    otp = "123456"
    users = {}
    otps = {
        email: {
            "digest": payment._otp_digest(email, otp),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        }
    }
    monkeypatch.setenv("BA_ASSISTANT_LOCAL_DEV", "1")
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_users", lambda: users)
    monkeypatch.setattr(payment, "_local_otps", lambda: otps)

    ok, message, user = payment.verify_login_otp(email, otp)

    assert ok is True
    assert message == "Signed in."
    assert user["email_verified"] is True
    assert email not in otps


def test_gate_rejects_invalid_email():
    allowed, message, user = payment.gate_analysis("not-an-email", consume_usage=True)

    assert allowed is False
    assert "verified email" in message
    assert user == {}


def test_gate_requires_verified_email_before_usage_increment(monkeypatch):
    store = {
        "user@example.com": payment._normalize_user(
            {"email": "user@example.com", "analyses_used": 0, "analyses_limit": 2, "email_verified": False},
            "user@example.com",
        )
    }
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_users", lambda: store)

    allowed, message, user = payment.gate_analysis("user@example.com", consume_usage=True)

    assert allowed is False
    assert "Verify your email" in message
    assert user["analyses_used"] == 0


def test_gate_consumes_verified_free_usage(monkeypatch):
    store = {
        "user@example.com": payment._normalize_user(
            {"email": "user@example.com", "analyses_used": 0, "analyses_limit": 2, "email_verified": True},
            "user@example.com",
        )
    }
    monkeypatch.setattr(payment, "_supabase", lambda: None)
    monkeypatch.setattr(payment, "_local_users", lambda: store)

    allowed, message, user = payment.gate_analysis("user@example.com", consume_usage=True)

    assert allowed is True
    assert message == "Free usage: 1/2."
    assert user["analyses_used"] == 1
    assert store["user@example.com"]["analyses_used"] == 1


def test_gate_does_not_consume_when_disabled(monkeypatch):
    user = payment._normalize_user(
        {"email": "user@example.com", "analyses_used": 1, "analyses_limit": 2, "email_verified": True},
        "user@example.com",
    )
    monkeypatch.setattr(payment, "get_user", lambda email: user)
    monkeypatch.setattr(payment, "create_user", lambda email: user)

    allowed, message, result = payment.gate_analysis("user@example.com", consume_usage=False)

    assert allowed is True
    assert message == "Free usage: 1/2."
    assert result["analyses_used"] == 1


def test_gate_allows_paid_active_user(monkeypatch):
    paid_user = payment._normalize_user(
        {"email": "pro@example.com", "plan": "pro", "status": "active", "email_verified": True, "analyses_used": 7},
        "pro@example.com",
    )
    monkeypatch.setattr(payment, "get_user", lambda email: paid_user)
    monkeypatch.setattr(payment, "create_user", lambda email: paid_user)
    monkeypatch.setattr(payment, "_increment_usage", lambda email, user: {**user, "analyses_used": 8})

    allowed, message, user = payment.gate_analysis("pro@example.com", consume_usage=True)

    assert allowed is True
    assert message == "Paid plan active."
    assert user["analyses_used"] == 8


def test_gate_blocks_when_free_limit_reached(monkeypatch):
    user = payment._normalize_user(
        {"email": "user@example.com", "analyses_used": 2, "analyses_limit": 2, "email_verified": True},
        "user@example.com",
    )
    monkeypatch.setattr(payment, "get_user", lambda email: user)
    monkeypatch.setattr(payment, "create_user", lambda email: user)

    allowed, message, result = payment.gate_analysis("user@example.com", consume_usage=True)

    assert allowed is False
    assert "Free limit reached (2/2)" in message
    assert result["analyses_used"] == 2


def test_create_razorpay_order_requires_credentials(monkeypatch):
    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": "")

    result = payment.create_razorpay_order("user@example.com")

    assert result == {"error": "Razorpay credentials are not configured."}


def test_create_razorpay_order_returns_api_error(monkeypatch):
    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": {"RAZORPAY_KEY_ID": "rk_test", "RAZORPAY_KEY_SECRET": "secret"}.get(name, default))
    monkeypatch.setattr(payment.requests, "post", lambda *args, **kwargs: DummyResponse(status_code=500, text="gateway down"))

    result = payment.create_razorpay_order("user@example.com")

    assert result == {"error": "gateway down"}


def test_create_razorpay_order_success(monkeypatch):
    captured = {}

    def fake_post(url, headers, data, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(data)
        captured["timeout"] = timeout
        return DummyResponse(json_data={"id": "order_123", "status": "created"})

    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": {"RAZORPAY_KEY_ID": "rk_test", "RAZORPAY_KEY_SECRET": "secret"}.get(name, default))
    monkeypatch.setattr(payment.requests, "post", fake_post)

    result = payment.create_razorpay_order("user@example.com", amount_in_inr=499, notes={"plan": "pro"})

    assert result == {"id": "order_123", "status": "created"}
    assert captured["url"] == "https://api.razorpay.com/v1/orders"
    assert captured["payload"]["amount"] == 49900
    assert captured["payload"]["notes"] == {"email": "user@example.com", "plan": "pro"}
    assert captured["timeout"] == 20


def test_verify_razorpay_webhook_valid_and_invalid_signature():
    payload = b'{"event":"payment.captured"}'
    secret = "topsecret"
    digest = payment.hmac.new(secret.encode("utf-8"), payload, payment.hashlib.sha256).hexdigest()

    assert payment.verify_razorpay_webhook(payload, digest, secret=secret) is True
    assert payment.verify_razorpay_webhook(payload, "bad-signature", secret=secret) is False


def test_process_razorpay_webhook_rejects_invalid_signature(monkeypatch):
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: False)

    ok, message = payment.process_razorpay_webhook("{}", "sig")

    assert ok is False
    assert message == "Invalid Razorpay webhook signature."


def test_process_razorpay_webhook_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: True)

    ok, message = payment.process_razorpay_webhook("{not-json", "sig")

    assert ok is False
    assert message.startswith("Invalid webhook JSON:")


def test_process_razorpay_webhook_requires_email(monkeypatch):
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: True)

    ok, message = payment.process_razorpay_webhook(json.dumps({"event": "payment.captured", "payload": {"payment": {"entity": {}}}}), "sig")

    assert ok is False
    assert "no customer email" in message


def test_process_razorpay_webhook_activates_pro(monkeypatch):
    updates = []
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: True)
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: updates.append((email, fields)) or {"email": email, **fields})

    payload = {
        "event": "payment.captured",
        "payload": {"payment": {"entity": {"id": "pay_1", "subscription_id": "sub_1", "notes": {"email": "user@example.com"}}}},
    }

    ok, message = payment.process_razorpay_webhook(json.dumps(payload), "sig")

    assert ok is True
    assert message == "Activated Pro for user@example.com."
    assert updates[0][0] == "user@example.com"
    assert updates[0][1]["plan"] == "pro"
    assert updates[0][1]["razorpay_payment_id"] == "pay_1"


def test_process_razorpay_webhook_downgrades_cancelled_user(monkeypatch):
    updates = []
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: True)
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: updates.append((email, fields)) or {"email": email, **fields})

    payload = {
        "event": "subscription.cancelled",
        "payload": {"subscription": {"entity": {"id": "sub_1", "notes": {"email": "user@example.com"}}}},
    }

    ok, message = payment.process_razorpay_webhook(json.dumps(payload), "sig")

    assert ok is True
    assert message == "Marked account inactive/free for user@example.com."
    assert updates[0][1]["plan"] == "free"
    assert updates[0][1]["status"] == "inactive"


def test_process_razorpay_webhook_ignores_unknown_event(monkeypatch):
    monkeypatch.setattr(payment, "verify_razorpay_webhook", lambda payload, signature: True)

    payload = {
        "event": "subscription.pending",
        "payload": {"subscription": {"entity": {"notes": {"email": "user@example.com"}}}},
    }

    ok, message = payment.process_razorpay_webhook(json.dumps(payload), "sig")

    assert ok is True
    assert message == "Webhook verified; ignored event subscription.pending."


def test_cancel_subscription_without_subscription_id_marks_cancelled(monkeypatch):
    updates = []
    monkeypatch.setattr(payment, "get_user", lambda email: {})
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: updates.append((email, fields)) or {"email": email, **fields})

    ok, message = payment.cancel_subscription("user@example.com")

    assert ok is True
    assert "marked cancelled locally" in message
    assert updates[0][1]["status"] == "cancelled"


def test_cancel_subscription_without_credentials_marks_cancelled(monkeypatch):
    updates = []
    monkeypatch.setattr(payment, "get_user", lambda email: {"razorpay_subscription_id": "sub_1"})
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: updates.append((email, fields)) or {"email": email, **fields})
    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": "")

    ok, message = payment.cancel_subscription("user@example.com")

    assert ok is True
    assert "credentials missing" in message
    assert updates[0][1]["plan"] == "free"


def test_cancel_subscription_returns_http_error(monkeypatch):
    monkeypatch.setattr(payment, "get_user", lambda email: {"razorpay_subscription_id": "sub_1"})
    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": {"RAZORPAY_KEY_ID": "rk_test", "RAZORPAY_KEY_SECRET": "secret"}.get(name, default))
    monkeypatch.setattr(payment.requests, "post", lambda *args, **kwargs: DummyResponse(status_code=500, text="boom"))

    ok, message = payment.cancel_subscription("user@example.com")

    assert ok is False
    assert message == "boom"


def test_cancel_subscription_success(monkeypatch):
    updates = []

    def fake_post(url, headers, data, timeout):
        return DummyResponse(status_code=200, json_data={"id": "sub_1", "status": "cancelled"})

    monkeypatch.setattr(payment, "get_user", lambda email: {"razorpay_subscription_id": "sub_1"})
    monkeypatch.setattr(payment, "_update_user", lambda email, fields: updates.append((email, fields)) or {"email": email, **fields})
    monkeypatch.setattr(payment, "_safe_secret", lambda name, default="": {"RAZORPAY_KEY_ID": "rk_test", "RAZORPAY_KEY_SECRET": "secret"}.get(name, default))
    monkeypatch.setattr(payment.requests, "post", fake_post)

    ok, message = payment.cancel_subscription("user@example.com")

    assert ok is True
    assert message == "Subscription cancellation requested."
    assert updates[0][1]["status"] == "cancelled"


def test_sign_out_clears_auth_session_keys():
    payment.st.session_state.update(
        {
            "auth_verified": True,
            "auth_email": "user@example.com",
            "auth_pending_email": "user@example.com",
            "auth_otp_code": "123456",
            "auth_code_sent": True,
            "auth_code_sent_email": "user@example.com",
            "keep_me": "still here",
        }
    )

    payment.sign_out()

    assert "auth_verified" not in payment.st.session_state
    assert "auth_email" not in payment.st.session_state
    assert "auth_pending_email" not in payment.st.session_state
    assert "auth_otp_code" not in payment.st.session_state
    assert "auth_code_sent" not in payment.st.session_state
    assert "auth_code_sent_email" not in payment.st.session_state
    assert payment.st.session_state["keep_me"] == "still here"
