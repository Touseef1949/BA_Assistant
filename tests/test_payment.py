import payment
from datetime import datetime, timedelta, timezone


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


def test_invalid_email_is_rejected():
    assert payment.is_valid_email("not-an-email") is False
    assert payment.is_valid_email("person@example.com") is True


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
