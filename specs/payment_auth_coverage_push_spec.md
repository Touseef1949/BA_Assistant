# BA Assistant — Payment/Auth Coverage Push Spec

Date: 2026-06-20
Project: /Users/touseefshaik/Documents/Pythonproject/Touseef_Project_Work/BA_Assistant/

## Goal
Raise `payment.py` coverage beyond 60% without broadening scope into UI redesign or new payment behavior.

## Current functionality

| Area | Current behavior | Current testing state |
|---|---|---|
| User normalization | Supports legacy `usage_*` and current `analyses_*` fields | Partially covered |
| Email validation | Rejects malformed emails | Covered |
| OTP request | Uses Supabase Auth in production, session-local OTPs in local dev | Largely uncovered |
| OTP verification | Verifies via Supabase in production or local digest/expiry checks in dev | Partially covered |
| Usage gate | Requires verified email, enforces free limit, increments usage, honors paid plans | Partially covered |
| Razorpay order creation | Builds order payload and posts to Razorpay | Uncovered |
| Webhook verification/processing | Verifies signature, activates/deactivates plans, ignores unrelated events | Uncovered |
| Subscription cancellation | Cancels with Razorpay when possible, otherwise downgrades locally | Uncovered |
| Sign-out cleanup | Clears auth session keys | Uncovered |

## Proposed functionality

No product behavior changes. Add tests for existing logic only:

1. OTP request edge cases:
   - invalid email
   - production without Supabase auth support
   - production without config
   - local-dev OTP generation
   - Supabase OTP success path
2. OTP verification edge cases:
   - invalid email / invalid code format
   - production without config
   - missing local OTP
   - expired local OTP
   - wrong local OTP
   - Supabase verify success path
3. Usage gate:
   - invalid email rejection
   - free user with `consume_usage=False`
   - paid active user path
   - free limit reached path
4. Razorpay helpers:
   - missing credentials
   - successful order creation
   - HTTP/API failure return
   - webhook signature valid/invalid
   - webhook JSON error / missing email / activate / deactivate / ignore event
   - cancellation without subscription id
   - cancellation without credentials
   - cancellation HTTP failure
   - cancellation success
5. Session cleanup:
   - `sign_out()` clears all auth-related session keys

## Out of scope

- `render_auth_panel()` widget/UI behavior
- `render_pricing()` widget/UI behavior
- Real live Supabase or Razorpay calls
- End-to-end payment checkout flow
- Deployment, UI polish, or health-monitor cron work

## Files to modify

- `tests/test_payment.py`
- optionally a new focused payment test file only if needed (prefer keeping it in `tests/test_payment.py`)

## Verification

Run in this order:

```bash
/usr/local/bin/python3 -m pytest tests/test_payment.py -q --tb=line --no-cov
/usr/local/bin/python3 -m pytest tests/test_payment.py -q --tb=line --cov=payment --cov-report=term-missing --cov-fail-under=60
/usr/local/bin/python3 -m pytest -q --tb=line
/usr/local/bin/python3 -m py_compile app.py payment.py preflight.py core/analyzer.py core/config.py ui/requirements_flow.py services/history_store.py services/report_utils.py
```

## Constraints

- Preserve existing payment/auth behavior exactly.
- Prefer monkeypatched fakes over broad mocks.
- Do not edit production logic unless tests expose a real bug.
- Keep the change bounded to payment/auth coverage push only.
