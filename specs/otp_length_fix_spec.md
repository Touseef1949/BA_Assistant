# BA Assistant — OTP Length Compatibility Fix

Date: 2026-06-20
Project: /Users/touseefshaik/Documents/Pythonproject/Touseef_Project_Work/BA_Assistant/

## Goal
Make the live BA Assistant auth flow compatible with the OTP length actually sent by Supabase so the deployed login can complete end-to-end.

## Current functionality
- Main-page auth flow is now correctly rendered above the requirements form.
- `request_login_otp()` successfully sends an email through Supabase Auth.
- Live verification email currently contains an 8-digit numeric code.
- `verify_login_otp()` only accepts `\d{6}`.
- The UI input for the OTP uses `max_chars=6` and labels the field as a 6-digit code.
- Result: the app sends the OTP but makes successful verification impossible.

## Proposed functionality
- Accept 6 to 8 digit numeric OTPs in `verify_login_otp()`.
- Increase UI OTP input `max_chars` from 6 to 8.
- Update user-facing copy from “6-digit code” to neutral wording that matches Supabase reality.
- Add tests covering 8-digit OTP acceptance while preserving rejection of malformed values.

## Files to modify
- `payment.py`
- `tests/test_payment.py`

## Verification
```bash
/usr/local/bin/python3 -m pytest tests/test_payment.py -q --tb=line --no-cov
/usr/local/bin/python3 -m pytest -q --tb=line
/usr/local/bin/python3 -m py_compile payment.py app.py
```

## Constraints
- Do not change the auth provider.
- Do not change the existing verified-email gate behavior.
- Keep the fix minimal and targeted to OTP compatibility.
