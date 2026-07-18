# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting for this repository, or email
`tshaik1990@gmail.com` with the subject `BA Assistant security report`.

Include the affected revision, reproduction steps, impact, and any suggested
mitigation. You can expect an acknowledgement within seven days. Do not include
real customer requirements, credentials, payment data, or access tokens.

## Supported versions

Only the latest tagged release and the current `main` branch receive security
updates.

## AI-specific risks

The maintained threat model is in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
Provider keys and Supabase/Razorpay secrets must stay in deployment secrets;
uploaded requirements can contain confidential information and should be
minimized before submission.
