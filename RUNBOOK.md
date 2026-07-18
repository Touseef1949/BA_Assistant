# BA Assistant Runbook

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BA_ASSISTANT_LOCAL_DEV=1
export DEEPSEEK_API_KEY="your-key"
streamlit run app.py
```

In local development, OTP codes are displayed in the app after you click `Send code`. Do not set `BA_ASSISTANT_LOCAL_DEV=1` in production.

## Production Secrets

Copy `.streamlit/secrets.toml.example` to Streamlit Cloud secrets and set:

- `DEEPSEEK_API_KEY` for all BA text-analysis agents and coordinator paths.
- `SUPABASE_URL` and `SUPABASE_KEY` for email OTP login and user records.
- `BA_ASSISTANT_AUTH_SECRET` as a long random value used for safe local history filenames.
- Razorpay keys if checkout is enabled.
- `GOOGLE_API_KEY` only if image extraction is enabled.

## Deploy Checks

Run the local quality gate before every deploy; it matches CI:

```bash
./scripts/quality.sh
```

`quality.sh` checks Ruff formatting and linting for the `core`/`services`/`scripts` boundaries, runs targeted mypy checks, compiles `app.py payment.py preflight.py`, runs the coverage-gated test suite, and runs `preflight.py --quick`. The suite enforces the configured 90% coverage gate, and `preflight.py --quick` skips live model calls. Run full preflight only in an environment with valid provider keys.

CI runs the same `./scripts/quality.sh` in a least-privilege `quality` job and runs Gitleaks with `.gitleaks.toml` in a separate `security` job.

## Auth And Usage

Users must verify email through OTP before any report generation. Usage is tracked with `analyses_used` and `analyses_limit`; legacy `usage_count` and `usage_limit` values are read for backward compatibility.

Production should use Supabase Auth OTP. Session-local OTP exists only when `BA_ASSISTANT_LOCAL_DEV=1` is explicitly set.

## Model Policy

All BA text-analysis paths use `deepseek-v4-flash`, including worker agents, coordinator agent, Standard mode, Interactive Q&A, Deep Team, prompt preview, and Mermaid generation.

The only model exception is optional image extraction, which uses Gemini when the user explicitly uploads an image and clicks extraction. PDF text extraction uses local `pdfplumber`.

## History Storage

Reports are persisted per verified email in `.ba_history/` by default. Filenames are salted hashes and do not include email addresses. Override with `BA_ASSISTANT_HISTORY_DIR` when the host requires a specific writable directory.

## Monitoring

Structured app errors are appended as JSONL to `logs/ba_assistant_errors.jsonl` by default. Override with `BA_ASSISTANT_ERROR_LOG` when the host provides a dedicated writable log path.

Health checks are safe and do not call model providers:

```bash
python3 scripts/health_monitor.py --base-url https://your-app.example.com
```

Add `--include-root` when you also want to verify that the Streamlit root page returns successfully.

## Safe Load Testing

The Locust scaffold only checks `/` and `/_stcore/health`:

```bash
locust -f tests/load/locustfile.py --host https://your-app.example.com
```

Do not add report-generation, image-extraction, or diagram-generation paths to load tests because those can consume provider credits.

## Incident Checks

- If login fails, verify Supabase Auth email OTP is enabled and secrets are present.
- If report generation is blocked, check that the user row has `email_verified=true` or `verified_at`.
- If quotas look wrong, inspect `analyses_used` and `analyses_limit` first, then legacy fields.
- If image extraction fails, confirm `GOOGLE_API_KEY`; normal BA report generation does not require it.
- If errors are reported without UI details, inspect the JSONL error log path configured by `BA_ASSISTANT_ERROR_LOG`.
