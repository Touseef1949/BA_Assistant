# BA Assistant

BA Assistant turns rough requirements into a structured BA/Product Owner report: scope, stakeholders, assumptions, functional and non-functional requirements, user stories, risks, architecture notes, roadmap, and Mermaid diagrams.

The Streamlit app now uses a main-content workflow:

1. Sign in with email OTP.
2. Paste or upload requirements.
3. Click `Generate BA Report`.
4. Export Markdown, TXT, PDF, or Mermaid.

The sidebar is secondary and contains account status, usage, history, and advanced settings.

## Model Policy

All BA text-analysis agents and coordinator paths use `deepseek-v4-flash`.

Optional image extraction uses Gemini only when the user explicitly uploads an image and clicks extract. PDF extraction is local through `pdfplumber`.

## Features

- OTP-style email login through Supabase Auth.
- Free-tier usage tracking with `analyses_used` and `analyses_limit`.
- Standard report generation, Interactive Q&A, and advanced Deep Team mode.
- Markdown, TXT, PDF, and Mermaid exports.
- Per-user report history stored under salted hashed filenames.
- Mobile-friendly main workflow with the sidebar acting as a secondary drawer.
- Pytest pyramid foundation with smoke, unit, integration, regression, and Streamlit AppTest coverage.
- GitHub Actions CI with compile, coverage, and secret-scan gates.
- Structured JSONL error logging, a safe health monitor, and a Locust scaffold that only hits `/` and `/_stcore/health`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BA_ASSISTANT_LOCAL_DEV=1
export DEEPSEEK_API_KEY="your-key"
streamlit run app.py
```

For local development, OTP codes are displayed in the app after `Send code`. Production must use Supabase OTP and must not set `BA_ASSISTANT_LOCAL_DEV=1`.

## Secrets

Use `.streamlit/secrets.toml.example` as the template.

Required for production:

- `DEEPSEEK_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `BA_ASSISTANT_AUTH_SECRET`

Optional:

- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`
- `RAZORPAY_WEBHOOK_SECRET`
- `GOOGLE_API_KEY` for image extraction

## Validation

```bash
/usr/local/bin/python3 -m py_compile app.py payment.py preflight.py
pytest -q
python3 preflight.py --quick
```

`pytest -q` enforces the Phase 1 coverage gate through `pytest-cov`.

For the repo-local pre-push equivalent:

```bash
PYTHON_BIN=/usr/local/bin/python3 ./scripts/pre_push_check.sh
```

To monitor a local or deployed Streamlit app without triggering model usage:

```bash
python3 scripts/health_monitor.py --base-url http://localhost:8501 --include-root
```

To run the safe load scaffold:

```bash
locust -f tests/load/locustfile.py --host http://localhost:8501
```

The Locust file only checks `/` and `/_stcore/health`; do not add LLM-consuming flows to load tests.

See `RUNBOOK.md` for production operations, incident checks, and deployment notes.
