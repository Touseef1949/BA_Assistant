---
title: BA Assistant
emoji: 📋
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: "1.42.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# BA Assistant

[Live app](https://tshaik1990-ba-assistant.hf.space) ·
[Case study](https://touseefshaik.com/apps/ba-assistant.html) ·
[Security policy](SECURITY.md) ·
[Changelog](CHANGELOG.md)

**Maturity:** public flagship, production-oriented, version 0.1.0.

BA Assistant helps Business Analysts and Product Owners turn rough requirements
into a reviewable report containing scope, stakeholders, assumptions,
functional and non-functional requirements, user stories, risks, architecture
notes, roadmap, and Mermaid diagrams. It shortens the first drafting pass; the
output still requires stakeholder and delivery-team review.

## Product flow

1. Sign in with email OTP.
2. Paste requirements or upload an approved PDF/image.
3. Generate a standard report, use Interactive Q&A, or opt into Deep Team mode.
4. Review and export Markdown, TXT, PDF, or Mermaid.

The [public case study](https://touseefshaik.com/apps/ba-assistant.html) is the
visual walkthrough. The sidebar is secondary and holds account status, usage,
history, and advanced settings.

## Architecture

```text
Streamlit UI -> auth and usage gate -> RequirementAnalyzer
             -> DeepSeek worker/coordinator models -> structured report
Optional image upload -> explicit Gemini extraction
PDF upload -> local pdfplumber extraction
Report -> sanitized Mermaid + Markdown/TXT/PDF exports + per-user history
```

Core orchestration lives in `core/`, reusable report/history/error services in
`services/`, the UI flow in `ui/`, and deterministic coverage in `tests/`.
Operational procedures are in [RUNBOOK.md](RUNBOOK.md).

## Supported environment

- Python 3.11 is the deployment and CI version; metadata permits 3.11–3.13.
- A local virtual environment and internet access for dependency installation.
- DeepSeek credentials for analysis. Supabase credentials are required for the
  production authentication path.

## Reproducible quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
export BA_ASSISTANT_LOCAL_DEV=1
export DEEPSEEK_API_KEY="your-key"
streamlit run app.py
```

`requirements.in` declares compatible runtime dependencies;
`requirements.lock` and `requirements-dev.lock` capture exact environments.
`requirements.txt` makes the Hugging Face build consume the runtime lock.
Regenerate with `uv pip compile requirements.in -o requirements.lock`
and `uv pip compile requirements-dev.in -o requirements-dev.lock`.

## Development quality gate

```bash
python -m pip install -r requirements-dev.lock
./scripts/quality.sh
```

The command checks Ruff formatting and linting for core/service boundaries,
targeted mypy checks, Python compilation, the coverage-gated deterministic test
suite, and offline preflight validation. CI runs the same command. Safe health
and load checks never invoke a model:

```bash
python scripts/health_monitor.py --base-url http://localhost:8501 --include-root
locust -f tests/load/locustfile.py --host http://localhost:8501
```

## Model, data, and privacy boundaries

- BA text-analysis paths use `deepseek-v4-flash`.
- Gemini is used only for an explicitly requested image extraction.
- PDF extraction is local, but extracted text can be sent to DeepSeek when the
  user generates a report.
- Report history uses salted hashed filenames; the hosting filesystem is not a
  general-purpose records store.
- Do not submit credentials, regulated personal data, or customer content that
  is not approved for the configured providers.

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for controls and residual risk.

## Secrets

Copy `.streamlit/secrets.toml.example` locally. Production requires
`DEEPSEEK_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, and
`BA_ASSISTANT_AUTH_SECRET`. Razorpay credentials and `GOOGLE_API_KEY` are
optional for their corresponding features. Never commit real secret files.

## Known limitations

- Generated analysis can be incomplete or wrong and is not delivery approval.
- Provider availability, rate limits, and model changes affect latency/output.
- Image extraction sends the uploaded image to the configured Gemini provider.
- Local-dev OTP display must never be enabled in production.
- The lightweight health/load paths prove availability, not model quality.

## Release and deployment

Pull requests must pass the `quality` and `security` jobs. A stable milestone is
tagged with Semantic Versioning and documented in [CHANGELOG.md](CHANGELOG.md).
The tagged GitHub revision is then synchronized to the Hugging Face Space and
verified through `/_stcore/health` and the public app journey.

Contributions are welcome through [CONTRIBUTING.md](CONTRIBUTING.md). This
project is licensed under the [MIT License](LICENSE).
