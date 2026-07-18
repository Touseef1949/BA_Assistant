# Contributing

1. Create a branch from `main`.
2. Use Python 3.11 and install the locked development environment:
   `python -m pip install -r requirements-dev.lock`.
3. Make a focused change and add tests for externally observable behavior.
4. Run `./scripts/quality.sh`.
5. Open a pull request using the repository template.

Never commit real requirements, reports, email addresses, payment records,
provider keys, or `.streamlit/secrets.toml`. Deterministic tests belong in pull
request CI. Paid model evaluations must remain explicit and separate.
