# BA Assistant — Production Hardening Phase 1 Spec

## Goal
Take BA Assistant from a rebuilt MVP with a minimal test baseline to a credible production-hardening Phase 1 release by adding reliability, security, CI gates, monitoring scaffolding, and a real test pyramid foundation.

## Why Phase 1
BA Assistant is improved functionally, but it is not yet production-grade. This phase focuses on the highest-value hardening work that materially reduces deployment risk without attempting a full architecture rewrite in one pass.

## Current functionality

| Area | Current state | Gaps |
|---|---|---|
| Tests | 10 pytest tests across 2 files | Not a pyramid; no smoke/integration/regression/AppTest split |
| Coverage enforcement | None | No measurable quality gate |
| CI | Basic GitHub Actions compile + pytest | No coverage threshold, no secrets scan, no stronger fail gates |
| Security scanning | None | No `.gitleaks.toml`, no automated secret scan |
| Pre-push gate | None | Broken code can still be pushed locally |
| Monitoring | None | No health-check script, no structured error logging |
| Load testing | None | No Locust baseline or safe endpoint load check |
| Ops docs | README + RUNBOOK + secrets example now exist | Need docs aligned with new hardening artifacts |
| Architecture | `app.py` monolith, `payment.py`, `preflight.py` | Not yet split into `logic.py` / `services/` / `core/` |

## Proposed new functionality for Phase 1

| Area | Proposed state |
|---|---|
| Test pyramid foundation | Separate smoke, unit, integration, regression, and AppTest coverage for the highest-value flows |
| Coverage gate | Add pytest-cov and enforce a practical minimum threshold in CI |
| CI hardening | Extend GitHub Actions to compile, run tests, enforce coverage, and run secret scanning |
| Security | Add `.gitleaks.toml` and a CI secret-scan step |
| Pre-push verification | Add a local pre-push hook script that blocks pushes on failing tests |
| Monitoring | Add health monitor script and structured JSONL error logging module |
| Load-test scaffold | Add safe Locust checks for page load + health endpoints only |
| Documentation | Update README/RUNBOOK to include new validation, monitoring, and security workflows |

## Non-goals for Phase 1

These are intentionally out of scope for this pass:
- full monolith breakup into `logic.py`, `services/`, `core/`
- 90%+ total coverage in one step
- full production observability platform integration
- deployment automation changes beyond test/security gates
- redesigning pricing/business rules

## Current vs proposed functionality

### Current
- App runs and core BA flow works.
- OTP gate exists.
- DeepSeek Flash is wired for text-analysis agents.
- Basic local validation exists.
- Minimal docs and minimal pytest baseline exist.

### Proposed (Phase 1)
- A meaningful production-quality safety net exists around the app.
- Core auth, gating, history, report generation, and main UI flows are covered by tests.
- CI fails on broken code, weak coverage, or obvious secret leaks.
- Local pushes can be guarded by a fast test gate.
- The app has monitoring and error-log scaffolding suitable for real operations.
- Basic safe load testing exists so we can measure health/page responsiveness without burning LLM credits.

## Files expected to change

### Existing files
- `app.py`
- `payment.py`
- `preflight.py`
- `README.md`
- `RUNBOOK.md`
- `requirements.txt`
- `.gitignore`
- `.github/workflows/test.yml`
- `.streamlit/secrets.toml.example`
- `pytest.ini`

### New files/directories expected
- `.gitleaks.toml`
- `scripts/health_monitor.py`
- `services/error_logging.py`
- `tests/test_smoke.py`
- `tests/test_integration.py`
- `tests/test_regression.py`
- `tests/test_apptest_ui.py`
- `tests/load/locustfile.py`
- `.git/hooks/pre-push` or repo-local equivalent script/check helper if direct hook creation is unsuitable

## Testing requirements

### Minimum suite additions for this phase
1. Smoke tests
   - imports resolve
   - app module loads
   - payment module loads
   - critical config constants exist

2. Unit tests
   - extend current helper/payment tests
   - auth OTP helper behavior
   - history read/write behavior
   - key normalization / gating helpers

3. Integration tests
   - verified auth + gate + usage increment chain
   - history persistence after report save
   - main analysis branching logic with mocks

4. Regression tests
   - locked expectations for simplified mode list
   - DeepSeek Flash enforcement
   - report export helpers

5. AppTest
   - unauthenticated CTA disabled path
   - sign-in flow widgets render correctly
   - requirements field + template selector render correctly
   - authenticated state reveals core workflow correctly

### Coverage gate
- Add `pytest-cov`
- Enforce a realistic first-step threshold in CI
- Recommended Phase 1 threshold: 35% overall minimum, so we create a real gate now without faking maturity

## Monitoring requirements

### Error logging
- Add structured JSONL error logger
- Must be safe for Streamlit app usage
- Should avoid crashing the main app if logging fails

### Health monitor
- Add script that checks the deployed/local app health endpoint and can be cron-run later
- Safe, no LLM calls
- Return non-zero on failure

## Load testing requirements
- Add Locust file
- Only hit safe endpoints:
  - `/`
  - `/_stcore/health`
- Do NOT load test BA analysis endpoints or anything that consumes DeepSeek credits

## Security requirements
- Add `.gitleaks.toml`
- Add CI secret-scan step
- Ensure `.gitignore` covers logs, caches, generated PDFs, and local runtime artifacts

## Implementation constraints
- Do not edit `archive/`
- Do not regress current BA flow
- Keep `streamlit run app.py` working
- Keep image extraction optional and separate from core hardening scope
- Prefer additive hardening over risky large-scale rewrites
- If AppTest proves flaky for a path, test the underlying invariant with mocks/unit tests rather than faking confidence

## Verification checklist
- `python3 -m py_compile app.py payment.py preflight.py`
- `pytest -q`
- `pytest -q --collect-only`
- `python3 preflight.py --quick`
- confirm new CI file includes coverage + security scan
- confirm Locust file exists and targets only safe endpoints
- confirm health monitor script exists
- confirm structured error logging module exists
- confirm docs mention the new production checks

## Deliverable for this phase
A production-hardening Phase 1 repo state that still stops short of full SRA-level production maturity, but materially improves safety, testability, and deploy confidence.
