# BA Assistant — Phase 2 Architecture Extraction (Slice 1) Spec

## Goal
Start Phase 2 with a small, safe refactor that reduces `app.py` monolith pressure without changing product behavior: extract report/history utility logic into dedicated service modules and deepen tests around those extracted paths.

## Why this slice first
Phase 1 hardening is complete: tests, CI gate, error logging, health checks, and load-test scaffold are in place. The clearest remaining engineering risk is structural: `app.py` is still 1,505 lines and mixes UI, agent orchestration, PDF export, Mermaid helpers, and per-user history persistence.

This slice focuses on code that is:
- logically separable
- low-risk to move
- already partly covered by tests
- valuable for raising coverage and making future extraction easier

## Current functionality

| Area | Current state | Pain point |
|---|---|---|
| App structure | `app.py` owns UI + analyzer + PDF + Mermaid + history persistence | 1,505-line monolith is harder to test and change safely |
| PDF export | `_safe_write_line`, `markdown_to_pdf_lines`, `generate_pdf` live in `app.py` | export logic is reusable but trapped inside UI module |
| Mermaid helpers | `sanitize_mermaid_code`, `extract_mermaid_code`, `is_valid_mermaid`, `render_mermaid` live in `app.py` | parsing/rendering concerns mixed with page flow |
| Report history | `_history_dir`, `_history_path`, `load_history`, `save_history` live in `app.py` and directly touch `st.session_state` | persistence logic is tied to Streamlit module-level state |
| Tests | helper tests exist but point at `app.py` functions | coverage can grow faster if utilities live in smaller pure modules |

## Proposed new functionality for this slice

| Area | Proposed state |
|---|---|
| Report export service | Move PDF/markdown export helpers into `services/report_utils.py` |
| History persistence service | Move history path/load/save helpers into `services/history_store.py` with explicit parameters instead of hidden `st.session_state` coupling |
| App wiring | `app.py` imports the extracted helpers and keeps UI behavior unchanged |
| Test coverage | Existing helper/integration tests updated to target the service modules directly; add a few service-level tests where useful |

## Product behavior: current vs proposed

### Current behavior
- Users can generate BA reports and download MD/TXT/PDF.
- Mermaid diagrams are extracted/rendered inline.
- Verified users get report history persisted to disk using hashed filenames.
- All of that logic lives inside `app.py`.

### Proposed behavior after Slice 1
- Same user-facing workflow and outputs.
- Same file naming and persistence behavior.
- Same Mermaid fallback behavior.
- Internally, the reusable non-UI logic lives in service modules instead of the main Streamlit page file.

## Scope of this slice

### In scope
- Create `services/report_utils.py`
- Create `services/history_store.py`
- Move the following functions out of `app.py`:
  - `_safe_write_line`
  - `markdown_to_pdf_lines`
  - `generate_pdf`
  - `sanitize_mermaid_code`
  - `extract_mermaid_code`
  - `is_valid_mermaid`
  - `_history_dir`
  - `_history_path`
  - `load_history`
  - `save_history`
- Keep `render_mermaid()` in `app.py` for now if the HTML component dependency makes extraction noisier than the value gained in this slice.
- Update `app.py` imports and call sites.
- Update tests to target the extracted modules.

### Out of scope
- Splitting `RequirementAnalyzer` yet
- Reworking sidebar/UI functions
- Payment/auth refactor
- Deep Team orchestration changes
- Mobile/UI redesign
- Cron deployment work

## Implementation design

### 1. `services/report_utils.py`
Responsibilities:
- sanitize PDF text lines
- convert markdown to PDF-friendly lines
- generate PDF bytes
- Mermaid code cleanup and extraction helpers

Inputs should be plain Python values only.
No Streamlit dependency in this module.

### 2. `services/history_store.py`
Responsibilities:
- compute safe hashed history path
- load persisted history list
- save persisted history list atomically
- build the newest history item

Design constraints:
- keep production requirement for `BA_ASSISTANT_AUTH_SECRET`
- preserve local-dev fallback only when `BA_ASSISTANT_LOCAL_DEV=1`
- accept `safe_secret_fn` as a dependency instead of importing Streamlit/page code
- accept current in-memory history as an argument, then return the new history list so `app.py` can store it in `st.session_state`

### 3. `app.py`
- Replace local helper definitions with imports from the new service modules
- Keep `render_downloads()` behavior and filenames unchanged
- Keep session-state mutation in `app.py`, but move persistence logic out

## Files expected to change

### Existing
- `app.py`
- `tests/test_app_helpers.py`
- `tests/test_integration.py`

### New
- `services/report_utils.py`
- `services/history_store.py`
- optionally `tests/test_history_store.py` if the refactor benefits from dedicated service tests

## Testing checklist

### Required
- `/usr/local/bin/python3 -m py_compile app.py payment.py preflight.py services/report_utils.py services/history_store.py`
- `pytest -q`
- confirm helper tests still validate:
  - DeepSeek flash model factories
  - Mermaid extraction behavior
  - PDF byte generation
  - history path hashing
  - history persistence through integration flow

### Nice to add in this slice
- direct unit test for `services.history_store.save_history()` returning capped 20-item history
- direct unit test for local-dev fallback behavior when auth secret is missing and local-dev is enabled

## Acceptance criteria
- `app.py` line count drops materially from 1,505
- all existing tests pass
- user-facing download/history behavior is unchanged
- new service modules are import-safe without Streamlit side effects
- extracted code is more reusable and easier to extend in the next slice

## Next slice after this one
If Slice 1 is stable, Slice 2 should extract:
1. template/upload helpers
2. prompt preview + interactive flow helpers
3. eventually `RequirementAnalyzer` into `core/analyzer.py`
