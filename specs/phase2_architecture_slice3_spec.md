# BA Assistant — Phase 2 Architecture Extraction (Slice 3) Spec

## Goal
Move `RequirementAnalyzer` and its analyzer-specific support code out of `app.py` into `core/analyzer.py`, with a small supporting config split, while preserving current BA Assistant behavior and keeping the repo green.

## Why this slice now
Slice 1 extracted report/history helpers.
Slice 2 extracted requirements-flow UI helpers.
The next highest-value monolith reduction is the analyzer itself.

Right now `app.py` still mixes:
- Streamlit page shell
- Agno imports and model factories
- prompt/compliance constants for the analyzer
- the `RequirementAnalyzer` class
- UI/runtime helpers

This slice should leave `app.py` much closer to a page-composition layer.

## Current functionality

| Area | Current state | Pain point |
|---|---|---|
| Analyzer class | `RequirementAnalyzer` still lives in `app.py` | core BA agent orchestration is still coupled to page module |
| Agno dependency surface | `Agent`, `Team`, `AgnoImage`, `OpenAIChat`, `Gemini` imports and related error flags live in `app.py` | app shell knows too much about AI runtime details |
| Model factories | `make_worker_model`, `make_coordinator_model`, `make_vision_model` live in `app.py` | analyzer construction logic is not isolated |
| Analyzer support helpers | `supports_parameter`, `response_content`, analyzer prompt constants live in `app.py` | helper logic is mixed with Streamlit logic |
| Runtime/UI helpers | `event_content`, `stream_to_markdown`, `require_runtime_dependencies`, `require_api_keys`, `bootstrap_environment` live in `app.py` | some should stay in the page shell, some should point to core-owned flags/config |

## Proposed new functionality

| Area | Proposed state |
|---|---|
| Core analyzer module | New `core/analyzer.py` owns Agno imports, model factories, analyzer constants, helper functions, and `RequirementAnalyzer` |
| Core config module | New `core/config.py` owns `safe_secret` and analyzer-related config constants (`TEXT_ANALYSIS_MODEL_ID`, `DEEPSEEK_BASE_URL`, `GOOGLE_OPENAI_BASE_URL`, prompt/report constants) |
| Page shell | `app.py` imports from `core.config` and `core.analyzer`, keeping user-facing behavior unchanged |
| Tests | Existing tests are updated to patch/mock analyzer logic from `core.analyzer` where needed; app-level smoke/regression behavior remains the same |

## Product behavior: current vs proposed

### Current behavior
- Standard mode uses one comprehensive agent.
- Deep Team uses the 5-agent team.
- Interactive Q&A uses the same analyzer pipeline with enriched context.
- Image extraction uses a vision agent.
- Mermaid generation uses the diagram agent.
- These behaviors are implemented from inside `app.py`.

### Proposed behavior after Slice 3
- Exactly the same user-facing and report-generation behavior.
- Internally, analyzer orchestration lives in `core/analyzer.py` instead of the page shell.

## Scope of this slice

### In scope
- Create `core/analyzer.py`
- Create `core/config.py`
- Create `core/__init__.py` if needed
- Move out of `app.py`:
  - Agno import/availability blocks and related error flags
  - `make_worker_model`
  - `make_coordinator_model`
  - `make_vision_model`
  - `supports_parameter`
  - `response_content`
  - `RequirementAnalyzer`
  - analyzer-specific constants required by that class (`TEXT_ANALYSIS_MODEL_ID`, DeepSeek/Google base URLs, `REPORT_STRUCTURE`, `PROMPT_INJECTION_GUARD`)
  - `safe_secret` into `core/config.py`
- Update `app.py` so it imports and reuses these pieces.
- Keep the app runnable and preserve current local uncommitted Slice 1 and Slice 2 work.

### Out of scope
- Moving Streamlit UI code from `app.py` beyond what is already extracted
- Reworking payment/auth module
- Deploy/push work
- fpdf2 warning cleanup
- health-monitor cron wiring

## Design rules

1. **Keep `app.py` behavior stable**
   - No user-facing text changes unless absolutely necessary.
   - No auth/payment/business-rule changes.

2. **Keep UI-only helpers in app.py**
   These should stay page-owned:
   - `event_content`
   - `stream_to_markdown`
   - `parse_questions`
   - `truncate`
   - `render_mermaid`
   - `require_runtime_dependencies` and `require_api_keys` may stay in `app.py`, but should read core-owned flags/config rather than own those definitions.

3. **Avoid circular imports**
   - `core/analyzer.py` must not import `app.py`.
   - `core/config.py` may own `safe_secret` and config constants.
   - `app.py` imports from core; core never imports from app.

4. **Keep app-level compatibility where practical**
   Existing tests and preflight currently import analyzer-related names from `app.py`. It is acceptable for `app.py` to re-export imported names from `core.config` / `core.analyzer` so external behavior stays stable.

## Files expected to change

### Existing
- `app.py`
- `preflight.py`
- `tests/test_app_helpers.py`
- `tests/test_integration.py`
- possibly `tests/test_smoke.py` / `tests/test_regression.py` if import paths or constants need slight updates

### New
- `core/analyzer.py`
- `core/config.py`
- `core/__init__.py` if needed

## Testing checklist

### Required
- `/usr/local/bin/python3 -m py_compile app.py payment.py preflight.py services/report_utils.py services/history_store.py ui/requirements_flow.py core/analyzer.py core/config.py`
- `/usr/local/bin/python3 -m pytest -q tests/test_app_helpers.py tests/test_integration.py`
- `/usr/local/bin/python3 -m pytest -q`
- `python3 preflight.py --quick`

### Test adjustments expected
- `tests/test_app_helpers.py` may need to patch `core.analyzer.OpenAIChat` instead of `app.OpenAIChat`
- `tests/test_integration.py` may need to instantiate `core.analyzer.RequirementAnalyzer` or rely on app re-exports
- Keep smoke/regression expectations around `app.TEXT_ANALYSIS_MODEL_ID` and related app-level public names working if possible

## Acceptance criteria
- `RequirementAnalyzer` no longer lives in `app.py`
- `app.py` line count drops materially again from the Slice 2 state (~1,165)
- all tests pass
- preflight quick pass succeeds
- current BA Assistant behavior is unchanged
- app shell becomes meaningfully closer to composition-only

## Next step after this slice
If Slice 3 is stable:
1. expand coverage gate to include `ui/` and new `core/` modules
2. push total coverage beyond 60%
3. run/document load-test results
4. wire health-monitor cron + alerts
5. run payment end-to-end verification
