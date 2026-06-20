# BA Assistant — Phase 2 Architecture Extraction (Slice 2) Spec

## Goal
Continue Phase 2 with the next bounded refactor: move the requirements-input flow helpers out of `app.py` without changing product behavior, so the page shell gets thinner and the later `RequirementAnalyzer` extraction becomes safer.

## Why this slice now
Slice 1 already moved report/history helpers into service modules and reduced `app.py` from ~1,505 lines to ~1,295. The next best slice is the requirements workflow because it is cohesive and user-facing, but still smaller-risk than moving the analyzer/team orchestration immediately.

This slice targets functions that currently sit together and drive one main user journey:
- choose template
- upload/extract document text
- optionally show prompt preview
- run interactive Q&A mode

## Current functionality

| Area | Current state | Pain point |
|---|---|---|
| Template picker | `render_template_selector()` in `app.py` mutates session state and reruns | input-flow logic still trapped in page shell |
| Document upload | `render_upload_area()` in `app.py` handles PDF/image extraction, runtime checks, session-state writes, and reruns | mixes UI, extraction orchestration, and app state transitions |
| Prompt preview | `render_prompt_preview()` in `app.py` builds a preview analyzer or fallback prompt text | preview logic mixed into main page file |
| Interactive Q&A mode | `render_interactive_flow()` in `app.py` drives multi-step Q&A generation + final report generation | one of the heaviest UI flows still inside monolith |
| Main page | `main()` still contains too much requirements-flow orchestration | app shell is still fatter than a production-grade composition layer should be |

## Proposed new functionality for this slice

| Area | Proposed state |
|---|---|
| Requirements flow module | Move template/upload/prompt-preview/interactive-flow helpers into a dedicated module, e.g. `ui/requirements_flow.py` |
| Page shell | `app.py` imports and wires the extracted flow helpers, keeping behavior unchanged |
| Dependency injection | Extracted helpers receive explicit callbacks/dependencies instead of reaching back into hidden globals wherever practical |
| Tests | Expand focused tests around the extracted requirements-flow module and keep AppTest behavior stable |

## Product behavior: current vs proposed

### Current behavior
- User picks a fintech template and requirements text is synced into the textarea.
- User can upload PDF or image and extract text into the requirements area.
- Prompt preview optionally shows the generated prompt or fallback report structure.
- Interactive Q&A mode asks clarifying questions, captures answers, and generates a final enriched report.
- These flows all live in `app.py`.

### Proposed behavior after Slice 2
- Same user-facing flow and widget behavior.
- Same template sync, upload/extract behavior, prompt preview behavior, and interactive Q&A behavior.
- Internally, the requirements workflow logic lives in a dedicated module rather than the main Streamlit page file.

## Scope of this slice

### In scope
- Create a dedicated requirements/UI flow module (preferred: `ui/requirements_flow.py`)
- Move out of `app.py`:
  - `render_template_selector`
  - `render_upload_area`
  - `render_prompt_preview`
  - `render_interactive_flow`
- Keep `reset_interactive()` and `run_paid_gate()` in `app.py` if that produces a cleaner seam; passing them as callbacks is acceptable.
- Update `app.py` to import and call the extracted helpers.
- Add/update focused tests for the extracted flow module.

### Out of scope
- Moving `RequirementAnalyzer` yet
- Refactoring payment/auth module
- Reworking sidebar layout
- Mermaid rendering shell extraction
- Deploy/push work

## Design rules

1. **Keep behavior stable**
   - No UI label changes unless required by the refactor.
   - No payment/auth logic changes.
   - No business-rule changes.

2. **Pass dependencies explicitly**
   The extracted module should receive explicit dependencies/callbacks rather than reaching back into page globals when possible.

   Examples:
   - `extract_pdf_text_fn`
   - `require_runtime_dependencies_fn`
   - `require_api_keys_fn`
   - `analyzer_factory`
   - `reset_interactive_fn`
   - `run_paid_gate_fn`
   - `stream_to_markdown_fn`
   - `save_history_fn`
   - `safe_secret_fn`

3. **Keep Streamlit-bound UI code out of services/**
   Since this slice still uses `st.*`, do **not** place it in `services/`. Prefer a dedicated UI/flow module such as:
   - `ui/requirements_flow.py`

4. **Avoid circular imports**
   If the extracted module needs `AppConfig`, either:
   - import it safely from `app.py` only if no cycle is created, or
   - loosen typing using `Any`, or
   - move `AppConfig` later in another slice only if absolutely necessary.

## Files expected to change

### Existing
- `app.py`
- `tests/test_apptest_ui.py`
- optionally `tests/test_app_helpers.py` or a new focused test file if better scoped

### New
- `ui/requirements_flow.py`
- `ui/__init__.py` if needed
- optionally `tests/test_requirements_flow.py`

## Testing checklist

### Required
- `/usr/local/bin/python3 -m py_compile app.py payment.py preflight.py ui/requirements_flow.py`
- `/usr/local/bin/python3 -m pytest -q`
- Keep AppTest expectations green:
  - template selector still renders
  - requirements textarea still renders
  - authenticated state still reveals core workflow
- Add focused tests where valuable for:
  - template selection sync behavior
  - prompt preview fallback behavior
  - interactive-stage transitions if they can be tested without brittle network paths

## Acceptance criteria
- `app.py` line count drops materially again from the Slice 1 state (~1,295)
- all existing tests pass
- user-facing requirements workflow is unchanged
- new module contains the requirements-flow logic cleanly enough that the next analyzer extraction is easier

## Next slice after this one
If Slice 2 is stable:
1. move `RequirementAnalyzer` into `core/analyzer.py`
2. deepen tests on analyzer/report-generation orchestration
3. then revisit production-grade gaps: monitoring cron, load-test evidence, payment end-to-end verification, and fpdf2 deprecations
