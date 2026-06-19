# BA Assistant — Streamlined Production Rebuild Spec

## Goal
Rework BA Assistant so it feels as simple and polished as Stock Research Assistant while preserving its core BA/PO value: paste requirements → generate a structured report. Ship a production-ready v1 that uses DeepSeek Flash for all text-analysis agents, adds a real login layer, improves mobile UX, and hardens the app with tests + deployment docs.

## Current functionality

| Area | Current behavior | Problems |
|---|---|---|
| Information architecture | Three-column feel: heavy sidebar config, large hero, right-side output framing | Feels complex and tool-like rather than productized |
| Auth / access | Email field in sidebar only; no OTP verification; gate is tied to `gate_analysis()` | Not a real login layer; weak trust; poor mobile flow |
| Pricing / usage | Pricing UI lives in sidebar expander; current schema logic mixes `usage_count` / `usage_limit` with a migration using `analyses_used` / `analyses_limit` | Risk of production mismatch and confusing limits |
| Models | Worker = `deepseek-v4-flash`, coordinator/comprehensive/team = `deepseek-v4-pro`, model selector exposed in UI | Too technical and inconsistent with desired default |
| Analysis modes | 7 visible analysis types plus advanced toggles | Too many choices before first action |
| Mobile UX | Sidebar is primary interaction surface; core workflow depends on it | Poor mobile friendliness, cramped first viewport |
| Results UX | Results/Diagrams/History/Prompt Preview tabs always visible | Too much feature surface before the first run |
| History | Session-only history | Not durable per user |
| Testing | No pytest suite | Not production-grade |
| Ops docs | README exists, but no runbook / secrets example / CI | Weak production readiness |

## Proposed new functionality

| Area | Proposed behavior |
|---|---|
| App framing | Rename UI framing to BA Assistant, shorten hero, make the core promise concise and outcome-focused |
| Primary flow | Main-content auth gate → requirements input → single primary CTA (`Generate BA Report`) → results |
| Sidebar | Secondary-only drawer: brand, verified-state badge, history/help, advanced settings |
| Auth | Reuse Stock Research Assistant OTP-style Supabase auth pattern with session-safe persistence rules |
| Usage gate | Require verified email before analysis; track usage against a normalized schema |
| Models | Use `deepseek-v4-flash` for all text analysis agents and coordinator paths; hide model choice from default UI |
| Analysis modes | Simplify default visible choices to a smaller set (recommended: Standard, Interactive Q&A, Deep Team under advanced) |
| Mobile | Use the Streamlit mobile-first overlay drawer pattern; keep all critical actions in main content |
| Results | Keep one primary result surface; reveal diagrams/history/prompt preview progressively |
| History | Persist generated reports per verified email to disk, similar to SRA patterns where practical |
| Production hardening | Add test baseline, smoke command, CI workflow, secrets example, and RUNBOOK |

## UX acceptance criteria

1. Above the fold on desktop and mobile must show:
   - concise value proposition
   - visible sign-in state or auth form
   - requirements input or clear path to it
   - one dominant CTA
2. The app must work on a 390×844 mobile viewport without requiring the sidebar for the core flow.
3. The sidebar must be secondary. Users should be able to sign in, paste requirements, and run a report without opening it.
4. Technical implementation details (API key detected, model IDs, Agno internals) must not dominate the first screen.
5. Secondary actions (`Generate Diagram`, prompt preview, history) must not compete with the primary CTA before first analysis.

## Product / behavior requirements

1. All BA text-analysis agents must use `deepseek-v4-flash`.
2. If image extraction still requires Gemini, keep that path as a non-default helper only; document it as an exception to the “all analysis agents” rule.
3. Require verified email before report generation.
4. The auth persistence logic must be safe on shared hosts: no cross-user file bleed in production.
5. Free-tier usage must be tracked consistently against one schema.
6. Existing report generation, Mermaid generation, PDF export, and interactive Q&A must keep working.

## Production hardening scope for this pass

### Required in this implementation
- pytest baseline covering auth / payment / key app helpers / simplified flow
- smoke validation command (`py_compile`, pytest subset, preflight)
- `.github/workflows/test.yml`
- `.streamlit/secrets.toml.example`
- `RUNBOOK.md`
- README refresh to match the simplified product

### Out of scope for this pass unless easy
- full service extraction of app.py into many modules
- payment provider redesign
- full 90%+ coverage target in one pass
- load-testing harness

## Files expected to change

- `app.py`
- `payment.py`
- `preflight.py`
- `README.md`
- `requirements.txt` (only if needed)
- `RUNBOOK.md` (new)
- `.streamlit/secrets.toml.example` (new)
- `.github/workflows/test.yml` (new)
- `tests/` (new baseline suite)

## Implementation notes

1. Borrow the OTP + auth-state flow from Stock Research Assistant, but adapt copy, quotas, and naming to BA Assistant.
2. Normalize the users-table fields in BA Assistant payment logic. Prefer `analyses_used` / `analyses_limit`, while remaining backward-compatible if old rows have `usage_count` / `usage_limit`.
3. Keep widget keys stable once introduced; avoid brittle Streamlit state churn.
4. Use a mobile-first CSS pass with visible sidebar toggle and main-content auth.
5. Hide advanced settings behind an expander / drawer instead of exposing model and complexity choices by default.

## Verification checklist

- `python3 -m py_compile app.py payment.py preflight.py`
- `pytest -q`
- run preflight quick mode successfully
- inspect desktop UI structure
- inspect mobile UI structure at 390×844
- verify auth-required flow does not show the full product workflow before sign-in
- verify report generation still routes through DeepSeek Flash text agents

## Constraints

- Do not add new paid APIs.
- Do not break existing exports (Markdown/TXT/PDF/Mermaid).
- Keep Streamlit Cloud / website deployability.
- Preserve the business value of BA Assistant while simplifying the UX surface.
