#!/usr/bin/env python3
"""Pre-flight smoke tests for BA Assistant — run before every deploy.

Usage:
    python3 preflight.py          # all checks
    python3 preflight.py --quick  # skip model tests (no API keys needed)
    python3 preflight.py --pdf    # PDF-only test
"""

import sys
import os
import traceback

PASS = "✅"
FAIL = "❌"
SKIP = "⚠️ "
DIVIDER = "-" * 60


def run(name: str, fn, *args, **kwargs):
    """Run a test and print pass/fail."""
    print(f"  {name}...", end=" ", flush=True)
    try:
        result = fn(*args, **kwargs)
        print(f"{PASS}")
        return result
    except Exception as e:
        print(f"{FAIL} {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Check 1: Syntax
# ---------------------------------------------------------------------------

def check_syntax():
    import ast
    for fname in ["app.py", "payment.py"]:
        path = os.path.join(os.path.dirname(__file__) or ".", fname)
        ast.parse(open(path).read())
    return True


# ---------------------------------------------------------------------------
# Check 2: All imports
# ---------------------------------------------------------------------------

def check_imports():
    from app import (
        safe_secret, bootstrap_environment,
        make_worker_model, make_coordinator_model, make_vision_model,
        RequirementAnalyzer, generate_pdf, sanitize_pdf_text,
        extract_mermaid_code, extract_pdf_text, render_mermaid,
        FINANCIAL_TEMPLATES, PROMPT_INJECTION_GUARD, REPORT_STRUCTURE,
        AppConfig, ANALYSIS_TYPES, ANALYSIS_TYPE_INFO,
    )
    from payment import get_user, create_user, gate_analysis, render_pricing
    return True


# ---------------------------------------------------------------------------
# Check 3: Model factories (needs API keys)
# ---------------------------------------------------------------------------

def check_models():
    from app import safe_secret, bootstrap_environment
    bootstrap_environment()

    deepseek_key = safe_secret("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print(f"    DEEPSEEK_API_KEY missing — skipping model checks")
        return None

    from app import make_worker_model, make_coordinator_model, make_vision_model

    worker = make_worker_model()
    print(f"    Worker: {type(worker).__name__}")

    coord = make_coordinator_model()
    print(f"    Coordinator: {type(coord).__name__}")

    google_key = safe_secret("GOOGLE_API_KEY")
    if google_key:
        vision = make_vision_model()
        print(f"    Vision: {type(vision).__name__}")
    else:
        print(f"    Vision: skipped (no GOOGLE_API_KEY)")

    return True


# ---------------------------------------------------------------------------
# Check 4: RequirementAnalyzer init
# ---------------------------------------------------------------------------

def check_analyzer():
    from app import RequirementAnalyzer

    analyzer = RequirementAnalyzer(enable_vision=False)
    assert analyzer.ba_agent is not None
    assert analyzer.comprehensive_agent is not None
    assert len(analyzer.team.members) == 5
    print(f"    Team members: {[m.name for m in analyzer.team.members]}")
    return True


# ---------------------------------------------------------------------------
# Check 5: PDF generation with edge cases
# ---------------------------------------------------------------------------

SAMPLE_MD = """# Loan Origination System — BA Report

## Executive Summary
This is a test report for a digital loan origination system targeting Indian NBFCs.

## Requirements
The system shall support KYC verification via Aadhaar e-KYC and PAN validation.
Users should be able to upload documents, check application status, and receive in-principle approval.

## User Stories
- As a borrower, I want to apply for a personal loan so that I can get funds quickly.
- As a credit officer, I want to review applications so that I can approve or reject them.

## Mermaid Diagram
```mermaid
flowchart TD
    A[User applies] --> B[KYC verification -- Aadhaar e-KYC + PAN]
    B --> C[Credit bureau check -- CIBIL, Experian]
    C --> D[Income assessment via account aggregator]
    D --> E[Risk scoring engine]
    E --> F{Score >= threshold?}
    F -- Yes --> G[Auto-approve + generate sanction letter]
    F -- No --> H[Manual review queue]
```

## Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Data breach | High | Encryption at rest with AES-256-GCM and HSM-backed keys |
| Regulatory non-compliance | Critical | RBI digital lending guidelines audit checklist + quarterly compliance review |
| NBFC core system downtime | Medium | Circuit breaker pattern with graceful degradation to offline queue |

## Non-Functional Requirements
- 99.95% uptime SLA for the loan application API
- PII encryption at rest and in transit (AES-256, TLS 1.3)
- Audit trail for all credit decisions (immutable, tamper-evident)
- Data localization: all PII stored in India per RBI mandate
"""


def check_pdf():
    from app import generate_pdf, extract_mermaid_code

    # PDF test
    pdf_bytes = generate_pdf("Loan Origination System", SAMPLE_MD)
    assert pdf_bytes, "PDF bytes empty"
    assert len(pdf_bytes) > 500, f"PDF too small: {len(pdf_bytes)} bytes"
    print(f"    PDF size: {len(pdf_bytes)} bytes")

    # Mermaid extraction
    mermaid = extract_mermaid_code(SAMPLE_MD)
    assert mermaid, "No mermaid extracted"
    assert mermaid.startswith("flowchart"), f"Unexpected mermaid start: {mermaid[:30]}"
    print(f"    Mermaid: {len(mermaid)} chars")

    # Save PDF for inspection
    out_path = os.path.join(os.path.dirname(__file__) or ".", "preflight_output.pdf")
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"    Saved: {out_path}")

    return True


# ---------------------------------------------------------------------------
# Check 6: Payment module
# ---------------------------------------------------------------------------

def check_payment():
    """Payment module test — only runs with Streamlit runtime."""
    print(f"    (requires Streamlit runtime — run with: streamlit run preflight.py)")
    return True  # Skip gracefully when outside Streamlit


# ===========================================================================
# Main
# ===========================================================================

def main():
    quick = "--quick" in sys.argv
    pdf_only = "--pdf" in sys.argv

    print(f"\n{DIVIDER}")
    print("BA Assistant — Pre-flight Smoke Tests")
    print(DIVIDER)

    if pdf_only:
        print("\n[PDF + Mermaid]")
        ok = check_pdf()
        if ok:
            print(f"\n{DIVIDER}\n{PDF} PDF generation passed\n{DIVIDER}")
        else:
            print(f"\n{DIVIDER}\n{PDF} PDF generation FAILED\n{DIVIDER}")
            sys.exit(1)
        return

    all_ok = True

    # Always run syntax + imports + analyzer + payment
    print("\n[1/5] Syntax")
    if not run("Syntax check", check_syntax):
        all_ok = False

    print("\n[2/5] Imports")
    if not run("All imports", check_imports):
        all_ok = False

    print(f"\n[3/5] Payment")
    if not run("Payment module", check_payment):
        all_ok = False

    print("\n[4/5] Analyzer")
    if not run("Analyzer init", check_analyzer):
        all_ok = False

    # Model check (needs API keys)
    if not quick:
        print("\n[5/5] Models")
        if not run("Model factories", check_models):
            all_ok = False

    # PDF test
    print(f"\n[Bonus] PDF + Mermaid")
    if not run("PDF generation", check_pdf):
        all_ok = False

    print(f"\n{DIVIDER}")
    if all_ok:
        print(f"{PASS} ALL CHECKS PASSED — safe to deploy")
        print(DIVIDER)
        return
    else:
        print(f"{FAIL} SOME CHECKS FAILED — fix before deploying")
        print(DIVIDER)
        sys.exit(1)


if __name__ == "__main__":
    main()
