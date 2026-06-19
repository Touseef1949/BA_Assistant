#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m py_compile app.py payment.py preflight.py
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" preflight.py --quick
