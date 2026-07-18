#!/usr/bin/env python3
"""Safe BA Assistant health monitor.

Checks Streamlit's health endpoint by default and can also verify the root page.
It does not call any LLM-consuming route.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def check_url(url: str, timeout: float) -> Tuple[bool, Dict[str, object]]:
    start = time.perf_counter()
    try:
        request = Request(url, headers={"User-Agent": "ba-assistant-health-monitor/1.0"})
        with urlopen(request, timeout=timeout) as response:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            status = int(getattr(response, "status", 0))
            return 200 <= status < 400, {"url": url, "status": status, "elapsed_ms": elapsed_ms}
    except HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"url": url, "status": exc.code, "elapsed_ms": elapsed_ms, "error": str(exc)}
    except URLError as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"url": url, "status": 0, "elapsed_ms": elapsed_ms, "error": str(exc.reason)}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"url": url, "status": 0, "elapsed_ms": elapsed_ms, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check BA Assistant safe health endpoints.")
    parser.add_argument("--base-url", default="http://localhost:8501", help="Base app URL.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--include-root", action="store_true", help="Also check / without triggering analysis."
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/") + "/"
    endpoints = ["_stcore/health"]
    if args.include_root:
        endpoints.insert(0, "")

    results = []
    ok = True
    for endpoint in endpoints:
        passed, result = check_url(urljoin(base, endpoint), args.timeout)
        results.append(result)
        ok = ok and passed

    print(json.dumps({"ok": ok, "checks": results}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
