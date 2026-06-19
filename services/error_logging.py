"""Fail-safe structured JSONL error logging for BA Assistant.

The app should never crash because the error logger cannot write. This module
therefore swallows its own filesystem errors while preserving structured events
when the runtime directory is writable.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "ba_assistant_errors.jsonl")


def _log_path() -> str:
    return os.getenv("BA_ASSISTANT_ERROR_LOG", DEFAULT_LOG_PATH)


def _clean_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not context:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in context.items():
        try:
            cleaned[str(key)] = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
        except Exception:
            cleaned[str(key)] = "<unrepresentable>"
    return cleaned


def log_error(event: str, exc: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
    """Append one structured JSONL error event.

    The payload avoids secrets by only writing explicit caller-provided context.
    """
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "error",
            "event": event,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "context": _clean_context(context),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return
