"""Per-user report history persistence for BA Assistant."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


SafeSecretFn = Callable[[str, str], str]
LogErrorFn = Callable[[str, Exception, Optional[Dict[str, Any]]], None]


def _truncate(text: str, length: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[: length - 1] + "…" if len(compact) > length else compact


def _history_dir() -> str:
    app_root = os.path.dirname(os.path.dirname(__file__)) or "."
    return os.getenv("BA_ASSISTANT_HISTORY_DIR", os.path.join(app_root, ".ba_history"))


def _history_path(email: str, safe_secret_fn: SafeSecretFn) -> str:
    email = (email or "").strip().lower()
    if not email:
        raise ValueError("Verified email is required for report history.")
    salt = safe_secret_fn("BA_ASSISTANT_AUTH_SECRET", "")
    if not salt:
        if os.environ.get("BA_ASSISTANT_LOCAL_DEV", "").strip().lower() in {"1", "true", "yes"}:
            salt = "local-history"
        else:
            raise RuntimeError("BA_ASSISTANT_AUTH_SECRET must be set before persisting report history.")
    digest = hashlib.sha256(f"{salt}:{email}".encode("utf-8")).hexdigest()
    return os.path.join(_history_dir(), f"{digest}.json")


def load_history(
    email: str,
    safe_secret_fn: SafeSecretFn,
    log_error_fn: Optional[LogErrorFn] = None,
) -> List[Dict[str, Any]]:
    if not email:
        return []
    path = _history_path(email, safe_secret_fn)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)][:20]
    except Exception as exc:
        if log_error_fn:
            log_error_fn("history_load_failed", exc, {"path": path})
        return []
    return []


def save_history(
    project_name: str,
    analysis_type: str,
    result: str,
    current_history: List[Dict[str, Any]],
    safe_secret_fn: SafeSecretFn,
    email: str = "",
) -> List[Dict[str, Any]]:
    item = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "project": project_name or "Untitled Project",
        "type": analysis_type,
        "preview": _truncate(result, 260),
        "result": result,
    }
    history = [item, *current_history][:20]
    if not email:
        return history

    path = _history_path(email, safe_secret_fn)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
    return history
