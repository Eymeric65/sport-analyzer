"""Previous sessions' ``claude-metrics.json`` for trends and training-load context."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


def load_history(session_dir: Path, day: date, lookback_days: int = 120) -> list[dict]:
    """Metrics of earlier sessions (same parent folder), oldest first."""
    out = []
    for d in sorted(session_dir.parent.iterdir()):
        f = d / "claude-metrics.json"
        if d == session_dir or not f.exists():
            continue
        try:
            m = json.loads(f.read_text())
            dd = date.fromisoformat(m["date"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if day - timedelta(days=lookback_days) <= dd < day:
            out.append(m)
    return sorted(out, key=lambda m: m["date"])


def load_last_days(history: list[dict], day: date, days: int) -> list[dict]:
    return [m for m in history if date.fromisoformat(m["date"]) > day - timedelta(days=days)]
