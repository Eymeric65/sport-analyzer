"""Previous sessions' ``claude-metrics.json`` for trends and training-load context."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path


def _read(d: Path) -> dict | None:
    f = d / "claude-metrics.json"
    if not f.exists():
        return None
    try:
        m = json.loads(f.read_text())
        date.fromisoformat(m["date"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    m["_dir"] = d.name
    m["_note"] = notes_excerpt(d)
    return m


def load_history(session_dir: Path, day: date, lookback_days: int = 120) -> list[dict]:
    """Metrics of earlier sessions (same parent folder), oldest first."""
    out = []
    for d in sorted(session_dir.parent.iterdir()):
        if d == session_dir or not d.is_dir():
            continue
        m = _read(d)
        if m and day - timedelta(days=lookback_days) <= date.fromisoformat(m["date"]) < day:
            out.append(m)
    return sorted(out, key=lambda m: m["date"])


def load_all(root: Path) -> list[dict]:
    """Every analysed session under ``root`` (the folder holding the YYYYMMDD folders), oldest first."""
    out = [m for d in sorted(root.iterdir()) if d.is_dir() and (m := _read(d))]
    return sorted(out, key=lambda m: m["date"])


def load_last_days(history: list[dict], day: date, days: int) -> list[dict]:
    return [m for m in history if date.fromisoformat(m["date"]) > day - timedelta(days=days)]


def notes_excerpt(session_dir: Path, limit: int = 240) -> str:
    """First prose paragraph of the coach notes, as plain text (for timelines and the next brief)."""
    f = session_dir / "claude-notes.md"
    if not f.exists():
        return ""
    text = f.read_text()
    text = re.sub(r"^---\n.*?\n---\n?", "", text, flags=re.S)
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if p.startswith(("- ", "* ")):  # notes are often bullets: take the first one
            p = p.splitlines()[0][2:].strip()
        if not p or p.startswith(("#", "|", ">")):
            continue
        p = re.sub(r"[*_`]+", "", re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", p)).replace("\n", " ")
        return p if len(p) <= limit else p[: limit - 1].rsplit(" ", 1)[0] + "…"
    return ""
