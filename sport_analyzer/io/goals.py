"""The athlete's objectives: ``goals.md`` in the folder above the session folders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Goals:
    primary: str = "cardio"
    cardio_sessions_per_week: int = 3
    session_minutes: tuple[int, int] = (45, 60)
    gym_sessions_per_week: int = 0
    raw: dict = field(default_factory=dict)
    body_md: str = ""
    path: Path | None = None

    @property
    def headline(self) -> str:
        m = re.search(r"\*\*(.+?)\*\*", self.body_md)
        return m.group(1) if m else f"Primary goal: {self.primary}"


def load_goals(session_dir: Path) -> Goals:
    for d in [session_dir.parent, session_dir.parent.parent]:
        p = d / "goals.md"
        if p.exists():
            text = p.read_text()
            m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
            meta, body = (yaml.safe_load(m.group(1)) or {}, m.group(2)) if m else ({}, text)
            mins = meta.get("session_minutes") or (45, 60)
            return Goals(
                primary=meta.get("primary", "cardio"),
                cardio_sessions_per_week=int(meta.get("cardio_sessions_per_week") or 3),
                session_minutes=tuple(mins) if isinstance(mins, list) else (mins, mins),
                gym_sessions_per_week=int(meta.get("gym_sessions_per_week") or 0),
                raw=meta,
                body_md=body.strip(),
                path=p,
            )
    return Goals()
