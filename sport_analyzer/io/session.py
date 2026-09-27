"""Discover everything recorded for one training day and bundle it in a ``Session``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from .concept2 import C2Piece, load_pieces
from .polar import PolarRecording, load_recordings


@dataclass
class Session:
    path: Path
    day: date
    brief: str
    pieces: list[C2Piece]
    recordings: list[PolarRecording]
    skipped: list[str] = field(default_factory=list)
    notes_md: str = ""  # claude-notes.md body (coach notes written after analysis)
    overrides: dict = field(default_factory=dict)  # claude-notes.md YAML front matter

    @property
    def h10(self) -> PolarRecording | None:
        """The main (longest) H10 recording, if any."""
        return max(self.recordings, key=lambda r: r.duration_s) if self.recordings else None


def _read_notes(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, ""
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if m:
        return yaml.safe_load(m.group(1)) or {}, m.group(2).strip()
    return {}, text.strip()


def load_session(session_dir: Path | str) -> Session:
    session_dir = Path(session_dir).expanduser().resolve()
    day = datetime.strptime(re.search(r"(\d{8})", session_dir.name).group(1), "%Y%m%d").date()
    brief_path = session_dir / "brief.md"
    brief = brief_path.read_text().strip() if brief_path.exists() else ""
    recs, skipped = load_recordings(session_dir)
    overrides, notes = _read_notes(session_dir / "claude-notes.md")
    return Session(
        path=session_dir,
        day=day,
        brief=brief,
        pieces=load_pieces(session_dir),
        recordings=recs,
        skipped=skipped,
        notes_md=notes,
        overrides=overrides,
    )
