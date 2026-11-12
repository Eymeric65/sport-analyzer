"""Loader for Concept2 PM5 stroke-data CSV exports (``concept2-result-<id>.csv``).

One row per stroke: elapsed time, distance, pace (s/500 m), watts, cal/h, stroke rate, HR.
Interval workouts are exported in a single file where elapsed time/distance restart at each
interval; they are split into one ``C2Piece`` per interval.
The export has no absolute start time; see ``align.py`` to place it on the H10 timeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = {
    "Number": "stroke",
    "Time (seconds)": "elapsed",
    "Distance (meters)": "distance",
    "Pace (seconds)": "pace",
    "Watts": "watts",
    "Cal/Hr": "cal_hr",
    "Stroke Rate": "spm",
    "Heart Rate": "hr",
}


@dataclass
class C2Piece:
    result_id: int
    path: Path
    interval: int  # 0 for a single piece, 1..n for intervals of an interval workout
    n_intervals: int
    strokes: pd.DataFrame  # columns from COLUMNS values (+ 't' once aligned)
    t_start: float | None = None  # Unix epoch seconds, set by alignment
    align_score: float | None = None  # mean |HR_c2 - HR_h10| at the chosen lag (bpm)

    @property
    def key(self) -> str:
        return f"{self.result_id}" + (f"#{self.interval}" if self.n_intervals > 1 else "")

    @property
    def duration_s(self) -> float:
        return float(self.strokes["elapsed"].iloc[-1])

    @property
    def distance_m(self) -> float:
        return float(self.strokes["distance"].iloc[-1])

    @property
    def has_hr(self) -> bool:
        return self.strokes["hr"].notna().sum() > 5


def load_file(path: Path) -> list[C2Piece]:
    df = pd.read_csv(path).rename(columns=COLUMNS)
    for c in COLUMNS.values():
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df.loc[df["pace"] <= 0, "pace"] = np.nan
    df.loc[(df["spm"] < 8) | (df["spm"] > 40), "spm"] = np.nan  # PM5 glitches (seen: 43, 155 spm)
    rid = int(re.search(r"result-(\d+)", path.stem).group(1))
    # a new interval starts whenever elapsed time goes backwards
    interval_id = (df["elapsed"].diff() < 0).cumsum().to_numpy()
    n = int(interval_id.max()) + 1
    pieces = []
    for i in range(n):
        seg = df[interval_id == i].reset_index(drop=True)
        seg["stroke"] = np.arange(1, len(seg) + 1)
        pieces.append(C2Piece(result_id=rid, path=path, interval=i + 1 if n > 1 else 0, n_intervals=n, strokes=seg))
    return pieces


def load_pieces(session_dir: Path) -> list[C2Piece]:
    """All Concept2 pieces of a session in chronological (result id, interval) order.

    Duplicate downloads of the same result (``...-121025841 (1).csv``) are ignored."""
    by_id: dict[int, Path] = {}
    for p in sorted(Path(session_dir).glob("concept2-result-*.csv"), key=lambda p: len(p.name)):
        rid = int(re.search(r"result-(\d+)", p.stem).group(1))
        by_id.setdefault(rid, p)
    pieces = [pc for rid in sorted(by_id) for pc in load_file(by_id[rid])]
    return pieces
