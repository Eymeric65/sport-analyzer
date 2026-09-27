"""Split a session into exercises and rests.

Exercises are the Concept2 pieces (once aligned) plus any rowing detected by the chest
accelerometer that no piece covers (e.g. a piece that was not saved on the PM5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .io.concept2 import C2Piece

MIN_UNLOGGED_S = 30.0


@dataclass
class Segment:
    kind: str  # "exercise" | "rest"
    t0: float
    t1: float
    label: str
    piece: C2Piece | None = None
    target: dict = field(default_factory=dict)  # from claude-notes.md: zone, watts, spm
    metrics: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


def build_segments(pieces: list[C2Piece], activity: pd.DataFrame | None, overrides: dict,
                   rec_t0: float | None, rec_t1: float | None) -> list[Segment]:
    labels = overrides.get("exercises", {}) or {}
    ex: list[Segment] = []
    t_cursor = 0.0
    for i, p in enumerate(pieces):
        if p.t_start is None:  # no H10: lay pieces end-to-end on a relative clock
            p.t_start = t_cursor
            p.strokes["t"] = p.t_start + p.strokes["elapsed"]
        t_cursor = p.t_start + p.duration_s + 60
        o = labels.get(p.key, labels.get(str(p.key), {})) or {}
        if isinstance(o, str):
            o = {"label": o}
        label = o.get("label") or (f"Interval {p.interval}/{p.n_intervals}" if p.n_intervals > 1 else f"Piece {i + 1}")
        ex.append(Segment("exercise", p.t_start, p.t_start + p.duration_s, label, p,
                          target={k: v for k, v in o.items() if k != "label"}))

    if activity is not None and len(activity):
        a = activity.copy()
        a["blk"] = (a["rowing"] != a["rowing"].shift()).cumsum()
        for _, d in a[a["rowing"]].groupby("blk"):
            t0, t1 = float(d["t"].iloc[0]), float(d["t"].iloc[-1])
            if t1 - t0 < MIN_UNLOGGED_S or d["spm"].median() < 14:
                continue
            covered = sum(max(0.0, min(t1, s.t1) - max(t0, s.t0)) for s in ex)
            if covered < 0.3 * (t1 - t0):
                ex.append(Segment("exercise", t0, t1, "Unlogged rowing (ACC only)"))
    ex.sort(key=lambda s: s.t0)

    if rec_t0 is None:  # no H10: rest durations are unknown, do not invent them
        return ex
    out: list[Segment] = []
    prev_end = rec_t0 if rec_t0 is not None else (ex[0].t0 if ex else 0.0)
    for s in ex:
        if s.t0 - prev_end > 15:
            out.append(Segment("rest", prev_end, s.t0, "Rest" if out else "Before start"))
        out.append(s)
        prev_end = s.t1
    if rec_t1 is not None and rec_t1 - prev_end > 15:
        out.append(Segment("rest", prev_end, rec_t1, "After finish"))
    return out
