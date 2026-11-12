"""Place Concept2 pieces on the H10 timeline.

The PM5 logs the HR it receives from the H10 over Bluetooth, so the per-stroke HR column of a
piece is the same signal as the H10 HR stream, shifted by an unknown start time.  For every
piece (in chronological order) we scan candidate start times and keep the one minimising the
mean absolute HR difference.  Pieces must not overlap and must follow each other.
"""

from __future__ import annotations

import numpy as np

from .io.concept2 import C2Piece
from .io.polar import PolarRecording

STEP_S = 0.25


def align_pieces(pieces: list[C2Piece], rec: PolarRecording) -> list[str]:
    """Set ``t_start``/``align_score`` on each piece and a ``t`` column on its strokes.

    Returns human-readable notes about the alignment quality."""
    notes = []
    hr_t = rec.hr["t"].to_numpy()
    hr_v = rec.hr["hr"].to_numpy(float)
    earliest = hr_t[0]
    for p in pieces:
        if not p.has_hr:
            notes.append(f"C2 {p.key}: no HR column, cannot be aligned")
            continue
        s = p.strokes.dropna(subset=["hr"])
        el, h = s["elapsed"].to_numpy(), s["hr"].to_numpy(float)
        latest = hr_t[-1] - el[-1]
        if latest < earliest:
            notes.append(f"C2 {p.key}: does not fit in the H10 recording")
            continue
        cands = np.arange(earliest, latest, STEP_S)
        # vectorised: for every candidate start, interpolate H10 HR at stroke times
        err = np.empty(len(cands))
        for i in range(0, len(cands), 2000):
            c = cands[i : i + 2000, None] + el[None, :]
            err[i : i + 2000] = np.abs(np.interp(c, hr_t, hr_v) - h[None, :]).mean(axis=1)
        best = int(np.argmin(err))
        p.t_start = float(cands[best])
        p.align_score = float(err[best])
        p.strokes["t"] = p.t_start + p.strokes["elapsed"]
        earliest = p.t_start + el[-1]
        if p.align_score > 3:
            notes.append(f"C2 {p.key}: weak alignment (mean HR error {p.align_score:.1f} bpm)")
    return notes
