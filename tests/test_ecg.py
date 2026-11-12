"""R-peak detector on synthetic 130 Hz ECG (no personal data)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sport_analyzer.analysis.ecg import detect_beats
from sport_analyzer.analysis.hrv import dfa_alpha1
from sport_analyzer.io.polar import PolarRecording

FS = 130.0


def _wave(t, c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def _recording(sign: float, n_beats: int = 900, seed: int = 0) -> tuple[PolarRecording, np.ndarray]:
    """ECG with a persistent (α1 ~1) RR series around 476 ms (126 bpm).

    sign=+1: classic lead, big positive R.  sign=-1: strap flipped, i.e. a small positive R
    followed by a big negative S, then a positive T wave (the shape seen on 2026-11-09/11).
    """
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 1.2, n_beats))  # random walk -> α1 well above 1
    rr = 0.476 + (walk - walk.mean()) / 1000
    beats = np.cumsum(rr)
    t = np.arange(0, beats[-1] + 1, 1 / FS)
    x = rng.normal(0, 5, len(t))
    for b in beats:
        if sign > 0:
            x += _wave(t, b, 0.012, 500) + _wave(t, b + 0.20, 0.04, 100)
        else:
            x += _wave(t, b - 0.035, 0.010, 100) + _wave(t, b, 0.012, -520) + _wave(t, b + 0.20, 0.04, 90)
    t0 = 1_790_450_000.0
    rec = PolarRecording(name="synthetic", path=None, start_local=datetime(2026, 1, 1), tz_offset_s=0.0)
    rec.ecg = pd.DataFrame({"t": t0 + t, "uv": x})
    th = np.arange(0, t[-1], 1.0)
    rec.hr = pd.DataFrame({"t": t0 + th, "hr": 126, "contact": True})
    return rec, rr


@pytest.mark.parametrize("sign", [1, -1])
def test_beats_follow_the_true_rhythm_in_either_polarity(sign):
    rec, rr_true = _recording(sign)
    b = detect_beats(rec)
    assert abs(len(b.rr_ms) - (len(rr_true) - 1)) <= 3
    assert b.artefact_rate < 0.01
    a_true = np.median([dfa_alpha1(rr_true[i:i + 120] * 1e3) for i in range(0, len(rr_true) - 120, 60)])
    a_det = np.median([dfa_alpha1(b.rr_ms[i:i + 120]) for i in range(0, len(b.rr_ms) - 120, 60)])
    assert a_det == pytest.approx(a_true, abs=0.15)
    assert np.median(b.amp) > 400  # R amplitude (for ECG-derived breathing) uses the dominant lobe
