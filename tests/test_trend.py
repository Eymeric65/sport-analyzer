"""Trend maths on synthetic data (no personal data in the repo)."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sport_analyzer.analysis import trend as T
from sport_analyzer.io.concept2 import C2Piece
from sport_analyzer.segment import Segment


def _piece(duration_s: float, watts, hr, spm: float = 20.0) -> C2Piece:
    e = np.arange(0, duration_s, 3.0)
    w = watts(e) if callable(watts) else np.full(len(e), float(watts))
    h = hr(e) if callable(hr) else np.full(len(e), float(hr))
    st = pd.DataFrame({"elapsed": e, "distance": e * 2.5, "watts": w, "hr": h, "spm": spm})
    return C2Piece(result_id="1", path=Path("x.csv"), interval=0, n_intervals=1, strokes=st)


def _seg(p, **metrics) -> Segment:
    return Segment(kind="exercise", t0=0.0, t1=p.duration_s, label="piece", piece=p, metrics=dict(metrics))


def test_piece_fitness_projects_along_slope():
    p = _piece(1800, 60, 150)
    f = T.piece_fitness(_seg(p), slope=1.0, hr_rest=70)
    assert abs(f["steady_watts"] - 60) < 1e-6 and abs(f["steady_hr"] - 150) < 1e-6
    assert abs(f["w140_est"] - 50) < 1e-6  # 10 bpm above 140 at 1 bpm/W -> 10 W less
    assert abs(f["ef_net"] - 60 / 80) < 1e-6


def test_piece_fitness_rejects_short_unsteady_or_far():
    assert T.piece_fitness(_seg(_piece(300, 60, 150)), 1.0) is None
    rng = np.random.default_rng(0)
    assert T.piece_fitness(_seg(_piece(1800, lambda e: rng.uniform(10, 150, len(e)), 150)), 1.0) is None
    far = T.piece_fitness(_seg(_piece(1800, 40, 110)), 1.0)
    assert far is not None and "w140_est" not in far  # 30 bpm away: no projection


def test_anchor_detected_at_fixed_watts():
    p = _piece(360, 61, lambda e: 100 + 30 * np.clip(e / 180, 0, 1))
    a = T.anchor_test([_seg(p, hrr60=22.0)], anchor_watts=60)
    assert a and abs(a["hr_anchor"] - 130) < 1e-6 and a["anchor_hrr60"] == 22.0
    assert T.anchor_test([_seg(_piece(360, 90, 140))], anchor_watts=60) is None


def test_kalman_flat_series_has_flat_level_and_shrinking_band():
    d = [date(2026, 1, 1) + timedelta(days=3 * i) for i in range(10)]
    k = T.kalman_level(d, [140.0] * 10, swc=0.03, relative=True)
    assert np.allclose(k["level"], 140.0)
    assert k["sd"].iloc[-1] < 0.03 * 140


def test_real_step_is_called_but_noise_is_not():
    d = [date(2026, 1, 1) + timedelta(days=3 * i) for i in range(16)]
    rng = np.random.default_rng(1)
    flat = 140 + rng.normal(0, 1.0, 16)
    step = np.r_[np.full(8, 145.0), np.full(8, 132.0)] + rng.normal(0, 1.0, 16)
    sess = lambda ys: [{"date": x.isoformat(), "summary": {"hr_anchor": y}, "segments": []} for x, y in zip(d, ys)]
    assert T.trend_summary(sess(flat), d[-1], 4)["arc"]["hr_anchor"]["status"] == "noise"
    assert T.trend_summary(sess(step), d[-1], 4)["arc"]["hr_anchor"]["status"] == "better"  # lower HR = fitter


def test_training_load_ewma_and_weeks():
    day0 = date(2026, 1, 5)  # a Monday
    sess = [{"date": (day0 + timedelta(days=i)).isoformat(), "summary": {"trimp": 50.0, "work_s": 1800, "zones_s": [0, 1800, 0, 0, 0]}}
            for i in (0, 2, 4, 7)]
    ld = T.training_load(sess, day0 + timedelta(days=13), goal_per_week=4)
    assert ld["weekly"][0]["sessions"] == 3 and ld["weekly"][1]["sessions"] == 1
    assert ld["weekly"][0]["easy_share"] == 1.0
    assert 0 < ld["ctl"] < ld["daily"]["atl"].max()  # fatigue reacts faster than fitness
    assert abs(ld["daily"]["ctl"].iloc[0] - 50 * (1 - np.exp(-1 / 42))) < 1e-9


def test_since_last_first_value_is_baseline():
    a = {"date": "2026-01-01", "summary": {"w140_est": 60.0}, "segments": []}
    b = {"date": "2026-01-03", "summary": {"w140_est": 61.0}, "segments": []}
    c = {"date": "2026-01-05", "summary": {"w140_est": 70.0}, "segments": []}
    assert T.since_last([], a)[0]["status"] == "new"
    assert T.since_last([a], b)[0]["status"] == "noise"  # +1 W < 5 %
    assert T.since_last([a, b], c)[0]["status"] == "better"
