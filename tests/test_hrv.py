"""DFA-α1 threshold fit on synthetic windows."""

import numpy as np
import pandas as pd

from sport_analyzer.analysis.hrv import hr_at_alpha1


def _windows(t, hr, a1):
    return pd.DataFrame({"t": t, "hr": hr, "alpha1": a1, "rmssd": 20.0, "artefact": 0.0, "reliable": True})


def test_rest_windows_do_not_fake_a_threshold():
    # 10 min of rest (HR 70-85, α1 ~1.2) then easy rowing (HR 120-135, α1 ~0.35, flat)
    t = np.arange(0, 3000, 20.0)
    rest = t < 600
    hr = np.where(rest, 70 + 15 * t / 600, 120 + 15 * (t - 600) / 2400)
    a1 = np.where(rest, 1.2, 0.35 + 0.02 * np.sin(t))
    df = _windows(t, hr, a1)
    assert hr_at_alpha1(df, spans=[(660, 3000)]) is None  # exercise only: no ramp, no threshold


def test_real_ramp_still_found():
    t = np.arange(0, 2400, 20.0)
    hr = 100 + 60 * t / 2400
    a1 = 1.3 - (hr - 100) * (1.3 - 0.4) / 60  # crosses 0.75 at HR 136.7
    df = _windows(t, hr, a1)
    est = hr_at_alpha1(df, spans=[(0, 2400)])
    assert abs(est - 136.7) < 1.5


def test_low_alpha1_at_easy_hr_is_not_graded():
    from sport_analyzer.report.narrative import alpha1_verdict

    assert alpha1_verdict(0.29, 126, 141)[0] == "doesn't match HR"
    assert alpha1_verdict(0.45, 165, 141)[0] == "above VT2"
    assert alpha1_verdict(1.1, 120, 141)[0] == "easy"


def test_fast_hr_rise_and_gaps_give_no_threshold():
    # anchor: HR 125-135, α1 ~1.1, steady; then a 2k: HR jumps 110 -> 180 in 2 min with α1 still ~1
    # (window lag), then sits at 182-190 with α1 ~0.4
    t_a = np.arange(0, 360, 10.0)
    hr_a = 125 + 10 * t_a / 360
    t_r = np.arange(400, 520, 10.0)
    hr_r = 110 + 70 * (t_r - 400) / 120
    t_s = np.arange(520, 960, 10.0)
    hr_s = 182 + 8 * (t_s - 520) / 440
    df = _windows(np.r_[t_a, t_r, t_s], np.r_[hr_a, hr_r, hr_s], np.r_[np.full(len(t_a), 1.1), np.full(len(t_r), 1.0), np.full(len(t_s), 0.4)])
    assert hr_at_alpha1(df) is None  # no steady windows between 135 and 182: not a measurement
