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

    assert alpha1_verdict(0.29, 126, 141)[0] == "not readable (rowing)"
    assert alpha1_verdict(0.45, 165, 141)[0] == "above VT2"
    assert alpha1_verdict(1.1, 120, 141)[0] == "easy"
