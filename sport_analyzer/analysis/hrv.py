"""HRV metrics on corrected RR intervals: RMSSD and DFA-alpha1.

DFA-alpha1 (short-term fractal scaling, box sizes 4-16 beats) drops with exercise intensity:
~1.0 at easy effort, 0.75 around the first ventilatory threshold (VT1 / aerobic threshold) and
0.5 around VT2 (Rogers et al. 2021, Gronwald et al. 2020).  Computed over 2-min windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_S = 120.0
STEP_S = 10.0
MAX_ARTEFACT = 0.02  # windows with more suspicious beats than this are flagged unreliable


def dfa_alpha1(rr_ms: np.ndarray, scales=range(4, 17)) -> float:
    rr = np.asarray(rr_ms, float)
    if len(rr) < 64:
        return np.nan
    y = np.cumsum(rr - rr.mean())
    fl = []
    sc = np.array(list(scales))
    for n in sc:
        m = len(y) // n
        seg = y[: m * n].reshape(m, n)
        x = np.arange(n)
        coef = np.polyfit(x, seg.T, 1)
        trend = np.outer(coef[0], x) + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - trend) ** 2)))
    return float(np.polyfit(np.log(sc), np.log(fl), 1)[0])


def rmssd(rr_ms: np.ndarray) -> float:
    d = np.diff(np.asarray(rr_ms, float))
    return float(np.sqrt(np.mean(d**2))) if len(d) > 2 else np.nan


def suspicious_beats(beat_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flag detected RR intervals deviating > 20 % from their 11-beat local median.

    Returns (end time of each RR, flag). Computed on the *raw* detection, before correction."""
    rr = np.diff(beat_t)
    med = pd.Series(rr).rolling(11, center=True, min_periods=1).median().to_numpy()
    return beat_t[1:], np.abs(rr - med) / med > 0.2


def rolling_hrv(rr_t: np.ndarray, rr_ms: np.ndarray, raw_beat_t: np.ndarray | None = None) -> pd.DataFrame:
    """Rolling DFA-a1 / RMSSD / mean HR. ``rr_ms[i]`` ends at ``rr_t[i+1]``.

    ``raw_beat_t`` (uncorrected detections) gives each window's artefact rate; windows above
    ``MAX_ARTEFACT`` get ``reliable = False`` (alpha1 is very sensitive to artefacts)."""
    end_t = rr_t[1:]
    if raw_beat_t is not None:
        sus_t, sus = suspicious_beats(raw_beat_t)
    rows = []
    for c in np.arange(end_t[0] + WINDOW_S / 2, end_t[-1] - WINDOW_S / 2, STEP_S):
        sel = (end_t >= c - WINDOW_S / 2) & (end_t < c + WINDOW_S / 2)
        r = rr_ms[sel]
        if len(r) < 80:
            continue
        art = float(sus[(sus_t >= c - WINDOW_S / 2) & (sus_t < c + WINDOW_S / 2)].mean()) if raw_beat_t is not None else 0.0
        rows.append((c, 60000 / r.mean(), dfa_alpha1(r), rmssd(r), art))
    df = pd.DataFrame(rows, columns=["t", "hr", "alpha1", "rmssd", "artefact"])
    df["reliable"] = df["artefact"] <= MAX_ARTEFACT
    return df


def hr_at_alpha1(df: pd.DataFrame, target: float = 0.75,
                 spans: list[tuple[float, float]] | None = None) -> float | None:
    """HR where alpha1 crosses ``target``: robust (Theil-Sen) fit of alpha1 on HR over reliable
    windows where alpha1 is in its informative range (0.3-1.5).

    ``spans`` (exercise intervals, in ``t``) restricts the fit to windows centred inside them.
    Mixing rest windows (alpha1 ~1+) with easy rowing (alpha1 driven down by EMG and stroke
    coupling) otherwise fits the rest-to-exercise jump and puts the threshold far too low.
    """
    from scipy.stats import theilslopes

    if spans is not None:
        inside = np.zeros(len(df), bool)
        for a, b in spans:
            inside |= (df["t"].to_numpy() >= a) & (df["t"].to_numpy() <= b)
        df = df[inside]
    d = df[df["reliable"] & (df["alpha1"] > 0.3) & (df["alpha1"] < 1.5)]
    if len(d) < 10 or d["hr"].max() - d["hr"].min() < 20:
        return None
    k, b, _, _ = theilslopes(d["alpha1"], d["hr"])
    if k >= 0:
        return None
    hr = float((target - b) / k)
    # only interpolate: a crossing far above the HRs actually reached is a guess, not a measurement
    if hr > d["hr"].max() + 5 or hr < d["hr"].min():
        return None
    return hr
