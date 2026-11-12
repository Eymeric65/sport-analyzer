"""Cardio-engine metrics: HR-power relationship, HR on/off kinetics, breathing recovery.

These are the numbers that track "more cardio":
* watts at a given HR (the HR-power line), especially at the aerobic threshold (VT1);
* how fast HR comes back down after an effort (recovery time constant tau);
* how fast HR rises into an effort (on-kinetics, faster with a bigger aerobic engine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CHUNK_S = 60.0


def hr_power_points(segments, hr1s: pd.DataFrame, settle_s: float = 120.0) -> pd.DataFrame:
    """60-s (watts, HR) chunks from steady pieces, skipping the first ``settle_s`` (HR lag)."""
    rows = []
    for seg in segments:
        p = seg.piece
        if p is None or p.duration_s < settle_s + CHUNK_S:
            continue
        st = p.strokes.dropna(subset=["watts"])
        for c0 in np.arange(seg.t0 + settle_s, seg.t1 - CHUNK_S + 1, CHUNK_S):
            w = st[(st["t"] >= c0) & (st["t"] < c0 + CHUNK_S)]["watts"]
            h = hr1s[(hr1s["t"] >= c0) & (hr1s["t"] < c0 + CHUNK_S)]["hr"]
            if len(w) >= 5 and len(h) >= 30:
                rows.append({"t": c0 + CHUNK_S / 2, "watts": w.mean(), "hr": h.mean(), "label": seg.label,
                             "minute_in_piece": (c0 - seg.t0) / 60})
    return pd.DataFrame(rows)


def hr_power_fit(pts: pd.DataFrame) -> dict | None:
    """Linear HR = a + b*W over the chunks. Needs a spread of powers to mean anything."""
    if len(pts) < 5 or pts["watts"].max() - pts["watts"].min() < 15:
        return None
    b, a = np.polyfit(pts["watts"], pts["hr"], 1)
    if b <= 0.1:
        return None
    pred = a + b * pts["watts"]
    r2 = 1 - ((pts["hr"] - pred) ** 2).sum() / ((pts["hr"] - pts["hr"].mean()) ** 2).sum()
    if r2 < 0.5:  # a steady piece at one power: HR scatter is noise, the slope means nothing
        return None
    return {"a": float(a), "b": float(b), "r2": float(r2),
            "hr_lo": float(pts["hr"].min()), "hr_hi": float(pts["hr"].max())}


def watts_at_hr(fit: dict | None, hr: float | None, max_extrap: float | None = None) -> float | None:
    """Watts at ``hr`` on the fitted line; with ``max_extrap``, None when ``hr`` is further than
    that many bpm outside the HR range the fit was made on."""
    if not fit or hr is None:
        return None
    if max_extrap is not None and "hr_lo" in fit and not fit["hr_lo"] - max_extrap <= hr <= fit["hr_hi"] + max_extrap:
        return None
    return float((hr - fit["a"]) / fit["b"])


def _expo(t, base, amp, tau):
    return base + amp * np.exp(-t / tau)


def recovery_kinetics(hr1s: pd.DataFrame, t_end: float, t_next: float, max_s: float = 180.0) -> dict | None:
    """Fit HR(t) = base + amp*exp(-t/tau) after an effort; tau in s (smaller = fitter)."""
    win = min(max_s, t_next - t_end)
    if win < 45:
        return None
    d = hr1s[(hr1s["t"] >= t_end) & (hr1s["t"] <= t_end + win)]
    t = d["t"].to_numpy() - t_end
    h = d["hr"].to_numpy()
    try:
        (base, amp, tau), _ = curve_fit(_expo, t, h, p0=(h[-1], h[0] - h[-1], 40), maxfev=5000,
                                        bounds=([40, 0, 5], [200, 150, 600]))
    except (RuntimeError, ValueError):
        return None
    return {"tau_s": float(tau), "hr_peak": float(h[:15].max()), "hr_base": float(base), "window_s": float(win)}


def onset_kinetics(hr1s: pd.DataFrame, t0: float, t1: float) -> dict | None:
    """Time for HR to cover 63 % (one time constant) of its rise during an effort + 30 s after."""
    d = hr1s[(hr1s["t"] >= t0 - 5) & (hr1s["t"] <= t1 + 30)]
    if len(d) < 30:
        return None
    h = d["hr"].to_numpy()
    tt = d["t"].to_numpy() - t0
    start, peak = h[:5].mean(), h.max()
    if peak - start < 10:
        return None
    k = np.argmax(h >= start + 0.63 * (peak - start))
    return {"t63_s": float(tt[k]), "rise_bpm": float(peak - start), "t_peak_s": float(tt[np.argmax(h)])}


def breathing_recovery(breathing: pd.DataFrame, t_end: float, t_next: float) -> pd.DataFrame:
    """Breathing-rate (rest) windows after an effort, indexed by seconds since the effort ended."""
    d = breathing[(breathing["t"] >= t_end + 16) & (breathing["t"] <= t_next - 16) & breathing["rate"].notna()
                  & (breathing["edr_q"] >= 0.35)]
    return pd.DataFrame({"since_s": d["t"] - t_end, "rate": d["rate"], "q": d["edr_q"]})
