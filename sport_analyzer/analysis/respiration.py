"""Breathing rate from the H10, without a respiration belt.

Three surrogate signals, resampled at 4 Hz:

* **EDR**: ECG-derived respiration, the beat-to-beat R-peak amplitude.  Chest expansion
  moves the heart relative to the electrodes and changes thoracic impedance, which
  modulates the QRS amplitude.  On this data it is by far the cleanest surrogate.
* **RSA**: respiratory sinus arrhythmia, i.e. RR intervals shorten during inspiration.
  Strong at rest, but vanishes at high HR.
* **ACC**: chest-wall motion seen by the strap accelerometer.  Only usable when still.

The breathing rate is the spectral peak of each signal (6-60 /min) over 32-s windows, and
EDR is the primary estimator.  While rowing, the stroke (about 20 /min) modulates all three
signals mechanically.  Breathing is then usually entrained to the stroke (1 or 2 breaths per
stroke), so it cannot be separated from the stroke artefact in the spectrum.  Those windows
are reported as *coupling* (EDR peak / stroke rate) with low confidence, not as a clean rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .ecg import Beats

FS = 4.0
WIN_S, STEP_S = 32.0, 5.0
F_LO, F_HI = 0.1, 1.0  # 6-60 breaths/min


def _bp(x: np.ndarray, lo=0.08, hi=1.2) -> np.ndarray:
    b, a = signal.butter(2, [lo, hi], btype="band", fs=FS)
    return signal.filtfilt(b, a, signal.detrend(x))


def surrogate_signals(beats: Beats, acc_t: np.ndarray, acc_xyz: np.ndarray) -> tuple[np.ndarray, dict]:
    grid = np.arange(beats.t[0] + 2, beats.t[-1] - 2, 1 / FS)
    amp = pd.Series(beats.amp)
    med = amp.rolling(9, center=True, min_periods=1).median()
    mad = (amp - med).abs().rolling(31, center=True, min_periods=1).median()
    amp = amp.where((amp - med).abs() < 5 * mad + 1e-9, med)  # clip artefact beats
    sig = {
        "edr": _bp(np.interp(grid, beats.t, amp.to_numpy())),
        "rsa": _bp(np.interp(grid, beats.rr_t[1:], beats.rr_ms)),
    }
    acc = np.column_stack([np.interp(grid, acc_t, acc_xyz[:, k]) for k in range(3)])
    sig["acc"] = np.column_stack([_bp(acc[:, k]) for k in range(3)])
    return grid, sig


def _harmonic_share(x: np.ndarray, harm_hz: list[float], tol_hz: float = 0.035) -> float:
    """Share of the 6-60 /min band power that sits on the stroke frequency and its multiples."""
    f, p = signal.welch(x, fs=FS, nperseg=len(x), nfft=2048)
    band = (f >= F_LO) & (f <= F_HI)
    on = np.zeros_like(f, dtype=bool)
    for h in harm_hz:
        on |= np.abs(f - h) < tol_hz
    return float(p[band & on].sum() / p[band].sum())


def _peak(x: np.ndarray, exclude_hz: list[float] | None = None, tol_hz: float = 0.035):
    """Spectral peak (breaths/min) and its quality (share of band power near the peak)."""
    f, p = signal.welch(x, fs=FS, nperseg=len(x), nfft=2048, axis=0)
    if p.ndim == 2:
        p = (p / p[(f > F_LO) & (f < F_HI)].sum(axis=0)).sum(axis=1)
    band = (f >= F_LO) & (f <= F_HI)
    if exclude_hz:
        for h in exclude_hz:
            band &= np.abs(f - h) > tol_hz
    if not band.any():
        return np.nan, 0.0
    fb, pb = f[band], p[band]
    k = int(np.argmax(pb))
    q = pb[np.abs(fb - fb[k]) < 0.03].sum() / pb.sum()
    return float(fb[k] * 60), float(q)


def breathing_windows(grid: np.ndarray, sig: dict, activity: pd.DataFrame) -> pd.DataFrame:
    n, step = int(WIN_S * FS), int(STEP_S * FS)
    rows = []
    for i in range(0, len(grid) - n, step):
        tc = grid[i + n // 2]
        a = activity.iloc[(activity["t"] - tc).abs().argmin()]
        rowing = bool(a["rowing"]) and abs(a["t"] - tc) < 10
        spm = float(a["spm"]) if rowing else np.nan
        row = {"t": tc, "rowing": rowing, "spm": spm}
        for k in ("edr", "rsa", "acc"):
            row[f"{k}_rate"], row[f"{k}_q"] = _peak(sig[k][i : i + n])
        if rowing:
            harm = [m * spm / 60 for m in (1, 2, 3)]
            row["coupling"] = row["edr_rate"] / spm
            row["locked"] = min(abs(row["edr_rate"] - m * spm) for m in (1, 2, 3)) < 2.0
            row["edr_free_rate"], row["edr_free_q"] = _peak(sig["edr"][i : i + n], exclude_hz=harm)
            row["stroke_share"] = _harmonic_share(sig["edr"][i : i + n], harm)
        rows.append(row)
    df = pd.DataFrame(rows)
    # fused rate: EDR where it is trustworthy (rest, good spectral quality)
    good = (~df["rowing"]) & (df["edr_q"] > 0.25)
    df["rate"] = np.where(good, df["edr_rate"], np.nan)
    df["confidence"] = np.where(good, "rest", np.where(df["rowing"], "rowing-coupled", "low"))
    return df


def agreement(df: pd.DataFrame, a: str = "edr", b: str = "acc", tol: float = 3.0) -> float | None:
    """Share of good-quality rest windows where two surrogates agree within ``tol`` /min."""
    d = df[(~df["rowing"]) & (df[f"{a}_q"] > 0.25) & (df[f"{b}_q"] > 0.25)]
    if len(d) < 3:
        return None
    return float((np.abs(d[f"{a}_rate"] - d[f"{b}_rate"]) < tol).mean())
