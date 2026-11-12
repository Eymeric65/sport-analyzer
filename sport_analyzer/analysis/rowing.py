"""Rowing mechanics from the chest accelerometer, plus per-piece Concept2 metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from ..io.concept2 import C2Piece
from ..io.polar import PolarRecording

FS = 20.0  # working rate for ACC analyses
WIN_S, STEP_S = 12.0, 2.0


def acc_grid(rec: PolarRecording, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Low-passed ACC (N x 3, mG) resampled on a uniform grid."""
    a = rec.acc
    b, aa = signal.butter(4, 0.4 * fs, fs=rec.acc_fs)
    grid = np.arange(a["t"].iloc[0], a["t"].iloc[-1], 1 / fs)
    xyz = np.column_stack([np.interp(grid, a["t"], signal.filtfilt(b, aa, a[k].to_numpy())) for k in "xyz"])
    return grid, xyz


def stroke_activity(grid: np.ndarray, xyz: np.ndarray, fs: float = FS) -> pd.DataFrame:
    """Rolling rowing detection: dominant cadence of the chest motion in 12-48 /min.

    ``rowing`` is true when that peak holds a large share of the motion power."""
    b, a = signal.butter(2, [0.12, 3.0], btype="band", fs=fs)
    m = signal.filtfilt(b, a, xyz, axis=0)
    n, step = int(WIN_S * fs), int(STEP_S * fs)
    rows = []
    for i in range(0, len(grid) - n, step):
        seg = m[i : i + n]
        f, p = signal.welch(seg, fs=fs, nperseg=n, nfft=4096, axis=0)
        p = p.sum(axis=1)
        band = (f >= 0.2) & (f <= 0.8)
        k = np.argmax(p[band])
        fpk = f[band][k]
        near = np.abs(f - fpk) < 0.04
        share = p[near].sum() / p[(f > 0.1) & (f < 3)].sum()
        rms = float(np.sqrt((seg**2).sum(axis=1).mean()))
        rows.append((grid[i + n // 2], fpk * 60, share, rms))
    df = pd.DataFrame(rows, columns=["t", "spm", "share", "rms"])
    df["rowing"] = (df["share"] > 0.25) & (df["rms"] > 40)
    # remove isolated flips
    df["rowing"] = df["rowing"].rolling(5, center=True, min_periods=1).median().astype(bool)
    return df


def stroke_events(grid: np.ndarray, xyz: np.ndarray, t0: float, t1: float, fs: float = FS) -> np.ndarray:
    """Times of individual strokes between t0 and t1 (one per cycle of the main motion axis)."""
    sel = (grid >= t0) & (grid <= t1)
    if sel.sum() < fs * 10:
        return np.array([])
    b, a = signal.butter(2, [0.15, 1.2], btype="band", fs=fs)
    seg = signal.filtfilt(b, a, xyz[sel], axis=0)
    u, s, vt = np.linalg.svd(seg - seg.mean(0), full_matrices=False)
    pc = u[:, 0] * s[0]
    f, p = signal.welch(pc, fs=fs, nperseg=min(len(pc), int(30 * fs)), nfft=4096)
    band = (f > 0.2) & (f < 0.8)
    f0 = f[band][np.argmax(p[band])]
    pk, _ = signal.find_peaks(pc, distance=int(0.6 * fs / f0), prominence=0.3 * np.std(pc))
    return grid[sel][pk]


def piece_metrics(p: C2Piece) -> dict:
    s = p.strokes
    w = s["watts"].dropna()
    ss = s[s["elapsed"] > 10]  # skip the start-up strokes
    ws = ss["watts"].dropna()
    spm = ss["spm"].dropna()
    out = {
        "duration_s": p.duration_s,
        "distance_m": p.distance_m,
        "watts_avg": float(w.mean()) if len(w) else None,
        "watts_steady_avg": float(ws.mean()) if len(ws) else None,
        "watts_cv": float(ws.std() / ws.mean()) if len(ws) > 3 else None,
        "watts_max": float(w.max()) if len(w) else None,
        "pace_500m": 500 * p.duration_s / p.distance_m if p.distance_m else None,
        "spm_avg": float(spm.mean()) if len(spm) else None,
        "spm_sd": float(spm.std()) if len(spm) > 3 else None,
        "strokes": int(len(s)),
        "m_per_stroke": p.distance_m / len(s) if len(s) else None,
    }
    if p.has_hr and len(ws):
        hr = ss["hr"].dropna()
        out["hr_avg_c2"] = float(hr.mean())
        out["w_per_bpm"] = float(ws.mean() / hr.mean())
    return out


def decoupling(p: C2Piece) -> float | None:
    """Pa:HR decoupling (%): drop of watts/HR between first and second half (steady pieces).

    < 5 % = well-paced aerobic effort; > 5-10 % = cardiac drift (too hard / heat / dehydration)."""
    s = p.strokes.dropna(subset=["watts", "hr"])
    s = s[s["elapsed"] > 60]
    if p.duration_s < 600 or len(s) < 40:
        return None
    half = s["elapsed"] < (s["elapsed"].iloc[0] + s["elapsed"].iloc[-1]) / 2
    r1 = s[half]["watts"].mean() / s[half]["hr"].mean()
    r2 = s[~half]["watts"].mean() / s[~half]["hr"].mean()
    return float(100 * (r1 - r2) / r1)
