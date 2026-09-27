"""R-peak detection on the H10 ECG (130 Hz, single chest lead).

While rowing, pectoral EMG swamps the ECG and generic detectors miss ~10 % of beats.  We use
an HR-guided detector: band-pass 8-25 Hz energy envelope, adaptive threshold, and a refractory
window of 0.6 x the RR expected from the strap's HR stream.  Remaining missed/extra/ectopic
beats are corrected with the Kubios algorithm (Lipponen & Tarvainen 2019) for HRV use.
"""

from __future__ import annotations

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal

from ..io.polar import PolarRecording


@dataclass
class Beats:
    t: np.ndarray  # R-peak times (Unix s), as detected
    amp: np.ndarray  # R-peak amplitude (uV, 0.5-40 Hz filtered) -> ECG-derived respiration
    rr_t: np.ndarray  # beat times after artefact correction
    rr_ms: np.ndarray  # corrected RR intervals, rr_ms[i] ends at rr_t[i+1]
    artefact_rate: float  # fraction of beats touched by the correction
    fs: float
    ecg_t: np.ndarray  # uniform ECG grid (for snippets)
    ecg: np.ndarray  # 0.5-40 Hz filtered ECG on that grid


def detect_beats(rec: PolarRecording) -> Beats:
    t, x = rec.ecg["t"].to_numpy(), rec.ecg["uv"].to_numpy()
    fs = round(rec.ecg_fs)
    tu = np.arange(t[0], t[-1], 1 / fs)
    xu = np.interp(tu, t, x)

    b, a = signal.butter(3, [8, 25], btype="band", fs=fs)
    env = np.clip(signal.filtfilt(b, a, xu), 0, None) ** 2
    w = int(0.05 * fs)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    thr = pd.Series(env).rolling(int(3 * fs), center=True, min_periods=1).quantile(0.98).to_numpy() * 0.25
    cand, _ = signal.find_peaks(env, height=thr, distance=int(0.25 * fs))

    exp_rr = 60.0 / np.interp(tu, rec.hr["t"].to_numpy(), rec.hr["hr"].clip(lower=30).to_numpy(float))
    keep: list[int] = []
    for c in cand:
        if keep and tu[c] - tu[keep[-1]] < 0.6 * exp_rr[c]:
            if env[c] > env[keep[-1]]:
                keep[-1] = c
        else:
            keep.append(c)

    b2, a2 = signal.butter(2, [0.5, 40], btype="band", fs=fs)
    g = signal.filtfilt(b2, a2, xu)
    h = int(0.04 * fs)
    pk = np.array([k - h + int(np.argmax(g[max(k - h, 0) : k + h + 1])) for k in keep])
    pk = np.unique(pk)

    info, pkc = nk.signal_fixpeaks(pk, sampling_rate=fs, method="kubios", iterative=True, show=False)
    n_fixed = sum(len(info[k]) for k in ("ectopic", "missed", "extra", "longshort"))
    rr_t = tu[pkc]
    return Beats(
        t=tu[pk],
        amp=g[pk],
        rr_t=rr_t,
        rr_ms=np.diff(rr_t) * 1e3,
        artefact_rate=n_fixed / max(len(pk), 1),
        fs=fs,
        ecg_t=tu,
        ecg=g,
    )


def beat_hr_series(beats: Beats, grid: np.ndarray, smooth_beats: int = 5) -> np.ndarray:
    hr = pd.Series(60000.0 / beats.rr_ms).rolling(smooth_beats, center=True, min_periods=1).median()
    return np.interp(grid, beats.rr_t[1:], hr.to_numpy())
