"""Run every analysis on a session and collect the results for the report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .align import align_pieces
from .analysis import cardio, ecg, hr as hrmod, hrv, respiration, rowing
from .history import load_history
from .io.goals import Goals, load_goals
from .io.session import Session, load_session
from .segment import Segment, build_segments

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Result:
    session: Session
    profile: hrmod.Profile
    segments: list[Segment]
    notes: list[str] = field(default_factory=list)  # data-quality notes
    summary: dict = field(default_factory=dict)
    hr1s: pd.DataFrame | None = None  # t, hr (1 Hz, from ECG beats or strap)
    activity: pd.DataFrame | None = None
    hrv: pd.DataFrame | None = None
    breathing: pd.DataFrame | None = None
    beats: ecg.Beats | None = None
    resp_grid: np.ndarray | None = None
    resp_sig: dict | None = None
    acc_strokes: dict = field(default_factory=dict)  # piece key -> stroke times
    goals: Goals = field(default_factory=Goals)
    history: list = field(default_factory=list)  # earlier sessions' metrics json
    hr_power: pd.DataFrame | None = None  # 60-s (watts, HR) chunks
    hr_power_fit: dict | None = None
    breath_recovery: dict = field(default_factory=dict)  # segment label -> DataFrame
    recommendations: dict = field(default_factory=dict)

    @property
    def t_origin(self) -> float:
        """Zero of the report's minute axis: H10 start, else the first piece."""
        if self.session.h10 is not None:
            return self.session.h10.t0
        return min((s.t0 for s in self.segments), default=0.0)


def load_profile(path: Path | None = None) -> hrmod.Profile:
    path = path or REPO / "profile.yaml"
    if not path.exists():
        return hrmod.Profile()
    d = yaml.safe_load(path.read_text()) or {}
    return hrmod.Profile(**{k: d.get(k) for k in ("age", "hr_rest", "hr_max", "sex", "weight_kg")})


def _in(t, a, b):
    return (t >= a) & (t < b)


def analyze(session_dir) -> Result:
    s = load_session(session_dir)
    prof = load_profile()
    notes = [f"Skipped Polar recording {x}" for x in s.skipped]
    rec = s.h10
    res = Result(session=s, profile=prof, segments=[], notes=notes)
    res.goals = load_goals(s.path)
    res.history = load_history(s.path, s.day)

    if rec is not None:
        notes += rec.notes
        notes += align_pieces(s.pieces, rec)
        for p in s.pieces:
            if p.align_score is not None:
                notes.append(f"C2 {p.key} placed at +{(p.t_start - rec.t0) / 60:.2f} min "
                             f"(mean HR mismatch {p.align_score:.1f} bpm)")
        b = ecg.detect_beats(rec)
        res.beats = b
        grid1 = np.arange(np.ceil(rec.t0), rec.t1, 1.0)
        hr_ecg = ecg.beat_hr_series(b, grid1)
        strap = np.interp(grid1, rec.hr["t"], rec.hr["hr"])
        valid = _in(grid1, b.t[0] + 3, b.t[-1] - 3)
        # HR metrics use the strap's own HR (validated on-device detection); ECG beats feed HRV/EDR
        res.hr1s = pd.DataFrame({"t": grid1, "hr": strap, "hr_ecg": np.where(valid, hr_ecg, np.nan)})
        mae = float(np.mean(np.abs(hr_ecg[valid] - strap[valid])))
        res.summary["ecg_vs_strap_mae"] = mae
        res.summary["ecg_artefact_rate"] = b.artefact_rate
        notes.append(f"ECG: {len(b.t)} beats detected, {100 * b.artefact_rate:.1f} % corrected "
                     f"(Kubios); ECG-HR vs strap-HR mean abs diff {mae:.1f} bpm")

        g, xyz = rowing.acc_grid(rec)
        res.activity = rowing.stroke_activity(g, xyz)
        for p in s.pieces:
            if p.t_start is not None:
                res.acc_strokes[p.key] = rowing.stroke_events(g, xyz, p.t_start + 10, p.t_start + p.duration_s)
        res.hrv = hrv.rolling_hrv(b.rr_t, b.rr_ms, b.t)
        res.resp_grid, res.resp_sig = respiration.surrogate_signals(b, g, xyz)
        res.breathing = respiration.breathing_windows(res.resp_grid, res.resp_sig, res.activity)
        res.summary["hr_at_alpha1_075"] = hrv.hr_at_alpha1(res.hrv)
        res.summary["hrv_unreliable_share"] = float((~res.hrv["reliable"]).mean())
        res.summary["breath_agree_edr_acc"] = respiration.agreement(res.breathing)
        res.summary["breath_agree_edr_rsa"] = respiration.agreement(res.breathing, "edr", "rsa")

    res.segments = build_segments(s.pieces, res.activity, s.overrides,
                                  rec.t0 if rec else None, rec.t1 if rec else None)
    _segment_metrics(res)
    _cardio_metrics(res)
    _session_metrics(res)
    from .analysis.recommend import recommend

    res.recommendations = recommend(res)
    return res


def _segment_metrics(res: Result) -> None:
    prof = res.profile
    for seg in res.segments:
        m = seg.metrics
        m["duration_s"] = seg.duration_s
        if seg.piece is not None:
            m.update(rowing.piece_metrics(seg.piece))
            m["decoupling_pct"] = rowing.decoupling(seg.piece)
            ev = res.acc_strokes.get(seg.piece.key)
            if ev is not None and len(ev) > 5:
                per = np.diff(ev)
                m["acc_spm"] = float(60 / np.median(per))
                m["acc_rhythm_cv"] = float(np.std(per) / np.mean(per))
        if res.hr1s is not None:
            h = res.hr1s[_in(res.hr1s["t"], seg.t0, seg.t1)]["hr"].to_numpy()
            if len(h) > 5:
                m["hr_avg"], m["hr_max"], m["hr_min"] = float(h.mean()), float(h.max()), float(h.min())
                m["hr_start"], m["hr_end"] = float(h[:10].mean()), float(h[-10:].mean())
                m["zones_s"] = hrmod.time_in_zones(h, 1.0, prof)
                if seg.kind == "exercise":
                    m["hrr60"] = hrmod.hr_recovery(res.hr1s["t"].to_numpy(), res.hr1s["hr"].to_numpy(), seg.t1)
                    if seg.duration_s > 360:  # drift after the first 3 min
                        tt = res.hr1s["t"].to_numpy()
                        sel = _in(tt, seg.t0 + 180, seg.t1)
                        k = np.polyfit((tt[sel] - seg.t0) / 60, res.hr1s["hr"].to_numpy()[sel], 1)[0]
                        m["hr_drift_bpm_per_min"] = float(k)
        if res.hrv is not None:
            d = res.hrv[_in(res.hrv["t"], seg.t0 + 60, seg.t1 - 30) & res.hrv["reliable"]]
            if len(d):
                m["alpha1"] = float(d["alpha1"].median())
        if res.breathing is not None:
            d = res.breathing[_in(res.breathing["t"], seg.t0 + 16, seg.t1 - 16)]
            if len(d):
                r = d["rate"].dropna()
                if len(r):
                    m["breath_rate"] = float(r.median())
                row = d[d["rowing"]]
                if seg.kind == "exercise" and len(row) >= 3:
                    c = np.round(row["coupling"])
                    m["breath_coupling_1"] = float((c == 1).mean())
                    m["breath_coupling_2"] = float((c == 2).mean())
                    m["edr_stroke_share"] = float(row["stroke_share"].median())


def _cardio_metrics(res: Result) -> None:
    if res.hr1s is None:
        return
    sm = res.summary
    pts = cardio.hr_power_points(res.segments, res.hr1s)
    res.hr_power = pts
    res.hr_power_fit = fit = cardio.hr_power_fit(pts)
    if fit:
        sm["hr_power_slope"] = fit["b"]
        sm["hr_power_r2"] = fit["r2"]
        # progress markers: no more than 5 bpm of extrapolation beyond today's HR range
        sm["watts_at_hr140"] = cardio.watts_at_hr(fit, 140, max_extrap=5)
        sm["watts_at_vt1"] = cardio.watts_at_hr(fit, sm.get("hr_at_alpha1_075"), max_extrap=5)
    segs = res.segments
    for i, seg in enumerate(segs):
        if seg.kind != "exercise":
            continue
        nxt = next((x.t0 for x in segs[i + 1:] if x.kind == "exercise"), res.hr1s["t"].iloc[-1])
        rk = cardio.recovery_kinetics(res.hr1s, seg.t1, nxt)
        # tau only means something after a real effort with >= 90 s of rest to fit
        if rk and rk["window_s"] >= 90 and rk["hr_peak"] >= 0.8 * res.profile.hr_max_est:
            seg.metrics["hr_recovery_tau_s"] = rk["tau_s"]
        ok = cardio.onset_kinetics(res.hr1s, seg.t0, seg.t1)
        if ok and seg.duration_s <= 180:
            seg.metrics["hr_t63_s"] = ok["t63_s"]
            seg.metrics["hr_rise_bpm"] = ok["rise_bpm"]
        if res.breathing is not None:
            br = cardio.breathing_recovery(res.breathing, seg.t1, nxt)
            if len(br) >= 3:
                res.breath_recovery[seg.label] = br


def vo2max_from_2k(seconds: float, weight_kg: float, trained: bool = False) -> float:
    """Concept2's published 2k -> VO2max formula (men), in ml/kg/min.

    Y (L/min) depends on the 2k time in minutes, the training level and the weight class (75 kg)."""
    x = seconds / 60
    if trained:
        y = (15.7 if weight_kg > 75 else 15.1) - 1.5 * x
    else:
        y = (10.7 if weight_kg > 75 else 10.2) - 0.9 * x
    return y * 1000 / weight_kg


def _session_metrics(res: Result) -> None:
    ex = [s for s in res.segments if s.kind == "exercise"]
    sm = res.summary
    w = res.profile.weight_kg
    if w and sm.get("watts_at_vt1"):
        sm["watts_at_vt1_per_kg"] = sm["watts_at_vt1"] / w
    tests = [s for s in ex if s.piece is not None and 1990 <= s.piece.distance_m <= 2010]
    if w and tests:
        t = min(s.piece.duration_s for s in tests)
        sm["test_2k_s"] = t
        # "low training" category for the first months; switch once training is regular (>= 3 months)
        sm["vo2max_2k"] = vo2max_from_2k(t, w, trained=False)
    sm["n_exercises"] = len(ex)
    sm["work_s"] = sum(s.duration_s for s in ex)
    sm["distance_m"] = sum(s.metrics.get("distance_m", 0) or 0 for s in ex)
    kj = 0.0
    for s in ex:
        if s.piece is not None and s.metrics.get("watts_avg"):
            kj += s.metrics["watts_avg"] * s.duration_s / 1000
    sm["work_kj"] = kj
    if res.hr1s is not None:
        t, h = res.hr1s["t"].to_numpy(), res.hr1s["hr"].to_numpy()
        pre = res.hr1s[res.hr1s["t"] < (ex[0].t0 if ex else t[-1])]["hr"]
        rest_fallback = float(pre.min()) if len(pre) else float(np.percentile(h, 2))
        sm["hr_rest_used"] = res.profile.hr_rest or rest_fallback
        sm["trimp"] = hrmod.trimp(t, h, res.profile, rest_fallback)
        sm["hr_max_session"] = float(h.max())
        sm["zones_s"] = hrmod.time_in_zones(h, 1.0, res.profile)
        sm["duration_s"] = float(t[-1] - t[0])
    else:
        sm["duration_s"] = sm["work_s"]
