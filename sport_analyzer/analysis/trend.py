"""Session-to-session continuity: comparable fitness markers, their trend, and training load.

Single-session numbers are noisy (sleep, heat, caffeine, a talkative row partner), so the
evolution is read the way coaches and the monitoring literature do it:

* **A standardised anchor block** at the start of every session: a fixed submaximal workload
  (default 6 min at 60 W, 20 spm) and the HR it costs, plus the HR drop in the 60 s after.
  Fixed-workload exercise HR and HRR are the most practical daily fitness markers
  (Buchheit 2014, *Front Physiol* 5:73; Lamberts' LSCT).
* **A fitness point from every steady piece**, the way Firstbeat estimates fitness from any
  qualified workout: steady watts and HR, projected to a common 140 bpm along the athlete's
  own HR-power slope, plus an efficiency factor (watts per beat above rest, cf. TrainingPeaks EF).
* **A noise-aware trend**: a local-level Kalman filter + RTS smoother (the state-space form of
  fitness tracking, cf. Kolossa et al. 2017) gives a smoothed level and an uncertainty band.
  A change is only called real when it beats the smallest worthwhile change (SWC), taken as the
  measure's typical day-to-day CV (≈ 3 % for exercise HR; Buchheit et al. 2014, JSSM 13:476).
* **Training load**: Banister fitness (CTL, 42-day EWMA of TRIMP) and fatigue (ATL, 7-day),
  plus sessions and time-in-zone per week against goals.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

DEFAULT_SLOPE = 0.85  # bpm per W, the HR-power slope measured on 2026-09-26; replaced by the pooled personal one
REF_HR = 140.0
REST_HR_DEFAULT = 70.0  # fixed reference so EF is comparable across days (the pre-row minimum is noisy)


# --------------------------------------------------------------------------- per session

def is_hard(summary: dict) -> bool:
    """For display: a hard session spent > 10 min at zone 4+ (a spike, a sprint finish or late drift doesn't count)."""
    z = summary.get("zones_s") or []
    return len(z) == 5 and z[3] + z[4] > 600


def _steady(p, e0: float, e1: float):
    s = p.strokes
    s = s[(s["elapsed"] >= e0) & (s["elapsed"] < e1)]
    w, h = s["watts"].dropna(), s["hr"].dropna()
    h = h[(h > 40) & (h < 230)]
    if len(w) < 10 or len(h) < 10:
        return None
    spm = float(s["spm"].dropna().median()) if s["spm"].notna().any() else None
    # robust centre and spread: a couple of demo pushes or a stop to answer the phone shouldn't disqualify the piece
    wm = float(w.median())
    cv = float(1.4826 * (w - wm).abs().median() / wm) if wm > 0 else 1.0
    return wm, float(h.mean()), spm, cv


def anchor_test(segments, anchor_watts: float = 60.0, tol: float = 0.10) -> dict | None:
    """The standardised anchor block: first piece whose minutes 1-6 sit at ``anchor_watts`` ±10 %.

    Returns HR over minute 5-6 (steady state at a fixed workload) and, when the anchor was its own
    PM5 piece followed by a stop, the HR drop in the next 60 s."""
    for seg in segments:
        p = seg.piece
        if seg.kind != "exercise" or p is None or p.duration_s < 330:
            continue
        st = _steady(p, 60, 360)
        if st is None or abs(st[0] - anchor_watts) > tol * anchor_watts:
            continue
        end = min(360.0, p.duration_s)
        last = _steady(p, end - 60, end)
        if last is None:
            continue
        out = {"hr_anchor": last[1], "anchor_watts": st[0], "anchor_spm": st[2], "anchor_piece": p.key}
        if p.duration_s <= 420 and seg.metrics.get("hrr60") is not None:
            out["anchor_hrr60"] = seg.metrics["hrr60"]
        return out
    return None


def piece_fitness(seg, slope: float, hr_rest: float = REST_HR_DEFAULT) -> dict | None:
    """Steady-state watts and HR of one piece (min 5 to half-way), projected to 140 bpm.

    Skipping the first 5 min avoids the HR on-kinetics; stopping at half-way keeps most of the
    cardiac drift out (and a sprint finish). Pieces >= 8 min whose window is steady (power CV < 20 %) qualify."""
    p = seg.piece
    if seg.kind != "exercise" or p is None or p.duration_s < 480:
        return None
    st = _steady(p, 300, max(p.duration_s / 2, 480))
    if st is None or st[3] >= 0.2:
        return None
    w, h, _, _ = st
    if h < hr_rest + 25:
        return None
    out = {"steady_watts": w, "steady_hr": h, "ef_net": w / (h - hr_rest)}
    if abs(REF_HR - h) <= 20:  # no more than ~25 W of projection along the slope
        out["w140_est"] = w + (REF_HR - h) / slope
    return out


def pooled_slope(history: list[dict], today: float | None = None) -> float:
    """Median personal HR-power slope (bpm/W) over sessions with a good fit."""
    v = [m["summary"]["hr_power_slope"] for m in history
         if m.get("summary", {}).get("hr_power_slope") and (m["summary"].get("hr_power_r2") or 0) > 0.6]
    if today:
        v.append(today)
    v = [x for x in v if 0.3 < x < 2.0]
    return float(np.median(v)) if v else DEFAULT_SLOPE


# --------------------------------------------------------------------------- markers registry

@dataclass(frozen=True)
class Marker:
    key: str
    label: str
    unit: str
    better: str  # "up" | "down"
    swc: float  # smallest worthwhile change
    relative: bool  # swc is a fraction of the level (True) or absolute units
    fmt: str
    tip: str  # glossary key


MARKERS = [
    Marker("w140", "Watts at 140 bpm", "W", "up", 0.05, True, "{:.0f}", "w140"),
    Marker("hr_anchor", "HR at the anchor", "bpm", "down", 0.03, True, "{:.0f}", "hr_anchor"),
    Marker("ef_net", "Watts per beat", "W/bpm", "up", 0.05, True, "{:.2f}", "ef_net"),
    Marker("decoupling", "Decoupling", "%", "down", 3.0, False, "{:.1f}", "decoupling"),
    Marker("hrr60", "HR recovery 60 s", "bpm", "up", 4.0, False, "{:.0f}", "hrr"),
    Marker("m_per_stroke", "Metres per stroke", "m", "up", 0.03, True, "{:.1f}", "m_stroke"),
    Marker("vt1_hr", "Aerobic threshold HR", "bpm", "up", 0.03, True, "{:.0f}", "vt1"),
]
MARKER = {m.key: m for m in MARKERS}


def session_values(m: dict) -> dict:
    """The comparable markers of one session, from its claude-metrics.json dict."""
    s = m.get("summary", {})
    ex = [x for x in m.get("segments", []) if x["kind"] == "exercise" and x.get("piece")]
    out = {}
    out["w140"] = s.get("watts_at_hr140") if s.get("watts_at_hr140") is not None else s.get("w140_est")
    out["hr_anchor"] = s.get("hr_anchor")
    out["ef_net"] = s.get("ef_net")
    # decoupling only on steady pieces: a sprint finish or a planned build makes it meaningless
    long = [x for x in ex if (x["metrics"].get("duration_s") or 0) >= 900 and x["metrics"].get("decoupling_pct") is not None
            and (x["metrics"].get("watts_cv") or 1) < 0.2]
    if long:
        out["decoupling"] = max(long, key=lambda x: x["metrics"]["duration_s"])["metrics"]["decoupling_pct"]
    # HRR60 only says something after a hard finish (same rule as the exercise cards)
    hrr = [x["metrics"]["hrr60"] for x in ex
           if (x["metrics"].get("hrr60") or 0) > 0 and (x["metrics"].get("hr_end") or 0) >= 140]
    out["hrr60"] = s.get("anchor_hrr60") if s.get("anchor_hrr60") is not None else (max(hrr) if hrr else None)
    mps = [x for x in ex if (x["metrics"].get("duration_s") or 0) >= 600 and x["metrics"].get("m_per_stroke")]
    if mps:
        out["m_per_stroke"] = max(mps, key=lambda x: x["metrics"]["duration_s"])["metrics"]["m_per_stroke"]
    out["vt1_hr"] = s.get("hr_at_alpha1_075")
    return {k: float(v) for k, v in out.items() if v is not None and np.isfinite(v)}


def series(sessions: list[dict]) -> dict[str, pd.DataFrame]:
    """Per marker: a (date, value) frame over the sessions that have it, oldest first."""
    rows = {k: [] for k in MARKER}
    for m in sessions:
        for k, v in session_values(m).items():
            rows[k].append((date.fromisoformat(m["date"]), v))
    return {k: pd.DataFrame(r, columns=["date", "value"]) for k, r in rows.items()}


# --------------------------------------------------------------------------- trend math

def kalman_level(dates, values, swc: float, relative: bool, drift_per_week: float = 0.01) -> pd.DataFrame:
    """Local-level (random walk) Kalman filter + RTS smoother on irregularly spaced sessions.

    Measurement noise = the SWC (the measure's day-to-day CV); process noise lets the true level
    move ~``drift_per_week`` of itself per week. Returns date, value, level, sd."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    y = np.asarray(values, float)
    n = len(y)
    if n == 0:
        return pd.DataFrame(columns=["date", "value", "level", "sd"])
    scale = abs(float(np.median(y))) or 1.0
    r = (swc * scale if relative else swc) ** 2
    q_day = (drift_per_week * scale) ** 2 / 7 if relative else (drift_per_week * 10 * swc) ** 2 / 7
    xf, pf, xp, pp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    x, p = y[0], 4 * r  # vague prior centred on the first point
    for i in range(n):
        dt = 0 if i == 0 else max((d[i] - d[i - 1]).days, 1)
        xp[i], pp[i] = x, p + q_day * dt
        k = pp[i] / (pp[i] + r)
        x, p = xp[i] + k * (y[i] - xp[i]), (1 - k) * pp[i]
        xf[i], pf[i] = x, p
    xs, ps = xf.copy(), pf.copy()
    for i in range(n - 2, -1, -1):  # Rauch-Tung-Striebel smoother
        c = pf[i] / pp[i + 1]
        xs[i] = xf[i] + c * (xs[i + 1] - xp[i + 1])
        ps[i] = pf[i] + c * c * (ps[i + 1] - pp[i + 1])
    return pd.DataFrame({"date": d.dt.date, "value": y, "level": xs, "sd": np.sqrt(ps)})


def signed(fmt: str, x: float) -> str:
    t = fmt.format(abs(x))
    return "±" + t if float(t) == 0 else ("+" if x > 0 else "−") + t


def change_call(mk: Marker, delta: float, level: float) -> str:
    """'better' / 'lower' (the gentle word for worse) / 'noise' against the SWC."""
    swc = mk.swc * abs(level) if mk.relative else mk.swc
    if abs(delta) < swc:
        return "noise"
    return "better" if (delta > 0) == (mk.better == "up") else "lower"


def since_last(sessions: list[dict], today: dict) -> list[dict]:
    """For each marker present today: change vs the previous session that has it, and vs the
    28-day baseline (median), called against the SWC."""
    vals = session_values(today)
    day = date.fromisoformat(today["date"])
    prev = [(date.fromisoformat(m["date"]), session_values(m)) for m in sessions if date.fromisoformat(m["date"]) < day]
    out = []
    for mk in MARKERS:
        if mk.key not in vals:
            continue
        v = vals[mk.key]
        before = [(d, x[mk.key]) for d, x in prev if mk.key in x]
        row = {"key": mk.key, "label": mk.label, "unit": mk.unit, "tip": mk.tip, "value": mk.fmt.format(v), "raw": v,
               "better_dir": mk.better}
        if not before:
            row.update(status="new", note="first measurement: today sets the baseline")
        else:
            pd_, pv = before[-1]
            delta = v - pv
            row.update(prev=pv, prev_date=pd_.isoformat(), delta=delta, delta_s=signed(mk.fmt, delta),
                       status=change_call(mk, delta, pv))
            base = [x for d, x in before if d >= day - timedelta(days=28)]
            if len(base) >= 2:
                b = float(np.median(base))
                row.update(base=b, base_delta=v - b, base_status=change_call(mk, v - b, b))
            row["note"] = {"noise": "within day-to-day noise", "better": "a real step forward",
                           "lower": "below last time (one day is not a trend)"}[row["status"]] + f" · vs {pd_:%d %b}"
        out.append(row)
    return out


def training_load(sessions: list[dict], until: date, goal_per_week: int) -> dict:
    """Banister CTL/ATL/TSB from daily TRIMP and a per-week summary (Monday-based weeks)."""
    if not sessions:
        return {}
    start = min(date.fromisoformat(m["date"]) for m in sessions)
    days = pd.date_range(start, until, freq="D")
    load = pd.Series(0.0, index=days)
    for m in sessions:
        d = pd.Timestamp(m["date"])
        if d in load.index:
            load[d] += m.get("summary", {}).get("trimp") or 0.0
    a42, a7 = 1 - np.exp(-1 / 42), 1 - np.exp(-1 / 7)
    ctl, atl, c, a = [], [], 0.0, 0.0
    for x in load.to_numpy():
        c += a42 * (x - c)
        a += a7 * (x - a)
        ctl.append(c)
        atl.append(a)
    daily = pd.DataFrame({"date": days.date, "trimp": load.to_numpy(), "ctl": ctl, "atl": atl})
    daily["tsb"] = daily["ctl"] - daily["atl"]

    weeks = {}
    for m in sessions:
        d = date.fromisoformat(m["date"])
        wk = d - timedelta(days=d.weekday())
        w = weeks.setdefault(wk, {"week": wk, "sessions": 0, "minutes": 0.0, "zones_min": [0.0] * 5, "trimp": 0.0})
        s = m.get("summary", {})
        w["sessions"] += 1
        w["minutes"] += (s.get("work_s") or 0) / 60
        w["trimp"] += s.get("trimp") or 0
        for i, z in enumerate(s.get("zones_s") or []):
            w["zones_min"][i] += z / 60
    wk = until - timedelta(days=until.weekday())
    while wk >= start - timedelta(days=start.weekday()):  # keep empty weeks visible: gaps are part of the story
        weeks.setdefault(wk, {"week": wk, "sessions": 0, "minutes": 0.0, "zones_min": [0.0] * 5, "trimp": 0.0})
        wk -= timedelta(days=7)
    weekly = sorted(weeks.values(), key=lambda w: w["week"])
    for w in weekly:
        tot = sum(w["zones_min"])
        # "easy" = below 80 % HRmax (Z1-Z3), which sits close to the DFA-α1 thresholds measured so far
        w["easy_share"] = sum(w["zones_min"][:3]) / tot if tot else None
    return {"daily": daily, "weekly": weekly, "goal": goal_per_week,
            "ctl": float(daily["ctl"].iloc[-1]), "atl": float(daily["atl"].iloc[-1]), "tsb": float(daily["tsb"].iloc[-1])}


def trend_summary(sessions: list[dict], until: date, goal_per_week: int) -> dict:
    """Everything the report, the progress page and the coach notes need about the arc."""
    ser = series(sessions)
    smooth = {}
    for mk in MARKERS:
        df = ser[mk.key]
        if len(df):
            smooth[mk.key] = kalman_level(df["date"], df["value"], mk.swc, mk.relative)
    arc = {}
    for k, df in smooth.items():
        if len(df) < 2:
            continue
        mk = MARKER[k]
        first, last = df.iloc[0], df.iloc[-1]
        delta = float(last["level"] - first["level"])
        noise = float(np.hypot(first["sd"], last["sd"]))
        status = change_call(mk, delta, float(first["level"]))
        if status != "noise" and abs(delta) < noise:  # beats the SWC, but not the filter's own uncertainty
            status = "noise"
        arc[k] = {"first": float(first["level"]), "last": float(last["level"]), "delta": delta, "sd": noise,
                  "n": len(df), "since": first["date"].isoformat(), "status": status}
    return {"series": ser, "smooth": smooth, "arc": arc, "load": training_load(sessions, until, goal_per_week)}
