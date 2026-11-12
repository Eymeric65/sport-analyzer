"""Plain-language explanations: tooltip glossary + data-driven insights per section."""

from __future__ import annotations

import numpy as np

GLOSSARY = {
    "hr": "Beats per minute from the H10 chest strap (its own on-board detection, the most reliable HR source).",
    "hrmax": "Maximum heart rate. Estimated from age with Tanaka's formula (208 − 0.7 × age) until a real max is measured. "
             "It anchors all the zones.",
    "trimp": "Training impulse (Banister): minutes weighted by how hard the heart worked, growing exponentially with intensity. "
             "About 50 = moderate session, 100+ = hard. Useful to compare days and weeks.",
    "energy": "Mechanical work done on the flywheel = average watts × time. Your body burns about 4× this (plus basal) because muscles are about 20–25 % efficient.",
    "vt1": "Aerobic threshold (VT1, first ventilatory threshold): the intensity where breathing first speeds up and lactate starts to rise. "
           "Below it you can go for hours, and this is where ~80 % of cardio training should happen.",
    "alpha1": "DFA α1: how 'fractal' the beat-to-beat rhythm is. ~1.0 and above = relaxed, rich variability. It falls as effort rises: "
              "0.75 ≈ aerobic threshold (VT1), 0.5 ≈ anaerobic threshold (VT2). It is measured from the ECG, with no mask or lab test needed. "
              "While rowing, the stroke rhythm can pull it down even at an easy HR, so a low value under your threshold HR is not graded.",
    "watts_vt1": "The power you can hold with HR at your aerobic threshold. This is the key cardio-progress number: the same HR "
                 "producing more watts = a bigger aerobic engine.",
    "power": "Average power on the flywheel. 'Steady' excludes the first 10 s of acceleration. CV = coefficient of variation: "
             "stroke-to-stroke spread, below 10 % is even pacing.",
    "pace": "Time per 500 m, the rowing standard. Pace depends on the cube root of watts, so halving the split takes 8× the power.",
    "spm": "Strokes per minute. Keeping it fixed (e.g. 20) forces the power to come from a longer, harder drive instead of a faster rate.",
    "m_stroke": "Distance per stroke. At a fixed stroke rate it is the best single measure of technique and power per stroke. "
                "Good club rowers do ~10–12 m at 20 spm.",
    "hrr": "Heart-rate recovery: how many bpm HR drops in the 60 s after the effort stops. > 25 bpm is good; it improves with cardio fitness.",
    "tau": "Recovery time constant: HR falls exponentially after an effort, and τ is the time to cover 63 % of that fall. "
           "Shorter = faster parasympathetic reactivation = fitter. Typical: 30–90 s after hard efforts.",
    "t63": "HR on-kinetics: seconds for HR to cover 63 % of its rise once an effort starts. It shows HR lagging behind effort, "
           "which is why short intervals can't be paced by HR.",
    "drift": "Cardiac drift: the slope of HR over a steady piece after the first 3 min. Near 0 = truly steady aerobic effort; "
             "> +0.5 bpm/min means the effort was above the aerobic threshold (or heat or dehydration).",
    "decoupling": "Power:HR decoupling: how much the watts-per-beat ratio dropped from the 1st to the 2nd half. < 5 % = aerobically sustainable.",
    "rhythm": "Variability of the time between strokes, measured by the chest accelerometer. < 5 % = very regular rhythm.",
    "coupling": "Share of 32-s windows where the strongest peak of the ECG breathing signal falls at 1× (1:1) or 2× (2:1) the stroke rate. "
                "The stroke's own motion also moves the electrodes at that rate, so a high number does NOT prove breathing is locked. "
                "See the spectrum chart below.",
    "edr": "ECG-derived respiration: the height of each R-peak rises and falls as the lungs fill (the heart moves relative to the electrodes). "
           "Following that wave gives the breathing rate with no extra sensor.",
    "rsa": "Respiratory sinus arrhythmia: the heart speeds up while you breathe in and slows while you breathe out. Strong at rest, "
           "it fades as HR rises, so it serves as a cross-check at rest only.",
    "breath_rate": "Breaths per minute estimated from EDR at rest (not while rowing). Typical rest: 12–20. It jumps to 25–40 after hard efforts.",
    "zones": "HR zones as % of HRmax (5-zone model used by Polar and Concept2). Add a resting HR to switch to %HR-reserve (Karvonen), which is more personal.",
    "hr_power": "Each dot is 1 min of a steady piece: average watts versus average HR. The line is your current HR–power relationship. "
                "As you get fitter it shifts right: more watts at the same HR.",
    "vo2max": "Estimated maximal oxygen uptake, from Concept2's 2k formula with your weight. It is the ceiling of the cardio engine: "
              "~40–45 for an untrained young man, 60–70 for trained rowers. Raised mostly by the hard 4×4 intervals.",
    "w140": "Watts you produce at 140 bpm. Taken from the HR–power line when a session has several powers, otherwise "
            "projected from one steady piece (minutes 5 to half-way) along your own HR-power slope. Higher = a bigger aerobic engine. "
            "Day-to-day noise is about ±5 %.",
    "hr_anchor": "Heart rate at minute 6 of the anchor block (6 min at a fixed 60 W, 20 spm, at the start of each session). Same work, so "
                 "a lower HR means the heart pumps more per beat. This is the cleanest session-to-session fitness marker; noise is about ±3 % "
                 "(sleep, heat, coffee).",
    "ef_net": "Efficiency factor: steady watts divided by heartbeats above a 70 bpm reference. Like TrainingPeaks' EF, it rises "
              "as the aerobic engine and the technique improve.",
    "ctl": "Fitness (CTL) is the 42-day average of daily TRIMP, and fatigue (ATL) the 7-day one (Banister's model). Fitness rising "
           "slowly and steadily is the goal. When fatigue sits well above fitness, an easy day pays off.",
    "trend": "The line is a Kalman-filtered trend: it follows your sessions but discounts single-day noise, and the band is its "
             "uncertainty (±1 SD). A change only counts as real when it is bigger than the marker's normal day-to-day noise.",
    "grey": "The 'grey zone' (between VT1 and ~88 % HRmax): hard enough to be tiring, not hard enough to push VO2max. "
            "Beginners spend too much time here; the best cardio programs mostly avoid it.",
}


def _fmt(x, f="{:.0f}"):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def breathing_insights(res) -> list[str]:
    out = []
    ex = [s for s in res.segments if s.kind == "exercise"]
    pre = next((s for s in res.segments if s.label == "Before start"), None)
    base = pre.metrics.get("breath_rate") if pre else None
    if base:
        out.append(f"**Resting breathing before the session: {base:.0f} /min**, in the normal 12–20 range.")
    for label, d in res.breath_recovery.items():
        if len(d) < 3:
            continue
        first = d.iloc[:3]["rate"].median()
        last = d.iloc[-3:]["rate"].median()
        span = d["since_s"].iloc[-1]
        out.append(f"After **{label}**: breathing ~{first:.0f} /min in the first minute, "
                   f"~{last:.0f} /min after {span / 60:.1f} min of rest"
                   + (" (still above the pre-session baseline: the rest was too short to fully recover)." if base and last > base + 4 else "."))
    cps = [s for s in ex if s.metrics.get("breath_coupling_1") is not None]
    if cps:
        avg1 = np.mean([s.metrics["breath_coupling_1"] for s in cps])
        share = np.mean([s.metrics.get("edr_stroke_share", np.nan) for s in cps])
        out.append(f"While rowing, the strongest peak of the breathing signal sat on the stroke rate in **{avg1 * 100:.0f}%** of windows, "
                   f"but only **{share * 100:.0f}%** of the signal's power is on the stroke frequency and its multiples. "
                   "So the stroke leaves a clear mark, but most of the signal is elsewhere: part of it is probably breathing that is "
                   "<i>not</i> locked 1:1 to the stroke, and part is noise. Honest verdict: breathing while rowing is not measurable yet. "
                   "The paced-breathing calibration is the first step to fix that.")
    ag = res.summary.get("breath_agree_edr_acc")
    if ag is not None:
        out.append(f"Reliability: at rest, EDR is sharp (clear single peak) but only {ag * 100:.0f}% of windows agree with the accelerometer, "
                   "which mostly picks up fidgeting. Treat rest rates as ±3 /min until calibrated.")
    return out


def alpha1_verdict(a: float, hr: float | None, vt1: float | None) -> tuple[str, str]:
    """(tile label, sentence) for a piece's median α1, cross-checked against its HR.

    While rowing, the stroke rhythm and EMG in the ECG pull α1 down even at an easy HR. A low α1
    with HR well under the aerobic threshold is that artefact, not intensity, so it is not graded.
    """
    if a <= 0.75 and hr and vt1 and hr < vt1 - 5:
        return ("not readable (rowing)",
                f"not readable: HR ({hr:.0f} bpm) was well under your aerobic threshold (≈ {vt1:.0f} bpm), "
                "and while rowing the stroke rhythm pulls α1 down, so it can't grade this piece")
    if a > 1.0:
        return "easy", "well below threshold: truly easy"
    if a > 0.75:
        return "below VT1", "just below threshold: ideal endurance intensity"
    if a > 0.5:
        return "grey zone", "above the aerobic threshold: the 'grey zone'"
    return "above VT2", "near/above the anaerobic threshold: hard"


def _vt1(res) -> float | None:
    return (res.recommendations.get("zones") or {}).get("vt1_hr") or res.summary.get("hr_at_alpha1_075")


def alpha1_insights(res) -> list[str]:
    out = []
    sm = res.summary
    hr75 = sm.get("hr_at_alpha1_075")
    wv = sm.get("watts_at_vt1")
    if not hr75 and res.hrv is not None:
        ok = res.hrv[res.hrv["reliable"]]
        if len(ok) and ok["alpha1"].quantile(0.1) > 0.75:
            out.append(f"α1 stayed above 0.75 almost all session (10th percentile {ok['alpha1'].quantile(0.1):.2f}, "
                       f"HR up to {res.summary.get('hr_max_session') or ok['hr'].max():.0f} bpm), so today can't place the threshold: "
                       "it only says the threshold sits above the HR you reached. That is what an easy day should look like.")
    if hr75:
        out.append(f"**Aerobic threshold ≈ {hr75:.0f} bpm** ({100 * hr75 / res.profile.hr_max_est:.0f}% of est. HRmax)"
                   + (f", which today corresponds to **≈ {wv:.0f} W**." if wv else "."))
    for s in res.segments:
        a = s.metrics.get("alpha1")
        if a is None or s.kind != "exercise":
            continue
        verdict = alpha1_verdict(a, s.metrics.get("hr_avg"), _vt1(res))[1]
        out.append(f"{s.label}: median α1 **{a:.2f}**, {verdict}.")
    cd = next((s for s in res.segments if s.kind == "exercise" and "cool" in s.label.lower()), None)
    if cd and cd.metrics.get("alpha1") and cd.metrics.get("hr_avg") and hr75 and cd.metrics["alpha1"] < 0.8 and cd.metrics["hr_avg"] < hr75:
        out.append(f"Note the cool-down: HR {cd.metrics['hr_avg']:.0f} bpm is below the threshold, yet α1 was {cd.metrics['alpha1']:.2f}. "
                   "Fatigue lowers α1 late in a session, so thresholds measured fresh (start of a session) are the ones to trust.")
    share = sm.get("hrv_unreliable_share")
    if share:
        out.append(f"{share * 100:.0f}% of windows were rejected for ECG noise (mostly the hardest rowing minutes). "
                   "A step test at a fixed stroke rate (planned this week) will give a much cleaner estimate.")
    return out


def cardio_insights(res) -> list[str]:
    out = []
    sm = res.summary
    fit = res.hr_power_fit
    if fit:
        out.append(f"Each extra watt currently costs **{fit['b']:.2f} bpm** (fit R² {fit['r2']:.2f})."
                   + (f" At 140 bpm you produce **≈ {sm['watts_at_hr140']:.0f} W**. This is the baseline to beat."
                      if sm.get("watts_at_hr140") else ""))
    long = [s for s in res.segments if s.kind == "exercise" and s.metrics.get("hr_drift_bpm_per_min") is not None]
    for s in long:
        d = s.metrics["hr_drift_bpm_per_min"]
        if d > 0.5:
            out.append(f"{s.label}: HR drifted **+{d:.1f} bpm/min** at constant power. That is the signature of an effort above the aerobic threshold.")
        elif abs(d) <= 0.5:
            out.append(f"{s.label}: HR was stable ({d:+.1f} bpm/min), a truly steady aerobic effort.")
    taus = [(s.label, s.metrics["hr_recovery_tau_s"]) for s in res.segments if s.metrics.get("hr_recovery_tau_s")]
    if taus:
        out.append("HR recovery time constants: " + ", ".join(f"{l} τ = {t:.0f} s" for l, t in taus)
                   + ". These should shorten as fitness improves.")
    t63 = [s.metrics["hr_t63_s"] for s in res.segments if s.metrics.get("hr_t63_s")]
    if t63:
        out.append(f"In the short bursts, HR needed ~{np.mean(t63):.0f} s to cover 63 % of its rise. "
                   "A 1-min effort is over before HR catches up, so pace short efforts by watts or feel, and use HR for efforts of 3 min or more.")
    return out


def trend_insights(trend: dict) -> list[str]:
    """Plain-language reading of the arc, honest about noise and the number of points."""
    from ..analysis.trend import MARKER

    out, steady = [], []
    arc = trend.get("arc", {})
    for k in ("w140", "hr_anchor", "ef_net", "m_per_stroke", "decoupling", "hrr60", "vt1_hr"):
        a = arc.get(k)
        if not a:
            continue
        mk = MARKER[k]
        f = mk.fmt
        move = f"from {f.format(a['first'])} to {f.format(a['last'])} {mk.unit} since {_date_s(a['since'])} ({a['n']} sessions)"
        if a["status"] == "better":
            out.append(f"**{mk.label}: a real gain**, {move}. That is beyond day-to-day noise.")
        elif a["status"] == "lower":
            out.append(f"**{mk.label}** has eased {move}. Worth watching, but sleep, heat and the session mix move it too.")
        else:
            steady.append((mk, a))
    if steady:
        few = min(a["n"] for _, a in steady) < 6
        items = ", ".join(f"{mk.label[0].lower() + mk.label[1:] if not mk.label.startswith('HR') else mk.label} ({mk.fmt.format(a['first'])} → {mk.fmt.format(a['last'])} {mk.unit})" for mk, a in steady)
        out.append(f"**Holding steady, within day-to-day noise:** {items}. "
                   + ("With this few points (and a 10-day break in October) the bands are still wide: "
                      "a real trend usually shows after 4–6 weeks of regular sessions." if few else ""))
    if "hr_anchor" not in trend.get("smooth", {}):
        out.append("**No anchor points yet.** Starting each session with the same 6-min anchor block will give the cleanest trend line here.")
    ld = trend.get("load") or {}
    if ld:
        wk = [w for w in ld["weekly"] if w["sessions"]][-1:]  # the last week with rowing
        if wk:
            w = wk[0]
            share = f", {w['easy_share'] * 100:.0f} % of it below 80 % HRmax" if w.get("easy_share") is not None else ""
            out.append(f"Week of {w['week']:%d %b}: **{w['sessions']} of {ld['goal']}** planned cardio sessions, "
                       f"{w['minutes']:.0f} min of rowing{share}.")
        if ld["atl"] > 1.3 * ld["ctl"] and ld["ctl"] > 5:
            out.append("Fatigue (7-day load) is well above fitness right now: recent days were big ones, so the next easy day is earning its keep.")
    return out


def _date_s(iso: str) -> str:
    from datetime import date

    return f"{date.fromisoformat(iso):%-d %b}"
