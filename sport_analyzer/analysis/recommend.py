"""Turn the analysis into actions: what to change, the next session, and the week ahead.

The cardio logic follows the best-supported model for improving aerobic fitness:
* ~80 % of the time **below the aerobic threshold (VT1)**: builds mitochondria, capillaries
  and stroke volume, and you can recover from it daily (Seiler 2010, Stöggl & Sperlich 2014);
* ~20 % **hard**, e.g. 4 x 4 min at 90-95 % HRmax with 3 min easy between: the most
  studied VO2max booster (Helgerud et al. 2007: +7 % VO2max in 8 weeks);
* **little in between**: the "grey zone" between VT1 and VT2 costs a lot of recovery for
  little extra adaptation, and it is where most beginners drift (today's 20-min piece).
The thresholds come from the athlete's own data (DFA-alpha1) whenever possible.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .cardio import watts_at_hr


def _r5(x: float) -> int:
    return int(5 * round(x / 5))


def personal_zones(res) -> dict:
    hm = res.profile.hr_max_est
    vt1 = res.summary.get("hr_at_alpha1_075")
    vt1_src = "DFA α1 (today)"
    if not vt1:  # an easy day never reaches the threshold: reuse the latest measured one
        past = [m for m in res.history if m.get("summary", {}).get("hr_at_alpha1_075")]
        if past:
            vt1, vt1_src = past[-1]["summary"]["hr_at_alpha1_075"], f"DFA α1 ({past[-1]['date']})"
        else:
            vt1, vt1_src = 0.77 * hm, "estimate (77 % HRmax)"
    fit = res.hr_power_fit
    easy_hi = _r5(vt1 - 5)
    z = {
        "vt1_hr": vt1,
        "vt1_src": vt1_src,
        "easy_hr": (min(_r5(0.6 * hm), easy_hi - 10), easy_hi),  # never inverted under a low cap
        "grey_hr": (easy_hi, _r5(0.88 * hm)),
        "hard_hr": (_r5(0.90 * hm), _r5(0.95 * hm)),
    }
    for k in ("easy", "grey", "hard"):
        lo, hi = z[f"{k}_hr"]
        if k == "easy":
            lo = hi - 10  # the useful easy band: the top 10 bpm under the cap
        wl, wh = watts_at_hr(fit, lo), watts_at_hr(fit, hi)
        z[f"{k}_w"] = (max(20, _r5(wl)), _r5(wh)) if wl is not None else None
    return z


def _was_hard(summary_zones_s, hr_max, bounds, work_s) -> bool:
    """Hard = more than 10 min at >= zone 4 (80 % HRmax)."""
    return summary_zones_s is not None and (summary_zones_s[3] + summary_zones_s[4]) > 600


def recommend(res) -> dict:
    sm, g, prof = res.summary, res.goals, res.profile
    ex = [s for s in res.segments if s.kind == "exercise"]
    z = personal_zones(res) if res.hr1s is not None else None
    actions: list[dict] = []

    # --- 1. intensity discipline on the long piece -------------------------------------
    longest = max(ex, key=lambda s: s.duration_s) if ex else None
    if z and longest and longest.metrics.get("zones_s") and longest.duration_s >= 600:
        zs = longest.metrics["zones_s"]
        above_vt1 = float(np.mean(res.hr1s[(res.hr1s["t"] >= longest.t0) & (res.hr1s["t"] < longest.t1)]["hr"] > z["vt1_hr"]))
        if above_vt1 > 0.4:
            w = z["easy_w"]
            actions.append({
                "title": f"Cap easy sessions at {z['easy_hr'][1]} bpm, not at a wattage",
                "why": f"In “{longest.label}”, HR was above your aerobic threshold (≈ {z['vt1_hr']:.0f} bpm, from {z['vt1_src']}) "
                       f"for {above_vt1 * 100:.0f}% of the time. That is the “grey zone”: tiring, "
                       "but a weaker signal for the aerobic engine than truly easy work.",
                "how": f"Set the PM5 to show HR. Row at whatever power keeps HR ≤ {z['easy_hr'][1]} bpm"
                       + (f" (≈ {w[0]}–{w[1]} W today)" if w else "")
                       + ". When HR creeps up late in the piece, drop 5 W. The power at that HR is what will rise over the weeks.",
            })

    # --- 2. stroke rate discipline in efforts --------------------------------------------
    over = [s for s in ex if s.target.get("spm") and s.metrics.get("spm_avg") and s.metrics["spm_avg"] - s.target["spm"] > 1.5]
    if over:
        worst = max(over, key=lambda s: s.metrics["spm_avg"])
        actions.append({
            "title": "Get power from the legs, not the stroke rate",
            "why": f"In “{worst.label}” the rate crept up to {worst.metrics['spm_avg']:.1f} spm against the {worst.target['spm']} asked for. "
                   "Rushing the recovery raises the rate and the HR without a longer stroke.",
            "how": "Keep the recovery slow (about 2× the drive time) and push harder with the legs. "
                   f"Your best stroke length today was {max(s.metrics.get('m_per_stroke') or 0 for s in ex):.1f} m; aim for ≥ 10 m at 20 spm.",
        })

    # --- 3. pacing (warm-up / first strokes) -------------------------------------------
    # only a real warm-up: the first of several pieces (a single long piece is the session itself)
    wu = ex[0] if len(ex) > 1 else None
    if wu and (wu.metrics.get("watts_cv") or 0) > 0.15:
        actions.append({
            "title": "Make the warm-up a ramp, not a sprint",
            "why": f"“{wu.label}” started hard and faded (power CV {wu.metrics['watts_cv'] * 100:.0f}%).",
            "how": "Do 3 steps of 3 min each (e.g. 45 → 55 → 65 W) so HR rises gradually. "
                   "It also gives DFA α1 a clean ramp to find your threshold on.",
        })

    # --- 4. calibration inputs -------------------------------------------------------------
    if not prof.hr_rest:
        actions.append({
            "title": "Measure your resting HR tomorrow morning",
            "why": "Zones currently use an age-based HRmax only. Adding resting HR (Karvonen) personalises them, "
                   "and its weekly trend is a free marker of cardio gains and fatigue.",
            "how": "Lie down for 3 min after waking with the H10 on (or count your pulse), take the lowest stable value, "
                   "and put it in profile.yaml as hr_rest.",
        })
    if res.breathing is not None:
        actions.append({
            "title": "Start each recording with 1 min of paced breathing",
            "why": "Breathing is extracted from the ECG without a reference. A known rate lets the report check "
                   "(and later train) the extraction on you.",
            "how": "Sit still after starting the H10 recording and breathe with a metronome at 12 /min "
                   "(5 s per breath) for 60 s, then start rowing. Note it in brief.md (“paced 12”).",
        })

    # --- 5. the anchor block: the one habit that makes every session comparable -----------
    if not sm.get("hr_anchor"):
        aw = int(prof.anchor_watts)
        actions.insert(0, {
            "title": f"Open every session with the {aw} W anchor",
            "why": "Your sessions vary (easy, intervals, rows with your wife), so their numbers can't be lined up directly. "
                   "One identical block at the start of each one gives a clean point every time: the HR it costs you. "
                   "Fixed-workload HR is the most practical day-to-day fitness marker in the monitoring research.",
            "how": f"Row 6 min at a flat {aw} W and {prof.anchor_spm} spm as its own PM5 piece, then sit still for 60 s before going on. "
                   f"Over the weeks, the HR at minute 6 should fall: that is the line in the Progress section.",
        })

    # --- next session + week ----------------------------------------------------------
    hard_today = _was_hard(sm.get("zones_s"), prof.hr_max_est, prof.bounds(), sm.get("work_s", 0)) or any(
        (s.metrics.get("hr_max") or 0) > 0.88 * prof.hr_max_est for s in ex)
    recent_hard = [m for m in res.history
                   if date.fromisoformat(m["date"]) > res.session.day - timedelta(days=3) and m.get("summary", {}).get("hard")]
    sm["hard"] = bool(hard_today)
    lo_min, hi_min = g.session_minutes
    sessions = _session_library(z, res, lo_min, hi_min)
    # without HR the session's intensity is unknown: default to easy rather than stacking hard days
    first_type = "easy" if (hard_today or recent_hard or res.hr1s is None) else "vo2"
    week = _week_plan(res.session.day, first_type, g, sessions, threshold_known=bool(sm.get("hr_at_alpha1_075")),
                      first_week=len(res.history) < 4)
    return {"actions": actions[:5], "zones": z, "next": sessions[first_type], "week": week, "sessions": sessions,
            "why_next": ("No heart-rate data today, so intensity is unknown: the safe default is an easy aerobic session."
                         if res.hr1s is None else
                         "Today was hard (a lot of time above the aerobic threshold), so the next cardio session is easy. "
                         "The adaptation happens while you recover."
                         if hard_today else
                         f"Today was easy, but {recent_hard[-1]['date']} was hard: one more easy day before the key hard session, "
                         "so it lands on fresh legs."
                         if first_type == "easy" else "You are recovered, so the next session is the key hard one.")}


def _session_library(z, res, lo_min: int, hi_min: int) -> dict:
    ew = f"{z['easy_w'][0]}–{z['easy_w'][1]} W" if z and z.get("easy_w") else "easy watts"
    ecap = z["easy_hr"][1] if z else 145
    hard = z["hard_hr"] if z else (170, 180)
    # interval power: what drove HR to ~90 % HRmax today, trimmed for 4-min efforts
    bursts = [s for s in res.segments if s.kind == "exercise" and s.duration_s <= 180 and s.metrics.get("watts_steady_avg")]
    burst_w = np.median([s.metrics["watts_steady_avg"] for s in bursts]) if bursts else None
    iw = f"{_r5(0.75 * burst_w)}–{_r5(0.85 * burst_w)} W" if burst_w else "a hard but sustainable effort"
    main_easy = max(25, hi_min - 18)
    aw, aspm = int(res.profile.anchor_watts), res.profile.anchor_spm
    # the same fixed block every session = a comparable fitness point every session (see analysis/trend.py)
    anchor = ("Anchor (own PM5 piece)", "6 min + 1 min still", f"exactly {aw} W, flat · then sit still 60 s", f"{aspm} spm")
    return {
        "easy": {
            "name": "Aerobic base",
            "tag": "EASY · Z2",
            "minutes": hi_min,
            "blocks": [
                anchor,
                ("Warm-up ramp", "4 min", f"rising to ≈ {z['easy_w'][0]} W" if z and z.get("easy_w") else "rising gently", "18–20 spm"),
                ("Steady row", f"{main_easy} min", f"HR ≤ {ecap} bpm (≈ {ew})", "20 spm, long strokes"),
                ("Cool-down", "5 min", "very easy", "18 spm"),
            ],
            "cue": "You should be able to talk in full sentences. If HR drifts over the cap, reduce watts, don't stop.",
            "success": f"HR stays ≤ {ecap} bpm for the whole steady block (the report checks this).",
        },
        "vo2": {
            "name": "VO2max intervals 4×4",
            "tag": "HARD · Z5",
            "minutes": 52,
            "blocks": [
                anchor,
                ("Warm-up ramp", "5 min", "easy, then 2 × 20 s pick-ups", "20 spm"),
                ("4 × 4 min ON", "16 min", f"≈ {iw}, HR reaching {hard[0]}–{hard[1]} bpm by minute 3", "22–24 spm"),
                ("3 min easy between", "9 min", f"≈ {ew}, keep moving", "18–20 spm"),
                ("Cool-down", "8 min", "very easy", "18 spm"),
            ],
            "cue": "The first interval should feel too easy. HR lags about 2 min behind the effort (see today's bursts), "
                   "so judge the watts on interval 1 by feel and check that HR arrives in range during intervals 2–4.",
            "success": "All 4 intervals held at the same watts ±5 %, and the last one is not the slowest.",
        },
        "step": {
            "name": "Threshold step test",
            "tag": "TEST · RAMP",
            "minutes": 42,
            "blocks": [
                anchor,
                ("Steps of 3 min, no rest", "24 min", "40 → 50 → 60 → 70 → 80 → 90 → 100 → 110 W, stop at the step you can't finish", "20 spm, fixed"),
                ("Cool-down", "10 min", "very easy", "18 spm"),
            ],
            "cue": "A fixed stroke rate and a clean H10 recording matter more than going far. This gives DFA α1 a clean ramp "
                   "to pin your aerobic threshold (today's estimate mixes very different efforts).",
            "success": "The report finds α1 = 0.75 with a tight fit, giving you personal watts and HR for the easy zone.",
        },
        "technique": {
            "name": "Easy + technique",
            "tag": "EASY · SKILL",
            "minutes": lo_min + 5,
            "blocks": [
                anchor,
                ("Drills: legs only → legs+back → full", "3 × 4 min", f"HR ≤ {ecap}", "18 spm"),
                ("Pause drill (pause at arms-away)", "3 × 3 min", f"HR ≤ {ecap}", "18–20 spm"),
                ("Steady, focus on length", f"{max(10, lo_min - 25)} min", f"HR ≤ {ecap}, aim ≥ 10 m/stroke", "20 spm"),
            ],
            "cue": "Sequence: legs → body swing → arms on the drive; arms → body → legs on the recovery. "
                   "Metres per stroke is the number that tells you the technique is working.",
            "success": "Metres per stroke is up compared with today at the same stroke rate.",
        },
    }


def _week_plan(day: date, first_type: str, g, sessions: dict, threshold_known: bool, first_week: bool) -> list[dict]:
    n = max(2, min(6, g.cardio_sessions_per_week))
    gym = g.gym_sessions_per_week
    # 7-day template (index = days after today). Hard work is kept 48 h apart and never the day after the gym leg day.
    order = ["easy", "gym", "vo2", "technique", "gym", "step" if first_week else "easy", "rest"]
    if first_type == "vo2":
        order = ["vo2", "easy", "gym", "technique", "gym", "easy", "rest"]
    cardio_slots = [i for i, t in enumerate(order) if t not in ("gym", "rest")]
    for i in cardio_slots[n:]:
        order[i] = "rest"
    gym_slots = [i for i, t in enumerate(order) if t == "gym"]
    for i in gym_slots[gym:]:
        order[i] = "rest"
    plan = []
    for i, t in enumerate(order, start=1):
        d = day + timedelta(days=i)
        if t in sessions:
            s = sessions[t]
            plan.append({"date": d, "type": t, "name": s["name"], "tag": s["tag"], "minutes": s["minutes"]})
        elif t == "gym":
            plan.append({"date": d, "type": t, "name": "Gym", "tag": "STRENGTH", "minutes": None})
        else:
            plan.append({"date": d, "type": "rest", "name": "Rest or walk", "tag": "RECOVER", "minutes": None})
    return plan
