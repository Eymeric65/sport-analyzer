"""Render a ``Result`` into a single self-contained ``claude-report.html``."""

from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import markdown as md
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from ..analysis import trend as T
from ..analysis.hr import ZONE_NAMES
from ..analyze import Result
from . import figures as F
from .narrative import GLOSSARY, _vt1, alpha1_insights, alpha1_verdict, breathing_insights, cardio_insights, trend_insights

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
ASSETS = HERE / "assets"


def mmss(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "–"
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def num(v, fmt="{:.0f}", suffix="") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return fmt.format(v) + suffix


def _div(fig) -> str:
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def _md(items: list[str]) -> list[str]:
    return [md.markdown(x)[3:-4] for x in items]  # strip <p></p>


def _kpis(seg, vt1: float | None = None) -> list[tuple]:
    """(label, value, sub, glossary key) tiles for an exercise card."""
    m = seg.metrics
    k = [("Time", mmss(m.get("duration_s")), "", None)]
    if seg.piece is not None:
        k += [
            ("Distance", num(m.get("distance_m"), suffix=" m"), num(m.get("m_per_stroke"), "{:.1f}", " m/stroke"), "m_stroke"),
            ("Power", num(m.get("watts_steady_avg") or m.get("watts_avg"), suffix=" W"),
             f"CV {num((m.get('watts_cv') or np.nan) * 100, suffix=' %')} · max {num(m.get('watts_max'))} W", "power"),
            ("Pace", mmss(m.get("pace_500m")), "/500 m", "pace"),
            ("Stroke rate", num(m.get("spm_avg"), "{:.1f}"), f"± {num(m.get('spm_sd'), '{:.1f}')} spm", "spm"),
        ]
    if "hr_avg" in m:
        k += [("Heart rate", num(m["hr_avg"]), f"max {num(m['hr_max'])} · end {num(m.get('hr_end'))} bpm", "hr")]
        # HRR60 only says something about fitness after a hard finish, not after an easy piece
        if m.get("hrr60") is not None and (m.get("hr_end") or 0) >= 140:
            k += [("HR recovery", num(m["hrr60"], suffix=" bpm"), "drop in 60 s", "hrr")]
        if m.get("hr_recovery_tau_s") is not None:
            k += [("Recovery τ", num(m["hr_recovery_tau_s"], suffix=" s"), "shorter = fitter", "tau")]
        if m.get("hr_t63_s") is not None:
            k += [("HR lag", num(m["hr_t63_s"], suffix=" s"), f"to 63 % of +{num(m.get('hr_rise_bpm'))} bpm", "t63")]
        if m.get("hr_drift_bpm_per_min") is not None:
            k += [("HR drift", num(m["hr_drift_bpm_per_min"], "{:+.2f}"), "bpm/min after min 3", "drift")]
    if m.get("decoupling_pct") is not None:
        k += [("Decoupling", num(m["decoupling_pct"], "{:.1f}", " %"), "power:HR, < 5 % steady", "decoupling")]
    if m.get("alpha1") is not None:
        a = m["alpha1"]
        k += [("DFA α1", num(a, "{:.2f}"), alpha1_verdict(a, m.get("hr_avg"), vt1)[0], "alpha1")]
    if m.get("acc_rhythm_cv") is not None:
        k += [("Rhythm", num(m["acc_rhythm_cv"] * 100, "{:.1f}", " %"), "stroke timing spread", "rhythm")]
    if m.get("breath_coupling_1") is not None:
        k += [("Breath:stroke", f"{m['breath_coupling_1'] * 100:.0f}% 1:1", f"{m.get('breath_coupling_2', 0) * 100:.0f}% 2:1", "coupling")]
    return k


def _target_check(seg) -> list[dict]:
    out, m, t = [], seg.metrics, seg.target
    if t.get("zone") and m.get("zones_s"):
        z = int(t["zone"])
        tot = sum(m["zones_s"]) or 1
        inside, above = m["zones_s"][z - 1] / tot, sum(m["zones_s"][z:]) / tot
        status = "good" if inside >= 0.6 else ("serious" if above > inside else "warning")
        out.append({"status": status, "text": f"Z{z} target · {inside * 100:.0f}% in zone · {above * 100:.0f}% above"})
    if t.get("watts") and m.get("watts_steady_avg"):
        d = m["watts_steady_avg"] - t["watts"]
        out.append({"status": "good" if abs(d) <= 0.1 * t["watts"] else "warning",
                    "text": f"{t['watts']} W target · {m['watts_steady_avg']:.0f} W ({d:+.0f})"})
    if t.get("spm") and m.get("spm_avg"):
        d = m["spm_avg"] - t["spm"]
        out.append({"status": "good" if abs(d) <= 1.0 else "warning", "text": f"{t['spm']} spm target · {m['spm_avg']:.1f} ({d:+.1f})"})
    return out


def metrics_json(res: Result) -> dict:
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if np.isnan(v) else round(float(v), 3)
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, date):
            return v.isoformat()
        return v

    return clean({
        "date": res.session.day.isoformat(),
        "profile": {"method": res.profile.method, "zone_bounds": res.profile.bounds()},
        "summary": res.summary,
        "segments": [
            {"kind": s.kind, "label": s.label, "piece": s.piece.key if s.piece else None,
             "start_min": (s.t0 - res.t_origin) / 60, "target": s.target, "metrics": s.metrics}
            for s in res.segments
        ],
        "recommendations": {k: v for k, v in res.recommendations.items() if k in ("actions", "zones", "why_next")}
        | {"next": res.recommendations.get("next", {}).get("name")},
        # the full plan made after this session: calendar.html shows the latest one and what each day had planned
        "plan": {k: res.recommendations.get(k) for k in ("why_next", "next", "week", "sessions", "zones")},
        "notes": res.notes,
    })


def _logo() -> str:
    svg = (ASSETS / "logo.svg").read_text().replace('fill="#000000"', 'fill="currentColor"')
    return svg.replace("<svg ", '<svg class="mark" aria-hidden="true" ', 1)


def _font(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode()


def render(res: Result, out: Path | None = None) -> Path:
    s = res.session
    out = out or s.path / "claude-report.html"
    sm = res.summary
    rec = res.recommendations
    ex = [x for x in res.segments if x.kind == "exercise"]
    rests = [x for x in res.segments if x.kind == "rest"]
    n_session = len(res.history) + 1

    kpis = [
        ("Session", mmss(sm.get("duration_s")), f"{len(ex)} exercises", None),
        ("Work", mmss(sm.get("work_s")), f"{sm.get('distance_m', 0):.0f} m rowed", None),
        ("Energy", num(sm.get("work_kj"), suffix=" kJ"), "on the flywheel", "energy"),
    ]
    if res.hr1s is not None:
        kpis += [
            ("HR max", num(sm.get("hr_max_session"), suffix=" bpm"), f"{100 * sm['hr_max_session'] / res.profile.hr_max_est:.0f}% of est. HRmax", "hrmax"),
            ("Load", num(sm.get("trimp")), "TRIMP", "trimp"),
        ]
        if sm.get("hr_at_alpha1_075"):
            kpis.append(("Aerobic threshold", num(sm["hr_at_alpha1_075"], suffix=" bpm"), "DFA α1 = 0.75 · estimate", "vt1"))
        if sm.get("watts_at_vt1"):
            kpis.append(("Watts at threshold", num(sm["watts_at_vt1"], suffix=" W"),
                         (f"{sm['watts_at_vt1_per_kg']:.2f} W/kg · " if sm.get("watts_at_vt1_per_kg") else "") + "the number to grow",
                         "watts_vt1"))
    if sm.get("vo2max_2k"):
        kpis.append(("VO2max (2k)", num(sm["vo2max_2k"], "{:.0f}"), f"ml/kg/min · 2k in {mmss(sm['test_2k_s'])}", "vo2max"))
        pre = next((x for x in rests if x.label == "Before start" and x.metrics.get("breath_rate")), None)
        if pre:
            kpis.append(("Breathing at rest", num(pre.metrics["breath_rate"], suffix=" /min"), "from the ECG", "breath_rate"))

    cards = []
    for seg in ex:
        cards.append({
            "label": seg.label,
            "start": num((seg.t0 - res.t_origin) / 60, "{:.1f}"),
            "piece": seg.piece.key if seg.piece else "not logged on PM5",
            "kpis": _kpis(seg, _vt1(res)),
            "checks": _target_check(seg),
            "fig": _div(F.piece_detail(res, seg)),
            "zones": _div(F.zone_bar(seg.metrics["zones_s"], 64)) if seg.metrics.get("zones_s") else "",
            "comment": md.markdown(seg.target["comment"]) if seg.target.get("comment") else "",
        })

    rest_rows = [{
        "label": r.label, "start": num((r.t0 - res.t_origin) / 60, "{:.1f}"), "dur": mmss(r.duration_s),
        "hr": f"{num(r.metrics.get('hr_start'))} → {num(r.metrics.get('hr_end'))}" if "hr_start" in r.metrics else "–",
        "breath": num(r.metrics.get("breath_rate"), "{:.0f}", " /min"), "alpha1": num(r.metrics.get("alpha1"), "{:.2f}"),
    } for r in rests]

    figs = {}
    if res.hr1s is not None:
        figs["hr_power"] = _div(F.hr_power(res))
        figs["alpha_t"] = _div(F.alpha1_timeline(res))
        figs["alpha_hr"] = _div(F.alpha1_vs_hr(res))
        figs["breath_rec"] = _div(F.breathing_recovery(res))
        figs["coupling"] = _div(F.coupling_bars(res))
        good = res.breathing[res.breathing["rate"].notna()].sort_values("edr_q", ascending=False)
        if len(good):
            figs["breath_trace"] = _div(F.breathing_trace(res, float(good["t"].iloc[0])))
        longest = max(ex, key=lambda x: x.duration_s) if ex else None
        if longest:
            sp = F.rowing_spectrum(res, (longest.t0 + longest.t1) / 2)
            if sp:
                figs["spectrum"], figs["spectrum_share"] = _div(sp[0]), sp[1]
        pre = [x for x in rests if x.duration_s > 20]
        if pre and longest:
            figs["ecg"] = _div(F.ecg_snippets(res, pre[0].t0 + min(40, pre[0].duration_s / 2), (longest.t0 + longest.t1) / 2))
    mj = metrics_json(res)
    tr = T.trend_summary(res.history + [mj], s.day, res.goals.cardio_sessions_per_week)
    today = s.day.isoformat()
    figs["fitness"] = _div(F.fitness_trend(tr, today))
    figs["load"] = _div(F.training_load(tr))
    figs["multiples"] = _div(F.small_multiples(tr, today))
    since = T.since_last(res.history, mj)[:4]

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    html = env.get_template("report.html").render(
        title=f"{s.day:%A %-d %B}",
        day=s.day,
        n_session=n_session,
        brief=md.markdown(s.brief),
        goal=res.goals,
        goal_md=md.markdown(res.goals.body_md) if res.goals.body_md else "",
        coach_notes=md.markdown(s.notes_md) if s.notes_md else "",
        kpis=kpis,
        tips=GLOSSARY,
        rec=rec,
        profile=res.profile,
        zone_names=ZONE_NAMES,
        zone_bounds=[round(b) for b in res.profile.bounds()],
        zone_colors=F.ZONE_COLORS,
        timeline=_div(F.timeline(res)),
        session_zones=_div(F.zone_bar(sm["zones_s"])) if sm.get("zones_s") else "",
        cards=cards,
        rests=rest_rows,
        has_h10=res.hr1s is not None,
        since=since,
        figs=figs,
        insights={"breath": _md(breathing_insights(res)) if res.hr1s is not None else [],
                  "alpha": _md(alpha1_insights(res)) if res.hr1s is not None else [],
                  "cardio": _md(cardio_insights(res)) if res.hr1s is not None else [],
                  "trend": _md(trend_insights(tr))},
        summary=sm,
        notes=res.notes,
        logo=_logo(),
        font_roman=_font("fraunces-roman-latin.woff2"),
        font_italic=_font("fraunces-italic-latin.woff2"),
        plotlyjs=get_plotlyjs(),
    )
    out.write_text(html)
    (s.path / "claude-metrics.json").write_text(json.dumps(mj, indent=1, default=str))
    from .progress import render_progress

    render_progress(s.path.parent)
    from .calendar import render_calendar

    render_calendar(s.path.parent)
    return out


def trend_json(res: Result) -> dict:
    """The arc, for writing coach notes that build on the previous sessions."""
    mj = metrics_json(res)
    tr = T.trend_summary(res.history + [mj], res.session.day, res.goals.cardio_sessions_per_week)
    ld = tr["load"]
    prev = res.history[-1] if res.history else None
    return {
        "since_last": [{k: r.get(k) for k in ("label", "value", "unit", "delta_s", "status", "prev_date", "base_status")}
                       for r in T.since_last(res.history, mj)],
        "arc": tr["arc"],
        "load": {"ctl": round(ld["ctl"], 1), "atl": round(ld["atl"], 1),
                 "this_week_sessions": ld["weekly"][-1]["sessions"], "goal": ld["goal"]} if ld else None,
        "previous": {"date": prev["date"], "note": prev.get("_note")} if prev else None,
    }
