"""``nutrition.html`` next to the session folders: calories in vs out, protein, weigh-ins."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from ..analysis import energy as E
from ..analysis import trend as T
from ..analyze import load_profile
from ..history import load_all
from ..io.food import FoodDay, food_dir, load_food
from ..io.goals import load_goals
from . import figures as F
from .narrative import GLOSSARY
from .render import TEMPLATES, _div, _font, _logo, _md, num

WINDOW_DAYS = 28


def build_days(body: E.Body, sessions: list[dict], food: list[FoodDay], deficit: float,
               start: date, end: date) -> list[dict]:
    """One row per calendar day from ``start`` to ``end``: target, intake, balance."""
    by_day: dict[str, list[dict]] = {}
    for m in sessions:
        by_day.setdefault(m["date"], []).append(m)
    fd = {f.day: f for f in food}
    rows = []
    d = start
    while d <= end:
        ss = by_day.get(d.isoformat(), [])
        f = fd.get(d)
        rowing = sum((m["summary"].get("kcal_active") or 0) for m in ss)
        extra = f.extra_kcal if f else 0.0
        hard = any(T.is_hard(m["summary"]) for m in ss)
        tg = E.day_target(body, rowing + extra, deficit, hard)
        kcal = f.kcal if f else None
        complete = bool(f and f.complete and kcal is not None)
        rows.append({
            "date": d, "sessions": [{"dir": m.get("_dir"), "kcal": m["summary"].get("kcal_active"),
                                     "kcal_hr": m["summary"].get("kcal_hr")} for m in ss],
            "hard": hard, "rowing": rowing, "extra": extra, "extra_note": f.extra_note if f else "",
            **tg, "kcal": kcal, "complete": complete,
            "protein": f.total("protein_g") if f else None, "carbs": f.total("carbs_g") if f else None,
            "fat": f.total("fat_g") if f else None,
            "balance": (kcal - tg["tdee"]) if complete else None,
            "meals": f.meals if f else [], "note": f.note if f else "",
            "weight_kg": f.weight_kg if f else None, "waist_cm": f.waist_cm if f else None,
        })
        d += timedelta(days=1)
    return rows


def week_summary(rows: list[dict], today: date) -> dict | None:
    """Average over the last 7 days that were fully logged."""
    wk = [r for r in rows if r["complete"] and today - timedelta(days=6) <= r["date"] <= today]
    if not wk:
        return None
    bal = float(np.mean([r["balance"] for r in wk]))
    prot = [r["protein"] for r in wk if r["protein"] is not None]
    return {"n": len(wk), "intake": float(np.mean([r["kcal"] for r in wk])),
            "tdee": float(np.mean([r["tdee"] for r in wk])), "balance": bal,
            "kg_per_week": bal * 7 / E.KCAL_PER_KG_FAT,
            "protein": float(np.mean(prot)) if prot else None}


def insights(week: dict | None, body: E.Body, rate: float, protein_target: float, weighins: list) -> list[str]:
    out = []
    if week is None:
        out.append("No fully logged day in the last week yet. A day counts once its Joplin note is marked complete.")
    else:
        aim = -rate
        k = week["kg_per_week"]
        if k <= aim * 0.5 and k >= aim * 2:
            out.append(f"Over {_days(week['n'])} you averaged **{week['balance']:+.0f} kcal/day**: right on pace "
                       f"for about {abs(k):.2f} kg a week. Nice and steady, which is what keeps the training quality up.")
        elif k < aim * 2:
            out.append(f"Over {_days(week['n'])} you averaged **{week['balance']:+.0f} kcal/day**, a faster pace "
                       f"(~{abs(k):.2f} kg/week) than the {rate:.2f} kg/week you set. Worth adding a snack around hard "
                       "sessions: a big deficit shows up first as flat legs on the 4×4.")
        else:
            out.append(f"Over {_days(week['n'])} you averaged **{week['balance']:+.0f} kcal/day** "
                       f"(about {k:+.2f} kg/week). Small wins: the easiest cut is usually snacks and drinks, not meals.")
        if week["protein"] is not None:
            if week["protein"] >= 0.9 * protein_target:
                out.append(f"Protein averaged **{week['protein']:.0f} g/day**, close to the {protein_target:.0f} g target. "
                           "That is what keeps the weight you lose off the muscle side.")
            else:
                out.append(f"Protein averaged **{week['protein']:.0f} g/day**; aim for about {protein_target:.0f} g. "
                           "Eggs, yoghurt, tofu, fish or a glass of milk at each meal close the gap.")
    if len(weighins) < 3:
        out.append("The targets come from formulas (±10 %). Without a scale, an occasional weigh-in (gym, pharmacy) "
                   "or a waist measurement in the note lets the trend check them.")
    return out


def render_nutrition(root: Path, today: date | None = None) -> Path | None:
    prof = load_profile()
    body = prof.body
    sessions = load_all(root) if root.is_dir() else []
    food = load_food(root)
    if not sessions and not food:
        return None
    today = today or date.today()
    goals = load_goals(root / sessions[-1]["_dir"]) if sessions else load_goals(root / "x")
    latest = sessions[-1]["_dir"] if sessions else None
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    common = dict(tips=GLOSSARY, goal=goals, latest=latest, nav_base="", generated=today, logo=_logo(),
                  font_roman=_font("fraunces-roman-latin.woff2"), font_italic=_font("fraunces-italic-latin.woff2"))
    if body is None:
        html = env.get_template("nutrition.html").render(body=None, **common)
        (root / "nutrition.html").write_text(html)
        return root / "nutrition.html"

    rate = goals.weight_rate_kg_per_week
    deficit = E.deficit_for_rate(rate)
    first = min([f.day for f in food] + [today])
    start = max(first, today - timedelta(days=WINDOW_DAYS - 1))
    rows = build_days(body, sessions, food, deficit, start, today)
    week = week_summary(rows, today)
    weighins = [(f.day, f.weight_kg) for f in food if f.weight_kg]
    waists = [(f.day, f.waist_cm) for f in food if f.waist_cm]
    # a typical easy row, for "if you row today" on a rest day
    easy = [m["summary"]["kcal_active"] for m in sessions[-10:]
            if m["summary"].get("kcal_active") and not T.is_hard(m["summary"])]
    typical = float(np.median(easy)) if easy else None
    t = rows[-1]
    protein_target = E.PROTEIN_G_PER_KG * body.weight_kg
    html = env.get_template("nutrition.html").render(
        body=body, rate=rate, deficit=deficit, today=t, week=week, typical=typical,
        protein_target=protein_target, weight_now=weighins[-1] if weighins else None,
        insights=_md(insights(week, body, rate, protein_target, weighins)),
        figs={"balance": _div(F.energy_balance(rows)), "body": _div(F.body_trend(weighins, waists, goals.weight_target_kg))},
        rows=list(reversed([r for r in rows if r["kcal"] is not None or r["sessions"]])),
        food_dir=food_dir(root), plotlyjs=get_plotlyjs(), num=num, **common,
    )
    out = root / "nutrition.html"
    out.write_text(html)
    return out



def _days(n: int) -> str:
    return "1 logged day" if n == 1 else f"{n} logged days"
