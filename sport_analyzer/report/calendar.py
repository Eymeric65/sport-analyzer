"""``calendar.html`` next to the session folders: what was done, and what comes next.

Past days show the sessions (linked to their reports). Days after the latest session show the
plan made from it (``plan`` in its claude-metrics.json). Past days that had a cardio session
planned but none logged show the plan faintly, as context rather than a reproach.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..analysis.trend import is_hard
from ..history import load_all
from ..io.goals import load_goals
from .narrative import GLOSSARY
from .render import TEMPLATES, _font, _logo, mmss

CARDIO = {"easy", "vo2", "step", "technique"}


def _done(m: dict) -> dict:
    s = m.get("summary", {})
    ex = [x["label"] for x in m.get("segments", []) if x["kind"] == "exercise"]
    label = ex[0] if len(ex) == 1 else (f"{len(ex)} pieces" if ex else "Session")
    return {"dir": m.get("_dir"), "label": label, "kind": "hard" if is_hard(s) else "easy",
            "work": mmss(s.get("work_s")), "km": (s.get("distance_m") or 0) / 1000, "trimp": s.get("trimp")}


def _planned_on(sessions: list[dict]) -> dict[date, dict]:
    """For every day, the plan from the latest session made before it."""
    out = {}
    for m in sessions:  # oldest first: later plans overwrite earlier ones
        made = date.fromisoformat(m["date"])
        for it in (m.get("plan") or {}).get("week") or []:
            d = date.fromisoformat(str(it["date"]))
            if d > made:
                out[d] = it
    return out


def build_calendar(sessions: list[dict], goal: int) -> dict:
    last = date.fromisoformat(sessions[-1]["date"])
    done: dict[date, list] = {}
    for m in sessions:
        done.setdefault(date.fromisoformat(m["date"]), []).append(_done(m))
    planned = _planned_on(sessions)
    first = date.fromisoformat(sessions[0]["date"])
    end = max([last] + list(planned))
    # one continuous run of weeks, from the first session's week to the end of the plan
    weeks = []
    wk = first - timedelta(days=first.weekday())
    while wk <= end:
        days = []
        for i in range(7):
            d = wk + timedelta(days=i)
            p = planned.get(d)
            cell = {"date": d, "done": done.get(d, []), "future": d > last, "plan": None,
                    "label": f"{d:%a %-d %b}" if d.day == 1 or d == wk and not weeks else f"{d:%a %-d}",
                    "new_month": d.day == 1}
            if p and not cell["done"] and (d > last or p["type"] in CARDIO):  # past gym/rest days aren't worth showing
                cell["plan"] = p
            days.append(cell)
        weeks.append({"days": days, "n": sum(len(c["done"]) for c in days), "goal": goal,
                      "km": sum(x["km"] for c in days for x in c["done"]), "future": wk > last,
                      "month": f"{(wk + timedelta(days=6)):%B}" if (wk + timedelta(days=6)).day <= 7 or not weeks else None})
        wk += timedelta(days=7)
    plan = sessions[-1].get("plan") or {}
    return {"weeks": weeks, "plan": plan, "last": last}


def render_calendar(root: Path) -> Path | None:
    sessions = load_all(root)
    if not sessions:
        return None
    goals = load_goals(root / sessions[-1]["_dir"])
    cal = build_calendar(sessions, goals.cardio_sessions_per_week)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    html = env.get_template("calendar.html").render(
        cal=cal, plan=cal["plan"], goal=goals, tips=GLOSSARY, latest=sessions[-1]["_dir"],
        nav_base="", generated=date.today(),
        logo=_logo(), font_roman=_font("fraunces-roman-latin.woff2"), font_italic=_font("fraunces-italic-latin.woff2"),
    )
    out = root / "calendar.html"
    out.write_text(html)
    return out
