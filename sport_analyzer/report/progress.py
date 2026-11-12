"""``progress.html`` next to the session folders: the whole history on one page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from ..analysis import trend as T
from ..history import load_all
from ..io.goals import load_goals
from . import figures as F
from .narrative import GLOSSARY, trend_insights
from .render import TEMPLATES, _div, _font, _logo, _md, mmss, num


def _row(m: dict, today: str) -> dict:
    s = m.get("summary", {})
    v = T.session_values(m)
    ex = [x["label"] for x in m.get("segments", []) if x["kind"] == "exercise"]
    label = ex[0] if len(ex) == 1 else (f"{len(ex)} pieces" + (f" · {ex[0]}" if ex else ""))
    return {
        "date": date.fromisoformat(m["date"]), "dir": m.get("_dir", m["date"].replace("-", "")),
        "label": label, "work": mmss(s.get("work_s")), "dist": num(s.get("distance_m"), suffix=" m"),
        "trimp": num(s.get("trimp")), "w140": num(v.get("w140")), "anchor": num(v.get("hr_anchor")),
        "mps": num(v.get("m_per_stroke"), "{:.1f}"), "hard": T.is_hard(s), "note": m.get("_note", ""),
        "today": m["date"] == today,
    }


def render_progress(root: Path) -> Path | None:
    sessions = load_all(root)
    if not sessions:
        return None
    last = date.fromisoformat(sessions[-1]["date"])
    goals = load_goals(root / sessions[-1]["_dir"])
    tr = T.trend_summary(sessions, max(last, date.today()), goals.cardio_sessions_per_week)
    today = sessions[-1]["date"]
    # headline tiles: first value -> trend level now, per marker
    tiles = []
    for k, a in tr["arc"].items():
        mk = T.MARKER[k]
        d = a["delta"]
        tiles.append({"label": mk.label, "tip": mk.tip, "unit": mk.unit, "status": a["status"],
                      "value": mk.fmt.format(a["last"]),
                      "delta_s": T.signed(mk.fmt, d),
                      "note": f"since {date.fromisoformat(a['since']):%-d %b} · {a['n']} sessions · "
                              + {"noise": "within noise", "better": "real gain", "lower": "eased"}[a["status"]]})
    ld = tr["load"]
    total = {"sessions": len(sessions), "km": sum((m["summary"].get("distance_m") or 0) for m in sessions) / 1000,
             "hours": sum((m["summary"].get("work_s") or 0) for m in sessions) / 3600,
             "since": date.fromisoformat(sessions[0]["date"])}
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    html = env.get_template("progress.html").render(
        total=total, tiles=tiles, tips=GLOSSARY, goal=goals, load=ld,
        insights=_md(trend_insights(tr)),
        figs={"fitness": _div(F.fitness_trend(tr, today)), "load": _div(F.training_load(tr)),
              "multiples": _div(F.small_multiples(tr, today))},
        rows=[_row(m, today) for m in reversed(sessions)],
        latest=sessions[-1]["_dir"],
        logo=_logo(), font_roman=_font("fraunces-roman-latin.woff2"), font_italic=_font("fraunces-italic-latin.woff2"),
        plotlyjs=get_plotlyjs(), generated=date.today(),
    )
    out = root / "progress.html"
    out.write_text(html)
    return out
