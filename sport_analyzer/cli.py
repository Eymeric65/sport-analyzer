"""``sport-analyzer <session_dir> [...]``: analyse a session and write claude-report.html in it."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

from .analyze import analyze
from .report.render import metrics_json, render, trend_json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="+", type=Path, help="session folder(s), e.g. ~/Drive/perso/sport/session/20260926")
    ap.add_argument("--json", action="store_true", help="print the metrics JSON (for writing coach notes)")
    ap.add_argument("--all", action="store_true",
                    help="treat each path as the folder holding YYYYMMDD sessions and re-render all of them, oldest first")
    ap.add_argument("--pages", action="store_true",
                    help="only re-render calendar, progress and nutrition for the folder holding the sessions "
                         "(after a new food log, no session to analyse)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    dirs = [d.expanduser() for d in args.sessions]
    if args.pages:
        from .report.calendar import render_calendar
        from .report.nutrition import render_nutrition
        from .report.progress import render_progress

        for d in dirs:
            for f in (render_calendar, render_progress, render_nutrition):
                if out := f(d):
                    print(f"✓ {out}")
        return
    if args.all:
        dirs = [x for d in dirs for x in sorted(d.iterdir()) if x.is_dir() and re.fullmatch(r"\d{8}", x.name)]
    for d in dirs:
        res = analyze(d)
        out = render(res)
        print(f"✓ {out}")
        if args.json:
            print(json.dumps(metrics_json(res) | {"trend": trend_json(res)}, indent=1, default=str))
        else:
            for s in res.segments:
                m = s.metrics
                print(f"  {s.kind:8s} {s.label:28s} {s.duration_s / 60:5.1f} min"
                      + (f"  {m['watts_avg']:.0f} W" if m.get("watts_avg") else "")
                      + (f"  HR {m['hr_avg']:.0f}/{m['hr_max']:.0f}" if m.get("hr_avg") else ""))


if __name__ == "__main__":
    main()
