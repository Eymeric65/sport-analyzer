"""``sport-analyzer <session_dir> [...]``: analyse a session and write claude-report.html in it."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from .analyze import analyze
from .report.render import metrics_json, render


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="+", type=Path, help="session folder(s), e.g. ~/Drive/perso/sport/session/20260926")
    ap.add_argument("--json", action="store_true", help="print the metrics JSON (for writing coach notes)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    for d in args.sessions:
        res = analyze(d.expanduser())
        out = render(res)
        print(f"✓ {out}")
        if args.json:
            print(json.dumps(metrics_json(res), indent=1))
        else:
            for s in res.segments:
                m = s.metrics
                print(f"  {s.kind:8s} {s.label:28s} {s.duration_s / 60:5.1f} min"
                      + (f"  {m['watts_avg']:.0f} W" if m.get("watts_avg") else "")
                      + (f"  HR {m['hr_avg']:.0f}/{m['hr_max']:.0f}" if m.get("hr_avg") else ""))


if __name__ == "__main__":
    main()
