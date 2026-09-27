"""Loader for recordings exported by the Polar sensor-logger app (JSONL, one batch per line).

Folder layout: ``PolarRecording_YYYYMMDD_HHMMSS/Polar H10 <id>/{ECG,ACC,HR,LOG}[_partN].jsonl``.

Time handling
-------------
Each sample carries a sensor ``timeStamp`` (ns since 2000-01-01 on the H10 clock) and each
batch a ``phoneTimestamp`` (ms, Unix epoch, time of reception).  The H10 clock is only set
when the app syncs it, so it can be years off.  We therefore map sensor time to phone time
with ``t_phone = t_sensor + offset`` where the offset is the lower envelope of
``phone_arrival - sensor_last_sample`` (arrival latency is always >= 0, so the smallest
values are the least-delayed batches).  Sample-level precision comes from the sensor clock.
All returned times are float seconds since the Unix epoch (``t``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DURATION_S = 60.0


@dataclass
class PolarRecording:
    name: str
    path: Path
    start_local: datetime  # from the folder name (phone local time)
    tz_offset_s: float  # local - UTC, derived from folder name vs phone timestamps
    ecg: pd.DataFrame = field(default_factory=pd.DataFrame)  # t, uv
    acc: pd.DataFrame = field(default_factory=pd.DataFrame)  # t, x, y, z (mG)
    hr: pd.DataFrame = field(default_factory=pd.DataFrame)  # t, hr, contact
    rr: pd.DataFrame = field(default_factory=pd.DataFrame)  # t (beat time), rr_ms
    clock_offsets: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def t0(self) -> float:
        return float(self.hr["t"].iloc[0]) if len(self.hr) else float(self.ecg["t"].iloc[0])

    @property
    def t1(self) -> float:
        return float(self.hr["t"].iloc[-1]) if len(self.hr) else float(self.ecg["t"].iloc[-1])

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0

    @property
    def ecg_fs(self) -> float:
        return 1.0 / np.median(np.diff(self.ecg["t"].to_numpy()[:5000]))

    @property
    def acc_fs(self) -> float:
        return 1.0 / np.median(np.diff(self.acc["t"].to_numpy()[:5000]))


def _parts(device_dir: Path, kind: str) -> list[Path]:
    """Return ``KIND.jsonl`` followed by ``KIND_part2.jsonl``, ``KIND_part3.jsonl``..."""
    files = []
    base = device_dir / f"{kind}.jsonl"
    if base.exists():
        files.append(base)
    parts = sorted(
        device_dir.glob(f"{kind}_part*.jsonl"),
        key=lambda p: int(re.search(r"_part(\d+)", p.name).group(1)),
    )
    return files + parts


def _read_batches(device_dir: Path, kind: str):
    for f in _parts(device_dir, kind):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _sampled_stream(device_dir: Path, kind: str, cols: list[str]):
    """Load a sensor-timestamped stream (ECG/ACC). Returns (sensor_ns, values, offset_s)."""
    ts, vals, lat = [], [], []
    for b in _read_batches(device_dir, kind):
        data = b["data"]
        if not data:
            continue
        ts.extend(s["timeStamp"] for s in data)
        vals.extend([s[c] for c in cols] for s in data)
        lat.append(b["phoneTimestamp"] / 1e3 - data[-1]["timeStamp"] / 1e9)
    if not ts:
        return None, None, None
    sensor_s = np.asarray(ts, dtype=np.int64) / 1e9
    order = np.argsort(sensor_s, kind="stable")
    sensor_s = sensor_s[order]
    values = np.asarray(vals, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(sensor_s) > 0])  # drop duplicated samples
    offset = float(np.percentile(np.asarray(lat), 2))
    return sensor_s[keep], values[keep], offset


def load_recording(rec_dir: Path) -> PolarRecording | None:
    rec_dir = Path(rec_dir)
    devices = [d for d in rec_dir.iterdir() if d.is_dir()]
    if not devices:
        return None
    dev = devices[0]
    m = re.search(r"(\d{8}_\d{6})", rec_dir.name)
    start_local = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")

    # HR: phone-timestamped, ~1 Hz, with RR intervals
    hr_rows, rr_ms, rr_batch_t = [], [], []
    for b in _read_batches(dev, "HR"):
        for s in b["data"]:
            t = b["phoneTimestamp"] / 1e3
            hr_rows.append((t, s["hr"], s.get("contactStatus", True)))
            for r in s.get("rrsMs") or []:
                rr_ms.append(r)
                rr_batch_t.append(t)
    hr = pd.DataFrame(hr_rows, columns=["t", "hr", "contact"])
    rec = PolarRecording(name=rec_dir.name, path=rec_dir, start_local=start_local, tz_offset_s=0.0, hr=hr)

    if rr_ms:
        # beat times: cumulative RR, anchored so that the cumsum hugs the batch arrival times
        rr_arr = np.asarray(rr_ms, float)
        cum = np.cumsum(rr_arr) / 1e3
        anchor = np.median(np.asarray(rr_batch_t) - cum)
        rec.rr = pd.DataFrame({"t": cum + anchor, "rr_ms": rr_arr})

    s, v, off = _sampled_stream(dev, "ECG", ["voltage"])
    if s is not None:
        rec.ecg = pd.DataFrame({"t": s + off, "uv": v[:, 0]})
        rec.clock_offsets["ECG"] = off
    s, v, off = _sampled_stream(dev, "ACC", ["x", "y", "z"])
    if s is not None:
        rec.acc = pd.DataFrame({"t": s + off, "x": v[:, 0], "y": v[:, 1], "z": v[:, 2]})
        rec.clock_offsets["ACC"] = off

    if rec.clock_offsets:
        sensor_year = 2000 + (rec.ecg["t"].iloc[0] - off) / (365.25 * 86400) if len(rec.ecg) else None
        if sensor_year and abs(sensor_year - start_local.year) > 0.5:
            rec.notes.append(
                f"H10 internal clock was not synchronised (it reads ~{sensor_year:.0f}); "
                "sample times were re-anchored on phone reception timestamps."
            )
    if len(hr):
        # folder name is phone local time; compare with the UTC epoch of the first sample
        diff = (start_local - datetime(1970, 1, 1)).total_seconds() - hr["t"].iloc[0]
        rec.tz_offset_s = round(diff / 900) * 900
    return rec


def load_recordings(session_dir: Path, min_duration_s: float = MIN_DURATION_S) -> tuple[list[PolarRecording], list[str]]:
    """Load all Polar recordings of a session, skipping short test stubs."""
    recs, skipped = [], []
    for d in sorted(Path(session_dir).glob("PolarRecording_*")):
        if not d.is_dir():
            continue
        rec = load_recording(d)
        if rec is None or len(rec.hr) == 0:
            skipped.append(f"{d.name} (empty)")
        elif rec.duration_s < min_duration_s:
            skipped.append(f"{d.name} ({rec.duration_s:.0f} s test stub)")
        else:
            recs.append(rec)
    return recs, skipped
