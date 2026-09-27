"""Loader tests on small synthetic recordings (no personal data in the repo)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sport_analyzer.align import align_pieces
from sport_analyzer.io.concept2 import load_pieces
from sport_analyzer.io.polar import load_recordings

EPOCH_2000 = 946684800  # Unix seconds at 2000-01-01
PHONE_T0 = 1_790_450_000.0  # Unix seconds of the synthetic recording start
HEADER = 'Number,"Time (seconds)","Distance (meters)","Pace (seconds)",Watts,Cal/Hr,"Stroke Rate","Heart Rate"\n'


def hr_curve(t):
    return 90 + 60 * np.clip((t - 60) / 120, 0, 1) - 40 * np.clip((t - 400) / 60, 0, 1)


def write_polar(session: Path, name: str, duration: float, sensor_offset_s: float, ecg_split: bool = False):
    dev = session / name / "Polar H10 TEST"
    dev.mkdir(parents=True)
    # ECG at 130 Hz in 0.5 s batches, sensor clock = phone clock + sensor_offset_s
    lines = []
    n = int(duration * 130)
    ts = np.arange(n) / 130
    for i in range(0, n, 65):
        data = [{"voltage": int(100 * np.sin(t)), "timeStamp": int((PHONE_T0 + t + sensor_offset_s - EPOCH_2000) * 1e9)}
                for t in ts[i:i + 65]]
        arrival = PHONE_T0 + ts[min(i + 64, n - 1)] + 0.05 + 0.1 * (i % 7 == 0)
        lines.append(json.dumps({"phoneTimestamp": int(arrival * 1000), "dataType": "ECG", "data": data}))
    (dev / "ECG.jsonl").write_text("\n".join(lines) + "\n")
    # ACC at 200 Hz, split in two parts
    acc = []
    ta = np.arange(int(duration * 200)) / 200
    for i in range(0, len(ta), 100):
        data = [{"timeStamp": int((PHONE_T0 + t + sensor_offset_s - EPOCH_2000) * 1e9), "x": -980, "y": 20, "z": -140}
                for t in ta[i:i + 100]]
        acc.append(json.dumps({"phoneTimestamp": int((PHONE_T0 + ta[min(i + 99, len(ta) - 1)] + 0.04) * 1000),
                               "dataType": "ACC", "data": data}))
    half = len(acc) // 2
    (dev / "ACC.jsonl").write_text("\n".join(acc[:half]) + "\n")
    (dev / "ACC_part2.jsonl").write_text("\n".join(acc[half:]) + "\n")
    # HR at 1 Hz
    hr = [json.dumps({"phoneTimestamp": int((PHONE_T0 + t) * 1000), "dataType": "HR",
                      "data": [{"hr": int(round(hr_curve(t))), "contactStatus": True, "rrsMs": [int(60000 / hr_curve(t))]}]})
          for t in np.arange(0, duration, 1.0)]
    (dev / "HR.jsonl").write_text("\n".join(hr) + "\n")


def write_c2(session: Path, rid: int, pieces: list[tuple[float, float]]):
    """pieces: list of (start_t, duration) in recording time; several pieces -> one interval file."""
    rows, k = [], 1
    for start, dur in pieces:
        for el in np.arange(1.0, dur, 3.0):
            rows.append(f"{k},{el:.1f},{el * 3:.1f},150.0,100,600,20,{int(round(hr_curve(start + el)))}\n")
            k += 1
    (session / f"concept2-result-{rid}.csv").write_text(HEADER + "".join(rows))


@pytest.fixture
def session(tmp_path):
    s = tmp_path / "20260101"
    s.mkdir()
    write_polar(s, "PolarRecording_20260101_100000", 600, sensor_offset_s=-7 * 365.25 * 86400)  # clock ~7 y off
    write_polar(s, "PolarRecording_20260101_095950", 8, sensor_offset_s=0)  # test stub
    write_c2(s, 111, [(100, 180)])
    write_c2(s, 112, [(395, 30), (430, 30)])
    (s / "concept2-result-112 (1).csv").write_text((s / "concept2-result-112.csv").read_text())
    return s


def test_stub_skipped_and_parts_merged(session):
    recs, skipped = load_recordings(session)
    assert len(recs) == 1 and len(skipped) == 1 and "stub" in skipped[0]
    r = recs[0]
    assert abs(len(r.acc) - 600 * 200) < 5  # both ACC parts
    assert np.all(np.diff(r.acc["t"]) > 0)
    assert abs(r.ecg_fs - 130) < 0.5


def test_clock_reanchored_on_phone_time(session):
    r = load_recordings(session)[0][0]
    assert any("not synchronised" in n for n in r.notes)
    # re-anchored with the minimal arrival latency (50 ms) -> within 60 ms of true time
    assert abs(r.ecg["t"].iloc[0] - PHONE_T0) < 0.06


def test_intervals_split_and_duplicates_ignored(session):
    pieces = load_pieces(session)
    assert [p.key for p in pieces] == ["111", "112#1", "112#2"]


def test_alignment(session):
    r = load_recordings(session)[0][0]
    pieces = load_pieces(session)
    align_pieces(pieces, r)
    for p, true_start in zip(pieces, [100, 395, 430]):
        assert abs((p.t_start - r.t0) - true_start) < 3, p.key
