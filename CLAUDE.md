# sport-analyzer

Turns one training-day folder (`~/Drive/perso/sport/session/YYYYMMDD/`) into `claude-report.html` in that folder.

Objectives live in `~/Drive/perso/sport/goals.md` (primary goal: cardio, 4+ cardio sessions/week, 45–60 min).
They drive `analysis/recommend.py`: the actions, the next-session prescription and the 7-day plan (80/20 polarised; 4×4 VO2max as the hard session).

## Daily workflow (when the user drops a new session)
1. `uv run sport-analyzer ~/Drive/perso/sport/session/YYYYMMDD --json` to render the report and print the metrics.
2. Read `brief.md` and the metrics. Then write `claude-notes.md` in the session folder:
   - YAML front matter `exercises:`, keyed by C2 piece key (`"121025840"`, or `"121025841#2"` for interval 2 of an interval workout), each with `label`, and optionally `zone`, `watts`, `spm` (targets from the brief) and `comment`.
   - A markdown body with the coach notes: what went well, what to change, what to do next time.
   - **Tone: be a trainer, not a scientist grading a lab result.** Lead with what went well, frame changes as encouragement ("next time, try…"), never as faults.
     - The brief gives the context: take it at face value and don't turn it against the user. If they say they were talking or rowing in sync with someone, a drifting cadence is expected. Don't flag it as a failure or repeat basics they already know (e.g. that cadence should be 20 spm).
     - Keep the numbers, but avoid blunt verdicts ("the weak point", "X was wrong"). One gentle technique cue beats a list of corrections.
3. Re-run the command without `--json` so the report includes the notes. Check it (headless Chrome screenshot).

## Data facts learned
- **Concept2 CSV**: one row per stroke and no absolute time. Interval workouts put all intervals in ONE file, with elapsed time resetting at each interval. Stroke-rate glitches (43, 155 spm) are masked. The user sometimes drops byte-identical duplicates (`… (1).csv`): the loader dedupes by result id.
- **Polar H10 (sensor-logger app, JSONL)**: ECG at 130 Hz, ACC at 200 Hz (mG, split into `ACC_partN`), HR at 1 Hz with RR.
  - The sensor clock may be unsynchronised, so time is re-anchored on `phoneTimestamp`.
  - Recordings shorter than 60 s are test stubs and are skipped.
- **Alignment**: each C2 piece is placed on the H10 timeline by matching its HR column (the PM5 receives HR from the H10).
- **ECG while rowing** is EMG-heavy. The HR-guided R-peak detector plus Kubios correction beats neurokit and the strap's RR, which drops 8–15 % of beats while rowing. DFA-α1 windows with more than 2 % suspicious beats are rejected.
- **Breathing**: EDR (R-amplitude) is clean at rest. While rowing it locks onto the stroke, so it is reported as coupling, not as a rate. The neural-net EDR (phase 2) should be validated against rest windows or a paced-breathing block.

## Report design
- Brand = eymeric.me (`~/freelance-landing-page/src/styles/tokens.*.css`, `~/Drive/freelance/_brand/`). Two inks (Prussian `#1f4e79` / dark `#63a2d8`, vermilion `#b83a20` / dark `#ea7150`), sumi text on washi `#f4f0e6`, Fraunces display (embedded from `report/assets/`), mono eyebrows, 3px corners.
- Plotly traces carry `meta=<role>`; the page's JS re-inks them from the `--c-<role>` CSS tokens in light and dark. Add shapes with `exclude_empty_subplots=False`, or Plotly silently drops them.
- Every metric tile has a `?` tooltip from `report/narrative.py:GLOSSARY`. Section insights are generated from the numbers there too, so keep the wording honest about uncertainty (breathing while rowing, α1 under ECG noise).
- The cardio-progress markers stored in claude-metrics.json (`watts_at_hr140`, `watts_at_vt1`, `hr_at_alpha1_075`) feed the Progress chart across sessions.

## Layout
`io/` loaders · `align.py` · `segment.py` · `analysis/` (ecg, hrv, hr, rowing, respiration, cardio, recommend) · `io/goals.py` · `history.py` · `analyze.py` orchestrator · `report/` (Plotly figures + Jinja template) · `profile.yaml` (age, resting HR, HRmax → zones).
Tests: `uv run pytest` (synthetic fixtures only; never commit personal data).
