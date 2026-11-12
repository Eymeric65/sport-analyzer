"""Energy maths and the food log, on synthetic data."""

import json
from datetime import date

import numpy as np

from sport_analyzer.analysis import energy as E
from sport_analyzer.io.food import load_food
from sport_analyzer.report.nutrition import build_days, render_nutrition, week_summary

BODY = E.Body(weight_kg=70, height_cm=175, age=30, sex="male", activity_factor=1.4)


def test_mifflin_reference_values():
    assert abs(E.mifflin(70, 175, 30, "male") - 1648.75) < 1e-6
    assert abs(E.mifflin(70, 175, 30, "female") - 1482.75) < 1e-6


def test_rowing_cost_model_matches_literature():
    # gross efficiency ~20 % at 150 W, lower on easy rows
    for watts, lo, hi in [(150, 0.19, 0.22), (60, 0.11, 0.15)]:
        gross = E.rowing_gross_kcal(watts * 0.06, 60, 70)  # one minute
        eff = watts * 60 / 4184 / gross
        assert lo < eff < hi, (watts, eff)
    # Compendium: 100 W stationary rowing ~7 MET
    met = E.rowing_gross_kcal(100 * 0.06, 60, 70) / 70 * 60
    assert 6.5 < met < 7.8


def test_session_energy_nets_out_baseline():
    e = E.session_energy(270, 2700, BODY)  # 100 W for 45 min
    base = BODY.everyday / 1440 * 45
    assert abs(e["kcal_active"] - (e["kcal_gross_mech"] - base)) < 1e-6
    assert "kcal_hr" not in e and E.session_energy(270, 2700, None) == {}


def test_keytel_only_over_spans_and_rises_with_hr():
    t = np.arange(0, 1200.0)
    lo = E.session_energy(0, 0, BODY, t, np.full(1200, 120.0), [(0, 600)])["kcal_hr"]
    hi = E.session_energy(0, 0, BODY, t, np.full(1200, 150.0), [(0, 600)])["kcal_hr"]
    full = E.session_energy(0, 0, BODY, t, np.full(1200, 150.0))["kcal_hr"]
    assert hi > lo > 0 and abs(full - 2 * hi) < 1


def test_day_target_trims_deficit_on_hard_days_and_floors_at_bmr():
    easy = E.day_target(BODY, 300, 400, hard=False)
    hard = E.day_target(BODY, 300, 400, hard=True)
    assert abs(easy["target"] - (BODY.everyday + 300 - 400)) < 1e-6
    assert hard["target"] > easy["target"] and abs(hard["deficit"] - 160) < 1e-6
    assert E.day_target(BODY, 0, 5000)["target"] == BODY.bmr * 1.1
    assert abs(E.deficit_for_rate(0.25) - 275) < 1e-6


def _write_day(folder, d, kcal, complete=True, **extra):
    meals = [{"name": "Lunch", "items": "rice, chicken", "kcal": kcal, "protein_g": 40}]
    body = {"date": d, "complete": complete, "meals": meals, **extra}
    (folder / f"{d}.yaml").write_text(json.dumps(body))  # JSON is valid YAML


def test_food_log_balance_and_page(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    food = tmp_path / "food"
    food.mkdir()
    for i, k in enumerate([2000, 2100, 1900, 2200, 2000, 2050, 1950]):
        _write_day(food, f"2026-01-0{i + 1}", k, weight_kg=70 - 0.1 * i)
    _write_day(food, "2026-01-08", 800, complete=False)
    (food / "notes.yaml").write_text("not: [a day")  # ignored
    days = load_food(root)
    assert len(days) == 8 and days[0].kcal == 2000 and days[-1].complete is False

    rows = build_days(BODY, [], days, 275, date(2026, 1, 1), date(2026, 1, 8))
    assert rows[-1]["balance"] is None  # a partial day stays out of the average
    wk = week_summary(rows, date(2026, 1, 8))
    assert wk["n"] == 6  # Jan 2-7 fall in the 7-day window and are complete
    expect = np.mean([2100, 1900, 2200, 2000, 2050, 1950]) - BODY.everyday
    assert abs(wk["balance"] - expect) < 1e-6
    assert abs(wk["kg_per_week"] - expect * 7 / 7700) < 1e-9

    out = render_nutrition(root, today=date(2026, 1, 8))
    html = out.read_text()
    assert "Fuel for the engine" in html and "partial" in html
