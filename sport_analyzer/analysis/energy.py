"""Energy balance: basal rate, what a session burned, and the day's calorie target.

Everything here is an estimate with a known error bar, so the wording on the page says so:
- BMR (Mifflin-St Jeor 1990) is within ±10 % for most people.
- Everyday activity is a fixed factor (``profile.yaml:activity_factor``), the biggest unknown.
- Rowing burn comes from the flywheel work, which the PM5 measures exactly, through a linear
  cost model: a fixed cost per minute for moving the body on the slide (scales with weight) plus
  ~3.5 kcal of metabolism per kcal of work. That gives ~20 % gross efficiency at 150 W
  (Hagerman 1984) but only ~13 % at 60 W, and matches the Compendium of Physical Activities
  (Ainsworth 2011: 4.8 MET at 50 W, 7 MET at 100 W). A flat efficiency underestimates easy rows.
- The Keytel (2005) HR equation is kept as a cross-check over the same pieces. It reads high for
  beginners, whose HR is high for the workload.

Without a scale the real expenditure can't be learned from the weight trend, so the target is
formula-based. Occasional weigh-ins in the food log switch on a trend line once there are a few.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KCAL_PER_KJ = 1 / 4.184
KCAL_PER_KG_FAT = 7700  # energy in 1 kg of body weight change, mostly fat (Hall 2008 uses ~7700)
ROW_BASE_KCAL_PER_KG_MIN = 0.045  # rest metabolism + moving the body on the slide, unloaded
ROW_KCAL_PER_KCAL_WORK = 3.5  # delta efficiency ~29 %
PROTEIN_G_PER_KG = 1.8  # 1.6-2.2 g/kg protects muscle in a deficit (Morton 2018, Helms 2014)
HARD_DAY_DEFICIT_SHARE = 0.4  # keep only 40 % of the deficit on a hard-session day: fuel the 4x4


@dataclass
class Body:
    weight_kg: float
    height_cm: float
    age: int
    sex: str = "male"
    activity_factor: float = 1.4  # desk work + some walking, training excluded

    @property
    def bmr(self) -> float:
        return mifflin(self.weight_kg, self.height_cm, self.age, self.sex)

    @property
    def everyday(self) -> float:
        """Burn on a day without training."""
        return self.bmr * self.activity_factor


def mifflin(weight_kg: float, height_cm: float, age: float, sex: str | None = "male") -> float:
    s = -161 if str(sex or "").lower().startswith(("f", "w")) else 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s


def keytel_kcal_per_min(hr, weight_kg: float, age: float, sex: str | None = "male"):
    """HR -> gross energy expenditure (Keytel 2005, no VO2max term). Only valid around 90-180 bpm."""
    hr = np.asarray(hr, dtype=float)
    if str(sex or "").lower().startswith(("f", "w")):
        k = -20.4022 + 0.4472 * hr - 0.1263 * weight_kg + 0.074 * age
    else:
        k = -55.0969 + 0.6309 * hr + 0.1988 * weight_kg + 0.2017 * age
    return np.clip(k, 0, None) / 4.184


def rowing_gross_kcal(work_kj: float, work_s: float, weight_kg: float) -> float:
    """Total energy spent while rowing (rest metabolism included)."""
    return ROW_BASE_KCAL_PER_KG_MIN * weight_kg * work_s / 60 + ROW_KCAL_PER_KCAL_WORK * work_kj * KCAL_PER_KJ


def session_energy(work_kj: float, work_s: float, body: Body | None,
                   t=None, hr=None, spans: list[tuple[float, float]] | None = None) -> dict:
    """kcal a session added on top of the everyday burn.

    ``kcal_active`` = gross burn while rowing minus what the body would have burnt in those minutes
    anyway (already inside BMR x activity factor), so days don't count it twice.
    ``kcal_hr`` = Keytel over the same ``spans`` (the pieces), gross, as a cross-check."""
    out: dict = {}
    if body is None:
        return out
    if work_kj and work_s:
        gross = rowing_gross_kcal(work_kj, work_s, body.weight_kg)
        out["kcal_gross_mech"] = gross
        out["kcal_active"] = max(gross - body.everyday / 1440 * work_s / 60, 0.0)
    if t is not None and hr is not None and len(t) > 60:
        t, hr = np.asarray(t, float), np.asarray(hr, float)
        sel = np.isfinite(hr)
        if spans is not None:
            sel &= np.any([(t >= a) & (t < b) for a, b in spans], axis=0) if spans else False
        dt = np.gradient(t) / 60
        if np.any(sel):
            out["kcal_hr"] = float(np.sum(keytel_kcal_per_min(hr[sel], body.weight_kg, body.age, body.sex) * dt[sel]))
    return out


def day_target(body: Body, active_kcal: float, deficit: float, hard: bool = False) -> dict:
    """The day's calorie budget and its parts."""
    tdee = body.everyday + active_kcal
    d = deficit * (HARD_DAY_DEFICIT_SHARE if hard else 1.0)
    target = max(tdee - d, body.bmr * 1.1)  # never plan to eat at or below BMR
    return {"bmr": body.bmr, "everyday": body.everyday, "active": active_kcal, "tdee": tdee,
            "deficit": tdee - target, "target": target,
            "protein_g": PROTEIN_G_PER_KG * body.weight_kg}


def deficit_for_rate(kg_per_week: float) -> float:
    return kg_per_week * KCAL_PER_KG_FAT / 7
