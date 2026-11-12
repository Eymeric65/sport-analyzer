"""Daily food log: ``food/YYYY-MM-DD.yaml`` next to ``goals.md``.

The athlete writes meals in plain words in Joplin (notebook *Food log*, one note per day titled
``YYYY-MM-DD``). Claude reads those notes, estimates each meal and writes this file:

    date: 2026-11-12
    source: joplin <note id>
    complete: true            # false while the day is only partly logged
    meals:
      - name: Breakfast
        items: 2 eggs, 2 slices of toast, coffee with milk
        kcal: 420
        protein_g: 22
        carbs_g: 34
        fat_g: 20
    extra_kcal: 0             # exercise not on the rower (gym, a long walk), net kcal
    extra_note: ""
    weight_kg: null           # optional occasional weigh-in (gym or pharmacy scale), morning if possible
    waist_cm: null            # optional, at the navel, relaxed
    note: ""

A number the athlete wrote in the note wins over Claude's estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

MACROS = ("kcal", "protein_g", "carbs_g", "fat_g")


@dataclass
class FoodDay:
    day: date
    meals: list[dict] = field(default_factory=list)
    complete: bool = True
    extra_kcal: float = 0.0
    extra_note: str = ""
    weight_kg: float | None = None
    waist_cm: float | None = None
    note: str = ""

    def total(self, key: str) -> float | None:
        v = [m[key] for m in self.meals if m.get(key) is not None]
        return float(sum(v)) if v else None

    @property
    def kcal(self) -> float | None:
        return self.total("kcal")


def food_dir(root: Path) -> Path:
    """``root`` is the folder holding the YYYYMMDD session folders; the log sits beside it."""
    return root.parent / "food"


def load_food(root: Path) -> list[FoodDay]:
    d = food_dir(root)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text()) or {}
            day = raw.get("date") or f.stem
            day = day if isinstance(day, date) else date.fromisoformat(str(day))
        except (ValueError, yaml.YAMLError):
            continue
        out.append(FoodDay(
            day=day, meals=[m for m in raw.get("meals") or [] if isinstance(m, dict)],
            complete=bool(raw.get("complete", True)), extra_kcal=float(raw.get("extra_kcal") or 0),
            extra_note=raw.get("extra_note") or "", weight_kg=raw.get("weight_kg"), waist_cm=raw.get("waist_cm"),
            note=raw.get("note") or "",
        ))
    return sorted(out, key=lambda x: x.day)
