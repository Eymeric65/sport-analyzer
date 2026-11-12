"""Heart-rate zones, time in zone, recovery and training load."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 5-zone model, fractions of HRmax (Polar / Concept2 convention) or of HR reserve (Karvonen)
ZONE_BOUNDS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
ZONE_NAMES = ["Z1 recovery", "Z2 endurance", "Z3 tempo", "Z4 threshold", "Z5 max"]


@dataclass
class Profile:
    age: int | None = None
    hr_rest: float | None = None
    hr_max: float | None = None
    sex: str | None = None
    weight_kg: float | None = None
    anchor_watts: float = 60.0  # the standardised anchor block (fixed workload every session)
    anchor_spm: int = 20

    @property
    def hr_max_est(self) -> float:
        if self.hr_max:
            return float(self.hr_max)
        if self.age:
            return 208 - 0.7 * self.age  # Tanaka 2001
        return 190.0

    @property
    def method(self) -> str:
        src = "measured" if self.hr_max else (f"Tanaka 208-0.7x{self.age}" if self.age else "default")
        return ("Karvonen (%HR reserve)" if self.hr_rest else "%HRmax") + f", HRmax {self.hr_max_est:.0f} ({src})"

    def bounds(self) -> list[float]:
        hm = self.hr_max_est
        if self.hr_rest:
            return [self.hr_rest + f * (hm - self.hr_rest) for f in ZONE_BOUNDS]
        return [f * hm for f in ZONE_BOUNDS]

    def zone_of(self, hr) -> np.ndarray:
        """0 = below Z1, 1..5 = zones."""
        return np.clip(np.searchsorted(self.bounds(), np.asarray(hr), side="right"), 0, 5)


def time_in_zones(hr: np.ndarray, dt: float, profile: Profile) -> list[float]:
    z = profile.zone_of(hr)
    return [float((z == k).sum() * dt) for k in range(1, 6)]


def hr_recovery(t: np.ndarray, hr: np.ndarray, t_end: float, after_s: float = 60) -> float | None:
    """Drop in HR ``after_s`` seconds after an effort ends (HRR60)."""
    if t_end + after_s > t[-1]:
        return None
    h0 = np.interp(t_end, t, hr)
    h1 = np.interp(t_end + after_s, t, hr)
    return float(h0 - h1)


def trimp(t: np.ndarray, hr: np.ndarray, profile: Profile, hr_rest_fallback: float) -> float:
    """Banister TRIMP (male weighting)."""
    hr_rest = profile.hr_rest or hr_rest_fallback
    hrr = np.clip((hr - hr_rest) / (profile.hr_max_est - hr_rest), 0, 1)
    dt = np.gradient(t) / 60
    return float(np.sum(dt * hrr * 0.64 * np.exp(1.92 * hrr)))
