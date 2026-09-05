"""Synthetic domain: a deceptive landscape with a known optimum.

Purpose: prove the verification machinery works against ground truth we control,
in milliseconds, before pointing it at anything real.

The landscape is deceptive BY CONSTRUCTION. An agent that does the most natural
thing -- optimise one variable at a time starting from "no companion planting"
-- converges to (spacing=3, MARIGOLD) and is stuck: every single-variable move
from there is worse. The global optima at (4, CLOVER) and (2, BEANS) require
moving two variables simultaneously, through a locally worse position.

Two further features give the epistemics engine real material:
  * MARIGOLD flips from helpful to harmful at wide spacing, so "marigold helps"
    is a refutable overgeneralisation an agent can plausibly verify then extend.
  * THISTLE is uniformly terrible except at spacing 5, so "thistle is always
    bad" is a *nearly* true claim that a careful agent should refute.

Every constant here is HIDDEN. Nothing in this module may be imported by
aiciv.agents.*; tests/test_boundary.py enforces that.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ...spec import VerificationSpec
from .base import Domain, ParamSpec, TileContext

COMPANIONS = ("NONE", "CLOVER", "BEANS", "MARIGOLD", "THISTLE")
SPACINGS = (1, 2, 3, 4, 5)
WATERS = (0, 1, 2, 3, 4)
PLANT_DAYS = tuple(range(12))
CYCLE_DAYS = 12

# rows = companion, cols = spacing 1..5
SYNERGY: dict[str, tuple[float, ...]] = {
    "NONE":     (0.00, 0.00, 0.00, 0.00, 0.00),
    "CLOVER":   (-0.30, -0.30, -0.30, 1.00, 1.00),
    "BEANS":    (1.00, 1.00, -0.30, -0.30, -0.30),
    "MARIGOLD": (0.25, 0.25, 0.25, -0.50, -0.50),
    "THISTLE":  (-0.80, -0.80, -0.80, -0.80, 0.40),
}

COMPANION_INDEX = {c: i for i, c in enumerate(COMPANIONS)}

BASE_YIELD = 3.00
PHASE_AMP = 0.60
PHASE_PEAK_DAY = 3
SPACING_CURVATURE = 0.25
SPACING_PEAK = 3
WATER_PENALTY = 0.30
SOIL_SLOPE = 0.25
SOIL_MID = 2
SKILL_FLOOR = 0.60          # a novice realises 60% of mu; an expert 100%
SIGMA = 0.50                # set by tests/test_calibration.py::test_power_analysis
HEALTH_TOLERANCE = 0.75


def water_opt(spacing: int) -> float:
    """Wider spacing wants more water per plot. Interacts with spacing."""
    return 1.0 + 0.75 * (spacing - 1)


class SyntheticDomain(Domain):
    name = "synthetic"
    version = "0.1"

    verification = VerificationSpec(
        min_trials_per_group=16,       # calibrated: see docs/statistics.md
        min_agents=2,
        min_tiles=2,
        alpha=0.01,
        confirmation_alpha=0.05,
        holdout_alpha=0.05,
        min_effect_size=0.5,
        correction="bh",
        max_trials=60,
        min_trials_holdout=12,      # calibrated at alpha=0.05
        balance_mode="adjust",
        calibrated_by="power-analysis-2026-09-05",
    )

    def __init__(self, *, water_hint: bool = True, sigma: float = SIGMA) -> None:
        self.water_hint = water_hint
        self._sigma = sigma

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def param_space(self) -> Mapping[str, ParamSpec]:
        return {
            "spacing": ParamSpec("spacing", "int", SPACINGS, "seed spacing"),
            "plant_day": ParamSpec("plant_day", "int", PLANT_DAYS, "day of the 12-day cycle"),
            "water": ParamSpec("water", "int", WATERS, "water units allocated"),
            "companion": ParamSpec("companion", "categorical", COMPANIONS, "companion planting"),
        }

    # ---- HIDDEN ----------------------------------------------------------

    def true_mu(self, recipe: Mapping[str, Any], tile: TileContext, skill: float) -> float:
        s = int(recipe["spacing"])
        d = int(recipe["plant_day"])
        w = int(recipe["water"])
        c = str(recipe["companion"])
        q = int(tile.soil_band)

        mu = (
            BASE_YIELD
            + PHASE_AMP * math.cos(2 * math.pi * (d - PHASE_PEAK_DAY) / CYCLE_DAYS)
            - SPACING_CURVATURE * (s - SPACING_PEAK) ** 2
            + SYNERGY[c][s - 1]
            - WATER_PENALTY * (w - water_opt(s)) ** 2
            + SOIL_SLOPE * (q - SOIL_MID)
        )
        return mu * (SKILL_FLOOR + (1.0 - SKILL_FLOOR) * skill)

    def crop_health(self, recipe: Mapping[str, Any]) -> str:
        """Deterministic, noiseless. Honest gradient on the water axis ONLY.

        This exists so agents can collapse the water dimension without being
        told anything about the companion x spacing interaction. Disabled in the
        hint-OFF ablation arm.
        """
        if not self.water_hint:
            return "unknown"
        err = int(recipe["water"]) - water_opt(int(recipe["spacing"]))
        if err < -HEALTH_TOLERANCE:
            return "wilted"
        if err > HEALTH_TOLERANCE:
            return "waterlogged"
        return "healthy"

    # ---- METRICS_ONLY ----------------------------------------------------

    def interaction_surface(self) -> dict[str, dict[int, float]]:
        """g(s, c) = spacing base + synergy. The part that carries the trap."""
        return {
            c: {s: -SPACING_CURVATURE * (s - SPACING_PEAK) ** 2 + SYNERGY[c][s - 1]
                for s in SPACINGS}
            for c in COMPANIONS
        }

    # ---- vectorised form (METRICS_ONLY / calibration harness) --------------

    #: SYNERGY as an array indexed [companion_index, spacing_index].
    SYNERGY_ARR = None  # populated at class definition time below

    def mu_batch(
        self,
        spacing,
        plant_day,
        water,
        companion_idx,
        soil_band,
        skill,
    ):
        """Vectorised true_mu over numpy arrays.

        Exists so the null-calibration harness can run tens of thousands of
        replicates in seconds without stepping the simulation. Must stay
        numerically identical to true_mu -- test_mu_batch_matches_scalar.
        """
        import numpy as np

        s = np.asarray(spacing, dtype=float)
        d = np.asarray(plant_day, dtype=float)
        w = np.asarray(water, dtype=float)
        ci = np.asarray(companion_idx, dtype=int)
        q = np.asarray(soil_band, dtype=float)
        k = np.asarray(skill, dtype=float)

        syn = SYNERGY_ARR[ci, s.astype(int) - 1]
        wopt = 1.0 + 0.75 * (s - 1.0)
        mu = (
            BASE_YIELD
            + PHASE_AMP * np.cos(2 * np.pi * (d - PHASE_PEAK_DAY) / CYCLE_DAYS)
            - SPACING_CURVATURE * (s - SPACING_PEAK) ** 2
            + syn
            - WATER_PENALTY * (w - wopt) ** 2
            + SOIL_SLOPE * (q - SOIL_MID)
        )
        return mu * (SKILL_FLOOR + (1.0 - SKILL_FLOOR) * k)


import numpy as _np

SYNERGY_ARR = _np.array([SYNERGY[c] for c in COMPANIONS], dtype=float)
SyntheticDomain.SYNERGY_ARR = SYNERGY_ARR
