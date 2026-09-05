"""Agronomy domain: the hidden function is real published science.

The synthetic domain proves the machinery works against ground truth we
invented. This one asks a harder question: can the civilization rediscover
relationships that actually hold in the real world?

Three standard response forms, composed:

  Reciprocal yield-density   1/w = a + b*rho          (Shinozaki & Kira 1956)
      per-plant weight falls hyperbolically with density; per-AREA yield is
      asymptotic, so a lodging penalty supplies the interior optimum that real
      crops have.

  FAO-33 water production    Ya/Ym = 1 - Ky(1 - ETa/ETm)   (Doorenbos & Kassam)
      Ky is the crop yield response factor; 1.25 is the published value for
      maize, i.e. yield falls faster than water does.

  Mitscherlich nitrogen      Y = A(1 - e^{-c(N + N0)})     (Mitscherlich 1909)
      diminishing returns to fertiliser, with N0 the soil's own supply.

The two interactions are the real ones, and they are what makes this domain
worth searching:

  * Nitrogen response is scaled by the water factor. Fertiliser a crop cannot
    drink does nothing -- so N and irrigation must be raised together.
  * Denser stands draw down soil nitrogen, so the optimal N rises with density.

Because all three forms are closed-form, the true optimum is computable, which
means METRICS can still grade a discovery exactly as it does in the synthetic
world. We know the answer; the agents do not.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ...spec import VerificationSpec
from .base import Domain, ParamSpec, TileContext

# --- agent-controllable parameters ---------------------------------------
DENSITIES = (2, 4, 6, 8, 10)          # plants per square metre
IRRIGATION = (0, 100, 200, 300, 400)  # mm applied over the season
NITROGEN = (0, 50, 100, 150, 200)     # kg N per hectare
SOWING_DAYS = tuple(range(12))
CYCLE_DAYS = 12

# --- reciprocal yield-density (per-plant weight, kg) ----------------------
YD_A = 0.30
YD_B = 0.085
LODGING_CRIT = 6.0          # stands denser than this start to fall over
LODGING_COEF = 0.16
#: Excess nitrogen weakens stems, bringing the lodging threshold forward.
#: This is why the agronomic optimum for N is interior rather than 'as much
#: as you can afford', and it is what couples N to density.
LODGING_N_COEF = 0.020      # effective plants/m2 added per kg N

# --- FAO-33 water ---------------------------------------------------------
KY = 1.25                   # maize, Doorenbos & Kassam
ETM_BASE = 520.0            # mm of maximum evapotranspiration over a season
RAIN_BASE = 240.0
RAIN_AMPLITUDE = 120.0      # sowing into the wet part of the cycle helps
RAIN_PEAK_DAY = 4
#: Water beyond ETm stops helping and starts hurting: waterlogging, anoxia.
#: Without this, irrigation saturates and has no interior optimum at all.
WATERLOG_COEF = 0.55

# --- Mitscherlich nitrogen ------------------------------------------------
MITS_C = 0.017              # per kg N
SOIL_N_BASE = 40.0          # N0: what the soil supplies unaided
SOIL_N_PER_BAND = 12.0
N_DRAWDOWN_PER_PLANT = 4.0  # denser stands deplete soil N -> N x density
#: Nitrate leaches from an over-watered profile, so heavy irrigation wastes
#: fertiliser -- a third real coupling between nitrogen and water.
LEACH_COEF = 0.45

SIGMA = 0.45                # calibrated in docs/statistics.md
HEALTH_TOLERANCE = 0.12     # on the ETa/ETm deficit ratio


def seasonal_rain(sowing_day: int) -> float:
    return RAIN_BASE + RAIN_AMPLITUDE * math.cos(
        2 * math.pi * (sowing_day - RAIN_PEAK_DAY) / CYCLE_DAYS)


def water_supply(irrigation: float, sowing_day: int) -> float:
    return seasonal_rain(sowing_day) + irrigation


def water_factor(irrigation: float, sowing_day: int) -> float:
    """FAO-33 deficit, plus a waterlogging penalty on the wet side.

    Ky > 1 means yield falls faster than water does. The excess term gives
    irrigation an interior optimum instead of a plateau.
    """
    supply = water_supply(irrigation, sowing_day)
    eta = min(ETM_BASE, supply)
    deficit = max(0.0, 1.0 - KY * (1.0 - eta / ETM_BASE))
    excess = max(0.0, supply / ETM_BASE - 1.0)
    return max(0.0, deficit * (1.0 - WATERLOG_COEF * excess ** 2))


def density_yield(density: float, nitrogen: float = 0.0) -> float:
    """Per-area potential: rho * w(rho), less lodging.

    Lodging depends on density AND nitrogen jointly: a heavily fertilised dense
    stand falls over. That joint term is why the best nitrogen rate is not
    simply the highest one available.
    """
    per_plant = 1.0 / (YD_A + YD_B * density)
    effective = density + LODGING_N_COEF * nitrogen
    lodging = LODGING_COEF * max(0.0, effective - LODGING_CRIT) ** 2
    return max(0.0, density * per_plant - lodging)


def nitrogen_factor(nitrogen: float, density: float, soil_band: int,
                    wf: float) -> float:
    """Mitscherlich, scaled by water and shifted by density.

    Both couplings are real: fertiliser needs water to be taken up, and a
    denser stand draws the soil's own nitrogen down faster.
    """
    n0 = max(0.0, SOIL_N_BASE + SOIL_N_PER_BAND * (soil_band - 2)
             - N_DRAWDOWN_PER_PLANT * density)
    available = (nitrogen + n0) * wf          # unusable without water
    ceiling = 1.0 - math.exp(-MITS_C * (NITROGEN[-1] + SOIL_N_BASE))
    return (1.0 - math.exp(-MITS_C * available)) / ceiling


class AgronomyDomain(Domain):
    name = "agronomy"
    version = "0.1"
    sweep_axes = ("nitrogen", "density")
    calibration_axis = "irrigation"

    verification = VerificationSpec(
        min_trials_per_group=20,       # calibrated: see docs/statistics.md
        min_agents=2,
        min_tiles=2,
        alpha=0.01,
        confirmation_alpha=0.05,
        holdout_alpha=0.05,
        min_effect_size=0.5,
        correction="bh",
        max_trials=70,
        balance_mode="adjust",
        min_trials_holdout=14,
        calibrated_by="agronomy-power-analysis-2026-09-05",
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
            "density": ParamSpec("density", "int", DENSITIES,
                                 "plants per square metre"),
            "irrigation": ParamSpec("irrigation", "int", IRRIGATION,
                                    "mm of water applied"),
            "nitrogen": ParamSpec("nitrogen", "int", NITROGEN,
                                  "kg N per hectare"),
            "plant_day": ParamSpec("plant_day", "int", SOWING_DAYS,
                                   "sowing day of the 12-day cycle"),
        }

    # ---- HIDDEN ----------------------------------------------------------

    def true_mu(self, recipe: Mapping[str, Any], tile: TileContext,
                skill: float) -> float:
        rho = float(recipe["density"])
        irr = float(recipe["irrigation"])
        n = float(recipe["nitrogen"])
        day = int(recipe["plant_day"])

        wf = water_factor(irr, day)
        # Nitrate leaches out of an over-watered profile.
        excess = max(0.0, water_supply(irr, day) / ETM_BASE - 1.0)
        n_eff = n * max(0.0, 1.0 - LEACH_COEF * excess)

        nf = nitrogen_factor(n_eff, rho, tile.soil_band, wf)
        mu = density_yield(rho, n_eff) * wf * nf
        return mu * (0.60 + 0.40 * skill)

    def crop_health(self, recipe: Mapping[str, Any]) -> str:
        """Deterministic, noiseless signal on the WATER axis only.

        Same bargain as the synthetic domain: the agent gets an honest gradient
        on one dimension and learns nothing about the nitrogen x water or
        nitrogen x density interactions, which are the parts worth discovering.
        """
        if not self.water_hint:
            return "unknown"
        eta = min(ETM_BASE, seasonal_rain(int(recipe["plant_day"]))
                  + float(recipe["irrigation"]))
        deficit = 1.0 - eta / ETM_BASE
        if deficit > HEALTH_TOLERANCE:
            return "wilted"
        if float(recipe["irrigation"]) > 0 and \
                seasonal_rain(int(recipe["plant_day"])) + float(recipe["irrigation"]) \
                > ETM_BASE * 1.25:
            return "waterlogged"
        return "healthy"

    # ---- METRICS_ONLY ----------------------------------------------------

    def mu_batch(self, density, irrigation, nitrogen, plant_day, soil_band, skill):
        """Vectorised true_mu, for the null-calibration harness."""
        rho = np.asarray(density, dtype=float)
        irr = np.asarray(irrigation, dtype=float)
        n = np.asarray(nitrogen, dtype=float)
        d = np.asarray(plant_day, dtype=float)
        q = np.asarray(soil_band, dtype=float)
        k = np.asarray(skill, dtype=float)

        rain = RAIN_BASE + RAIN_AMPLITUDE * np.cos(
            2 * np.pi * (d - RAIN_PEAK_DAY) / CYCLE_DAYS)
        supply = rain + irr
        eta = np.minimum(ETM_BASE, supply)
        deficit = np.maximum(0.0, 1.0 - KY * (1.0 - eta / ETM_BASE))
        excess = np.maximum(0.0, supply / ETM_BASE - 1.0)
        wf = np.maximum(0.0, deficit * (1.0 - WATERLOG_COEF * excess ** 2))

        n_eff = n * np.maximum(0.0, 1.0 - LEACH_COEF * excess)
        n0 = np.maximum(0.0, SOIL_N_BASE + SOIL_N_PER_BAND * (q - 2)
                        - N_DRAWDOWN_PER_PLANT * rho)
        ceiling = 1.0 - math.exp(-MITS_C * (NITROGEN[-1] + SOIL_N_BASE))
        nf = (1.0 - np.exp(-MITS_C * (n_eff + n0) * wf)) / ceiling

        per_plant = 1.0 / (YD_A + YD_B * rho)
        effective = rho + LODGING_N_COEF * n_eff
        lodging = LODGING_COEF * np.maximum(0.0, effective - LODGING_CRIT) ** 2
        yd = np.maximum(0.0, rho * per_plant - lodging)

        return yd * wf * nf * (0.60 + 0.40 * k)
