"""Vectorised honest-evidence generator for the null-calibration harness, plus
world-authored trial synthesis for the verifier tests.

Generates the trials an agent would actually produce, WITHOUT stepping the
simulation. This is what lets the robustness suite run 20,000 replicates in
seconds instead of days (plan section 9).

Two regimes matter:
  controlled   -- the agent fixed plant_day and used the water hint
  uncontrolled -- the agent varied everything; covariates confound the estimate
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from aiciv.knowledge.trials import make_trial
from aiciv.world.domains.base import TileContext
from aiciv.world.domains.synthetic import (
    COMPANION_INDEX, PLANT_DAYS, SPACINGS, WATERS, SyntheticDomain,
)

BEST_DAY = 3  # the phase peak; an agent finds this quickly, it is 1-D and smooth


def draw(
    domain: SyntheticDomain,
    rng: np.random.Generator,
    *,
    companion: str,
    spacings,
    n: int,
    regime: str = "controlled",
    skill_lo: float = 0.0,
    skill_hi: float = 0.3,
):
    """Return (yields, covariate_matrix). Covariates are PRE_TREATMENT + skill."""
    s = rng.choice(np.asarray(spacings), size=n)
    q = rng.integers(0, 5, size=n)
    k = rng.uniform(skill_lo, skill_hi, size=n)

    if regime == "controlled":
        d = np.full(n, BEST_DAY)
        w = np.rint(1.0 + 0.75 * (s - 1)).astype(int)     # via the health hint
    else:
        d = rng.choice(np.asarray(PLANT_DAYS), size=n)
        w = rng.choice(np.asarray(WATERS), size=n)

    ci = np.full(n, COMPANION_INDEX[companion])
    mu = domain.mu_batch(s, d, w, ci, q, k)
    y = np.maximum(0.0, mu + rng.normal(0.0, domain.sigma, size=n))
    cov = np.column_stack([
        q, w, np.sin(2 * np.pi * d / 12), np.cos(2 * np.pi * d / 12), k,
    ])
    return y, cov


# --- world-authored trials, for the verifier tests -----------------------

class TrialFactory:
    """Mints signed trials exactly as the world would.

    Tests must never hand-build a Trial: going through make_trial keeps the
    signature real, so a test that passes proves the production path works.
    """

    def __init__(self, domain: SyntheticDomain, secret: bytes, seed: int = 0):
        self.domain = domain
        self.secret = secret
        self.rng = np.random.default_rng(seed)
        self._next = 1

    def batch(
        self,
        *,
        agent: int,
        companion: str,
        spacings=(4, 5),
        n: int,
        start_tick: int,
        stratum: str = "discovery",
        tiles=(10, 11, 12, 13),
        skill: float = 0.2,
        controlled: bool = True,
    ) -> list:
        out = []
        for i in range(n):
            s = int(self.rng.choice(np.asarray(spacings)))
            tile = int(tiles[i % len(tiles)])
            soil = tile % 5
            day = BEST_DAY if controlled else int(self.rng.integers(0, 12))
            water = (int(round(1.0 + 0.75 * (s - 1))) if controlled
                     else int(self.rng.integers(0, 5)))
            recipe = {"spacing": s, "plant_day": day, "water": water,
                      "companion": companion}
            o = self.domain.evaluate(recipe, TileContext(tile, soil), skill, self.rng)
            out.append(make_trial(
                trial_id=self._next, agent_id=agent, tile_id=tile,
                planted_tick=start_tick + i, harvested_tick=start_tick + i + 6,
                recipe=recipe, soil_band=soil, skill_at_plant=skill,
                yield_kg=o.observed_yield, crop_health=o.crop_health,
                stratum=stratum, true_mu=o.true_mu, secret=self.secret,
            ))
            self._next += 1
        return out


def rows_of(trials) -> list[dict]:
    """Trial objects -> the dict rows the verifier consumes.

    ``flat()`` merges the recipe up to the top level, which is the shape claim
    matching and covariate assembly expect.
    """
    return [t.flat() for t in trials]
