"""Phase 6 gate: the agronomy domain.

The synthetic domain proves the machinery works against ground truth we
invented. This one asks whether the same machinery works when the hidden
function is real published science, so that a discovery corresponds to an
actual agronomic fact rather than to an arbitrary landscape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aiciv.world.domains.agronomy import (
    DENSITIES, IRRIGATION, KY, NITROGEN, AgronomyDomain, density_yield,
    nitrogen_factor, seasonal_rain, water_factor,
)
from aiciv.world.domains.base import TileContext


@pytest.fixture(scope="module")
def domain():
    return AgronomyDomain()


@pytest.fixture(scope="module")
def tile():
    return TileContext(0, 2)


def mu(domain, tile, rho, irr, n, day=4, skill=1.0):
    return domain.true_mu(dict(density=rho, irrigation=irr, nitrogen=n,
                               plant_day=day), tile, skill)


# --- the published forms behave as published ------------------------------

def test_fao33_water_response_is_steeper_than_linear():
    """Ky > 1 means yield falls FASTER than water does.

    Tested on a DRY sowing day. On a wet one, extra irrigation pushes supply
    past ETm and the waterlogging term takes over -- which is correct
    behaviour, but it is not what this test is about.
    """
    assert KY > 1.0
    dry_day = 10
    assert seasonal_rain(dry_day) < seasonal_rain(4)

    adequate = water_factor(300, dry_day)
    deficit = water_factor(100, dry_day)
    assert adequate > deficit > 0

    # The published property: a proportional shortfall in ET costs MORE than
    # the same proportion of yield.
    from aiciv.world.domains.agronomy import ETM_BASE
    supply = seasonal_rain(dry_day) + 100
    et_shortfall = 1.0 - supply / ETM_BASE
    yield_shortfall = 1.0 - deficit
    assert yield_shortfall > et_shortfall


def test_water_has_an_interior_optimum():
    """Real irrigation is not monotone: past ETm you get waterlogging.

    Without this the parameter has no optimum to find and the axis is a
    giveaway rather than a discovery.
    """
    vals = [water_factor(i, 4) for i in IRRIGATION]
    best = int(np.argmax(vals))
    assert 0 < best < len(IRRIGATION) - 1, f"water optimum at a boundary: {vals}"


def test_reciprocal_yield_density_then_lodging():
    """Per-plant weight falls hyperbolically; per-area yield rises then falls."""
    per_area = [density_yield(r) for r in DENSITIES]
    assert per_area[1] > per_area[0]                     # rises at first
    assert per_area[-1] < max(per_area)                  # lodging pulls it back
    assert 0 < int(np.argmax(per_area)) < len(DENSITIES) - 1


def test_mitscherlich_shows_diminishing_returns(domain, tile):
    """Each extra unit of N buys less than the one before it."""
    gains = []
    prev = mu(domain, tile, 6, 200, 0)
    for n in NITROGEN[1:]:
        cur = mu(domain, tile, 6, 200, n)
        gains.append(cur - prev)
        prev = cur
    assert gains[0] > gains[1] > gains[2]


def test_seasonal_rain_is_cyclic():
    assert seasonal_rain(4) > seasonal_rain(10)
    assert seasonal_rain(0) == pytest.approx(seasonal_rain(12))


# --- the interactions are real and findable -------------------------------

def test_nitrogen_needs_water(domain, tile):
    """Fertiliser a crop cannot drink does nothing. The N response must be
    much larger when the crop is watered than when it is not."""
    dry = mu(domain, tile, 6, 0, 200) - mu(domain, tile, 6, 0, 0)
    wet = mu(domain, tile, 6, 200, 200) - mu(domain, tile, 6, 200, 0)
    assert wet > 1.5 * dry


def test_optimal_nitrogen_depends_on_density(domain, tile):
    """The N x density coupling, and it runs the counterintuitive way: a dense
    stand lodges under heavy N, so its best rate is LOWER."""
    best = {}
    for rho in DENSITIES:
        best[rho] = max(NITROGEN, key=lambda n: mu(domain, tile, rho, 200, n))
    assert best[DENSITIES[0]] > best[DENSITIES[-1]], best
    assert len(set(best.values())) >= 3, f"interaction is too flat: {best}"


def test_over_irrigation_wastes_fertiliser(domain, tile):
    """Nitrate leaches from an over-watered profile, so the same N buys less."""
    sensible = mu(domain, tile, 6, 200, 150) - mu(domain, tile, 6, 200, 0)
    drowned = mu(domain, tile, 6, 400, 150) - mu(domain, tile, 6, 400, 0)
    assert sensible > drowned


def test_soil_band_supplies_nitrogen(domain):
    poor = mu(domain, TileContext(0, 0), 6, 200, 0)
    rich = mu(domain, TileContext(0, 4), 6, 200, 0)
    assert rich > poor


# --- METRICS can still grade a discovery exactly --------------------------

def test_true_optimum_is_computable_and_not_degenerate(domain, tile):
    best, best_mu = domain.true_optimum(tile, 1.0)
    assert best_mu > 0
    assert len(domain.enumerate_recipes()) == 5 * 5 * 5 * 12
    # density and nitrogen must both be interior, or those axes are giveaways
    assert best["density"] not in (DENSITIES[0], DENSITIES[-1])
    assert best["nitrogen"] not in (NITROGEN[0], NITROGEN[-1])


def test_mu_batch_matches_scalar(domain, tile):
    """The vectorised path feeds the calibration harness; drift there corrupts
    every calibration number silently."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        rho = int(rng.choice(DENSITIES)); irr = int(rng.choice(IRRIGATION))
        n = int(rng.choice(NITROGEN)); day = int(rng.integers(0, 12))
        q = int(rng.integers(0, 5)); k = float(rng.uniform(0, 1))
        scalar = domain.true_mu(dict(density=rho, irrigation=irr, nitrogen=n,
                                     plant_day=day), TileContext(0, q), k)
        batch = domain.mu_batch([rho], [irr], [n], [day], [q], [k])[0]
        assert scalar == pytest.approx(batch, abs=1e-9)


def test_water_hint_is_deterministic_and_ablatable(domain):
    dry = dict(density=6, irrigation=0, nitrogen=100, plant_day=10)
    assert {domain.crop_health(dry) for _ in range(30)} == {"wilted"}
    assert AgronomyDomain(water_hint=False).crop_health(dry) == "unknown"


def test_verification_spec_is_calibrated(domain):
    v = domain.verification
    assert v.calibrated_by.startswith("agronomy-")
    assert v.min_trials_per_group == 20
    assert v.min_trials_holdout == 14
    # Agronomy is noisier per unit of signal than the synthetic world, so it
    # needs MORE evidence. A spec that matched synthetic's would be a copy,
    # not a calibration.
    from aiciv.world.domains.synthetic import SyntheticDomain
    assert v.min_trials_per_group > SyntheticDomain.verification.min_trials_per_group


def test_domain_runs_in_the_engine():
    """End to end: the same engine, verifier and policies, different physics."""
    import aiciv.agents.policies  # noqa: F401
    from aiciv.config import RunConfig
    from aiciv.world.engine import Engine

    e = Engine(RunConfig(seed=42, ticks=200, n_agents=5, policy="random",
                         domain="agronomy"), AgronomyDomain())
    e.run()
    assert len(e.trials) > 20
    assert all(t.verify(e.run_secret) for t in e.trials)
