"""Phase 2 gate: the synthetic domain in isolation.

These four tests validate the scientific premise of the whole project before a
single line is built on top of it. If any of them fails, the landscape is
mistuned and no amount of agent engineering will produce a meaningful result.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from aiciv.world.domains.base import TileContext
from aiciv.world.domains.synthetic import (
    COMPANION_INDEX, COMPANIONS, PLANT_DAYS, SPACINGS, WATERS, SyntheticDomain,
)

GLOBAL_OPTIMA = {("CLOVER", 4), ("BEANS", 2)}
GREEDY_TRAP = ("MARIGOLD", 3)


@pytest.fixture(scope="module")
def domain() -> SyntheticDomain:
    return SyntheticDomain()


@pytest.fixture(scope="module")
def surface(domain):
    return domain.interaction_surface()


def _coordinate_ascent(surface, companion: str, spacing: int):
    """One-factor-at-a-time optimisation: the most natural thing an agent does."""
    c, s = companion, spacing
    for _ in range(50):
        moved = False
        best_s = max(SPACINGS, key=lambda x: surface[c][x])
        if surface[c][best_s] > surface[c][s] + 1e-12:
            s, moved = best_s, True
        best_c = max(COMPANIONS, key=lambda x: surface[x][s])
        if surface[best_c][s] > surface[c][s] + 1e-12:
            c, moved = best_c, True
        if not moved:
            break
    return (c, s)


def test_optimum_is_where_we_think(domain, surface):
    """Brute force the whole surface; the global optima must be where designed."""
    best = max(((c, s) for c in COMPANIONS for s in SPACINGS),
               key=lambda cs: surface[cs[0]][cs[1]])
    assert best in GLOBAL_OPTIMA
    top = max(surface[c][s] for c in COMPANIONS for s in SPACINGS)
    winners = {(c, s) for c in COMPANIONS for s in SPACINGS
               if surface[c][s] == pytest.approx(top)}
    assert winners == GLOBAL_OPTIMA


def test_greedy_hillclimb_gets_trapped(surface):
    """The core premise: single-variable optimisation cannot find the optimum.

    An agent starts having never tried a companion, so companion=NONE is the
    honest starting point. From there, every starting spacing must converge to
    the trap. If this ever fails, the landscape has stopped being deceptive and
    'discovery' in this world no longer requires reasoning about interactions.
    """
    for s0 in SPACINGS:
        assert _coordinate_ascent(surface, "NONE", s0) == GREEDY_TRAP

    # And the trap must be a genuine local maximum, not a waypoint.
    c, s = GREEDY_TRAP
    assert max(SPACINGS, key=lambda x: surface[c][x]) == s
    assert max(COMPANIONS, key=lambda x: surface[x][s]) == c

    # It must also be strictly worse than the global optimum, or being trapped
    # would cost nothing and the whole exercise would be pointless.
    top = max(surface[cc][ss] for cc in COMPANIONS for ss in SPACINGS)
    assert top - surface[c][s] >= 0.4


def test_greedy_trap_basin_is_substantial(surface):
    """From arbitrary starts, the trap must capture a real share of agents.

    Not all of them -- a landscape nobody escapes is as uninformative as one
    nobody falls into.
    """
    fixed = [_coordinate_ascent(surface, c, s) for c in COMPANIONS for s in SPACINGS]
    trapped = sum(1 for f in fixed if f not in GLOBAL_OPTIMA)
    assert 0.25 <= trapped / len(fixed) <= 0.75


def test_noise_makes_single_trial_uninformative(domain):
    """A lucky trial must not be proof, at the skill levels where it matters.

    Checked analytically: P(one treatment trial > one control trial) for two
    independent normals is Phi(delta / (sigma*sqrt(2))).
    """
    surface = domain.interaction_surface()
    trap_gap = max(surface[c][s] for c in COMPANIONS for s in SPACINGS) - surface["MARIGOLD"][3]

    for skill in (0.0, 0.3):                      # early run: where discoveries happen
        realised = trap_gap * (0.60 + 0.40 * skill)
        p = norm.cdf(realised / (domain.sigma * np.sqrt(2)))
        assert p < 0.85, f"single trial too informative at skill={skill}: p={p:.3f}"


def test_mu_batch_matches_scalar(domain):
    """The vectorised path feeds the null harness; drift there corrupts every
    calibration number silently."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = int(rng.choice(SPACINGS)); d = int(rng.choice(PLANT_DAYS))
        w = int(rng.choice(WATERS)); c = str(rng.choice(COMPANIONS))
        q = int(rng.integers(0, 5)); k = float(rng.uniform(0, 1))
        scalar = domain.true_mu(
            dict(spacing=s, plant_day=d, water=w, companion=c), TileContext(0, q), k)
        batch = domain.mu_batch([s], [d], [w], [COMPANION_INDEX[c]], [q], [k])[0]
        assert scalar == pytest.approx(batch, abs=1e-12)


def test_water_hint_is_deterministic_and_noiseless(domain):
    r = dict(spacing=3, plant_day=3, water=2, companion="NONE")
    assert {domain.crop_health(r) for _ in range(50)} == {"healthy"}
    assert domain.crop_health({**r, "water": 0}) == "wilted"
    assert domain.crop_health({**r, "water": 4}) == "waterlogged"


def test_water_hint_can_be_disabled_for_ablation():
    assert SyntheticDomain(water_hint=False).crop_health(
        dict(spacing=3, plant_day=3, water=0, companion="NONE")) == "unknown"


def test_recipe_validation_rejects_out_of_range(domain):
    good = dict(spacing=3, plant_day=3, water=2, companion="NONE")
    assert domain.validate_recipe(good) is None
    assert "spacing" in domain.validate_recipe({**good, "spacing": 9})
    assert "companion" in domain.validate_recipe({**good, "companion": "OAK"})
    assert "missing" in domain.validate_recipe({k: v for k, v in good.items() if k != "water"})
    assert "unknown" in domain.validate_recipe({**good, "fertiliser": 1})


def test_parameter_space_size(domain):
    assert len(domain.enumerate_recipes()) == 5 * 12 * 5 * 5 == 1500
