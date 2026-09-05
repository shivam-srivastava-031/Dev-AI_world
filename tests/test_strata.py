"""The three-stratum invariant.

    agent cannot know stratum
    agent cannot target stratum
    verifier can accumulate enough holdout evidence

The third used to fail in practice: GENERALIZED was defined but unreachable,
which makes it a decorative state rather than a metric. It was fixed at the
WORLD level -- a fallow period after harvest rotates agents across tiles -- and
not by telling anyone anything about strata.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy import stats

import aiciv.agents.policies.scripted  # noqa: F401  registers
from aiciv.config import RunConfig
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine
from aiciv.world.grid import Stratum, generate_world

STRATA = ("discovery", "confirmation", "holdout")


@pytest.fixture(scope="module")
def run():
    """1200 ticks: the measured cost of reaching GENERALIZED under a brute-force
    control policy (docs/statistics.md 4.2). Shared across this module."""
    e = Engine(RunConfig(seed=42, ticks=1200, n_agents=5,
                         policy="scripted_factorial"), SyntheticDomain())
    e.run()
    return e


# --- invariant 1: the agent cannot know the stratum -----------------------

def test_stratum_never_appears_in_an_observation(run):
    from aiciv.agents.observation import build_observation
    for aid in run.state.agent_ids():
        blob = repr(build_observation(run.state, aid).to_dict())
        for word in ("stratum", "discovery", "confirmation", "holdout"):
            assert word not in blob, f"{word!r} leaked into an Observation"


def test_agent_visible_trial_view_hides_stratum(run):
    for t in run.trials[:50]:
        assert "stratum" not in t.agent_view()
        assert "true_mu" not in t.agent_view()
        assert "signature" not in t.agent_view()


# --- invariant 2: the agent cannot target the stratum ---------------------

def test_fallow_is_stratum_independent():
    """The mechanism that moves agents around must not correlate with strata,
    or it would become an indirect channel for targeting them."""
    grid = generate_world(42)
    # Fallow duration is a global constant applied on harvest; the only thing
    # that could correlate is which tiles are arable at all.
    arable = Counter(t.stratum.value for t in grid.arable_tiles())
    allt = Counter(t.stratum.value for t in grid.tiles)
    obs = np.array([arable[s] for s in STRATA], dtype=float)
    exp = np.array([allt[s] for s in STRATA], dtype=float)
    exp = exp / exp.sum() * obs.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    assert 1 - stats.chi2.cdf(chi2, df=2) > 0.05, (
        "terrain removal is correlated with stratum; arable tiles are not a "
        "fair sample of the partition"
    )


def test_farmed_tiles_are_a_fair_sample_of_the_partition(run):
    """If agents could sense strata, the tiles they choose would deviate from
    availability. Measured on DISTINCT tiles: trial counts are inflated by
    revisits and are not independent draws."""
    farmed = {int(t.tile_id) for t in run.trials}
    obs = Counter(run.state.grid.by_id(t).stratum.value for t in farmed)
    avail = Counter(t.stratum.value for t in run.state.grid.arable_tiles())
    tot = sum(avail.values())

    o = np.array([obs.get(s, 0) for s in STRATA], dtype=float)
    e = np.array([avail[s] / tot for s in STRATA]) * o.sum()
    chi2 = float(((o - e) ** 2 / e).sum())
    p = 1 - stats.chi2.cdf(chi2, df=2)
    assert p > 0.01, (
        f"farmed tiles deviate from the available stratum mix (chi2 p={p:.4f}); "
        f"observed {dict(obs)}, expected {dict(zip(STRATA, e.round(1)))}"
    )


# --- invariant 3: the verifier can actually reach GENERALIZED -------------

def test_generalized_is_reachable_without_privileged_information(run):
    """The gate. An honest control policy that never sees a stratum must be
    able to drive a claim all the way to GENERALIZED."""
    gens = [r for r in run.kb if r.state.value == "generalized"]
    assert gens, (
        f"no claim reached GENERALIZED in 1200 ticks; states were "
        f"{ {k: v for k, v in run.kb.counts().items() if v} }"
    )
    for r in gens:
        v = r.verdicts["generalized"]
        assert v["passed"] and v["effect"] > 0
        assert v["ci_low"] > 0, "a generalized claim whose CI includes zero"


def test_generalized_uses_only_holdout_evidence(run):
    """Each stage must consume its own stratum, or 'held on tiles we never
    looked at' is not what happened."""
    from dataclasses import asdict
    rows = [asdict(t) for t in run.trials]
    by_id = {int(r["trial_id"]): r for r in rows}
    for r in run.kb:
        for stage, stratum in (("supported", "discovery"),
                               ("confirmed", "confirmation"),
                               ("generalized", "holdout")):
            for tid in r.consumed.get(stage, set()):
                assert by_id[int(tid)]["stratum"] == stratum, (
                    f"{stage} consumed a {by_id[int(tid)]['stratum']} trial")


def test_stages_never_share_a_trial(run):
    for r in run.kb:
        seen: set[int] = set()
        for stage, ids in r.consumed.items():
            overlap = seen & set(ids)
            assert not overlap, f"{r.claim.claim_id}: trial reused in {stage}"
            seen |= set(ids)


def test_holdout_sample_size_is_calibrated_not_guessed(run):
    """min_trials_holdout is smaller than the discovery-stage n on purpose: the
    holdout stage re-tests an effect already established twice, at a looser
    alpha. It is still a measured value (docs/statistics.md)."""
    v = run.domain.verification
    assert v.min_trials_holdout == 12
    assert v.min_trials_holdout < v.min_trials_per_group
    assert v.holdout_alpha > v.alpha
