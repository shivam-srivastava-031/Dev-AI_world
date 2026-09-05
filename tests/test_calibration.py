"""Statistical acceptance -- the CI tier.

Three tiers exist and are never conflated (docs/statistics.md section 4):

  calibration (here)  -- does the single-stage test hold its nominal alpha?
                         Deliberately loose alpha so 2000 replicates can
                         actually resolve the answer.
  robustness (nightly)-- the full-pipeline false-confirmation rate at 20 000
                         replicates. tests/nightly/test_robustness.py
  acceptance (report) -- pre-registered, stated as the robustness interval.

The harness never steps the simulation: it uses the verifier plus the vectorised
generator, which is the only reason characterising a rare event is feasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiciv.knowledge.analysis import (
    ancova, benjamini_hochberg, clopper_pearson, hedges_g, welch,
)
from aiciv.world.domains.synthetic import SyntheticDomain
from tests.support.evidence import draw

CALIBRATION_ALPHA = 0.05     # loose ON PURPOSE: resolvable at 2000 replicates
N_REPLICATES = 2000


@pytest.fixture(scope="module")
def domain():
    return SyntheticDomain()


def _run(domain, n, treat_c, ctrl_c, reps, alpha, *, use_ancova, g_floor, seed):
    hits = 0
    for i in range(reps):
        rng = np.random.default_rng(seed * 10_000_019 + i)
        a, ca = draw(domain, rng, companion=treat_c, spacings=(4, 5), n=n)
        b, cb = draw(domain, rng, companion=ctrl_c, spacings=(4, 5), n=n)
        if use_ancova:
            r = ancova(np.concatenate([a, b]), np.array([1] * n + [0] * n),
                       np.vstack([ca, cb]))
        else:
            r = welch(a, b)
        if r.p_value < alpha and r.hedges_g >= g_floor:
            hits += 1
    return hits, reps


@pytest.mark.parametrize("use_ancova", [False, True], ids=["welch", "ancova"])
def test_single_stage_is_not_anti_conservative(domain, use_ancova):
    """A truly-null comparison must not be rejected MORE often than alpha.

    Deliberately one-sided. Over-rejection inflates every discovery count
    downstream and is the failure mode that would make "verified, p=0.003" a
    lie. Under-rejection merely costs power and is safe.

    A two-sided "the CI contains nominal" assertion would be wrong here: at
    2000 replicates a 95% CI excludes the true value 5% of the time BY
    CONSTRUCTION, so such a test is flaky by design rather than by accident.
    The precise two-sided calibration lives in the nightly suite, where 20 000
    replicates make the interval tight enough for the question to be answerable
    (measured there: Welch 0.0485 [0.0455, 0.0515], ANCOVA 0.0516
    [0.0486, 0.0548] -- both on nominal).
    """
    hits, reps = _run(domain, 16, "CLOVER", "CLOVER", N_REPLICATES,
                      CALIBRATION_ALPHA, use_ancova=use_ancova, g_floor=-np.inf,
                      seed=7)
    lo, hi = clopper_pearson(hits, reps)

    assert lo <= CALIBRATION_ALPHA, (
        f"single-stage FPR {hits/reps:.4f} CI [{lo:.4f}, {hi:.4f}] is "
        f"significantly ABOVE nominal {CALIBRATION_ALPHA}: the test "
        f"over-rejects and every discovery count is inflated"
    )
    # Sanity floor: a test that essentially never rejects is broken, not safe.
    assert hits / reps > CALIBRATION_ALPHA / 3, (
        f"single-stage FPR {hits/reps:.4f} is implausibly far below nominal; "
        f"the test may not be rejecting at all"
    )


def test_effect_size_floor_is_conservative(domain):
    """Adding the g floor may only ever lower the false-positive rate."""
    bare, reps = _run(domain, 16, "CLOVER", "CLOVER", 1000, CALIBRATION_ALPHA,
                      use_ancova=True, g_floor=-np.inf, seed=11)
    floored, _ = _run(domain, 16, "CLOVER", "CLOVER", 1000, CALIBRATION_ALPHA,
                      use_ancova=True, g_floor=0.5, seed=11)
    assert floored <= bare


def test_calibrated_n_achieves_target_power(domain):
    """The domain's min_trials_per_group must still deliver >0.8 power.

    Guards the calibration in docs/statistics.md against silent drift: change
    sigma or the skill multiplier and this fails rather than quietly degrading.
    """
    n = domain.verification.min_trials_per_group
    assert n == 16, "min_trials_per_group changed; re-run the calibration"
    hits, reps = _run(domain, n, "CLOVER", "NONE", 1200, 0.01,
                      use_ancova=True, g_floor=0.5, seed=13)
    lo, _ = clopper_pearson(hits, reps)
    assert lo > 0.78, f"power {hits/reps:.3f} CI lower {lo:.3f} below target"


def test_uncontrolled_trials_are_much_weaker(domain):
    """Confounded evidence must not verify as easily as controlled evidence.

    This gap is the design pressure toward controlled comparison. If it ever
    closes, agents can verify by flailing and the experiment measures nothing.
    """
    n = 16
    ctrl_hits, reps = _run(domain, n, "CLOVER", "NONE", 800, 0.01,
                           use_ancova=True, g_floor=0.5, seed=17)

    unc = 0
    for i in range(800):
        rng = np.random.default_rng(19 * 10_000_019 + i)
        a, ca = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n,
                     regime="uncontrolled")
        b, cb = draw(domain, rng, companion="NONE", spacings=(4, 5), n=n,
                     regime="uncontrolled")
        r = ancova(np.concatenate([a, b]), np.array([1] * n + [0] * n),
                   np.vstack([ca, cb]))
        if r.p_value < 0.01 and r.hedges_g >= 0.5:
            unc += 1
    assert unc / 800 < 0.6 * (ctrl_hits / reps)


# --- primitives -----------------------------------------------------------

def test_welch_reports_an_interval_not_just_a_p_value():
    rng = np.random.default_rng(0)
    r = welch(rng.normal(1.0, 0.5, 20), rng.normal(0.0, 0.5, 20))
    assert r.ci_low < r.effect < r.ci_high
    assert r.ci_low > 0                       # a real effect excludes zero
    assert "95% CI" in r.describe() and "Hedges g" in r.describe()


def test_welch_is_one_sided():
    """A claim declares its direction at registration; being wrong in an
    interesting way must not be rewarded."""
    rng = np.random.default_rng(1)
    lo, hi = rng.normal(0.0, 0.5, 30), rng.normal(2.0, 0.5, 30)
    assert welch(hi, lo).p_value < 0.001
    assert welch(lo, hi).p_value > 0.99


def test_ancova_recovers_effect_under_confounding():
    """A covariate correlated with treatment must be adjusted away."""
    rng = np.random.default_rng(3)
    n = 400
    treat = np.array([1] * n + [0] * n)
    soil = np.concatenate([rng.normal(1.5, 1, n), rng.normal(-1.5, 1, n)])
    y = 0.5 * treat + 1.0 * soil + rng.normal(0, 0.5, 2 * n)

    naive = welch(y[treat == 1], y[treat == 0])
    adjusted = ancova(y, treat, soil.reshape(-1, 1))
    assert naive.effect > 2.0                        # badly confounded
    assert adjusted.effect == pytest.approx(0.5, abs=0.15)


def test_hedges_g_small_sample_correction():
    rng = np.random.default_rng(5)
    a, b = rng.normal(1, 1, 5), rng.normal(0, 1, 5)
    assert abs(hedges_g(a, b)) < abs(
        (np.mean(a) - np.mean(b)) /
        np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
    assert hedges_g(np.array([1.0]), np.array([0.0])) == 0.0


def test_benjamini_hochberg_step_up():
    assert benjamini_hochberg([], 0.05) == []
    assert benjamini_hochberg([0.001, 0.9, 0.8], 0.05) == [True, False, False]
    assert benjamini_hochberg([0.9] * 5, 0.05) == [False] * 5
    # step-up: a middling p is carried by a smaller one at the right rank
    assert benjamini_hochberg([0.01, 0.02, 0.03], 0.05) == [True, True, True]


def test_clopper_pearson_edges():
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    lo, hi = clopper_pearson(0, 20000)
    assert lo == 0.0 and hi < 2e-4
    lo, hi = clopper_pearson(20, 20)
    assert hi == 1.0 and lo > 0.8
