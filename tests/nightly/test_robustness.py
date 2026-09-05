"""Statistical acceptance -- the nightly robustness tier.

Not run in CI (``norecursedirs`` in pyproject). Run explicitly:

    python -m pytest tests/nightly -q

This tier answers the two questions the fast tier cannot resolve:

  1. Is each single-stage test calibrated on nominal? (two-sided, 20 000 reps)
  2. What is the FULL-PIPELINE false-confirmation rate, i.e. the probability
     that a claim with no true effect reaches CONFIRMED?

Question 2 is the number the report is written against. It is expected around
alpha * confirmation_alpha = 0.01 * 0.05 = 5e-4, which cannot be characterised
at 2000 replicates (SE there is 0.05 -- useless). At 20 000 replicates the SE at
a true 5e-4 is about 1.6e-4, which is enough to bound it.

None of this steps the simulation. It uses the verifier plus the vectorised
evidence generator, which is the only reason a rare event is measurable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiciv.knowledge.analysis import ancova, clopper_pearson, welch
from aiciv.world.domains.synthetic import SyntheticDomain
from tests.support.evidence import draw

REPS_CALIBRATION = 20_000
REPS_PIPELINE = 20_000
NOMINAL = 0.05
PIPELINE_CEILING = 2e-3      # hard ceiling; the point estimate should be ~5e-4


@pytest.fixture(scope="module")
def domain():
    return SyntheticDomain()


@pytest.mark.parametrize("use_ancova", [False, True], ids=["welch", "ancova"])
def test_single_stage_calibration_two_sided(domain, use_ancova):
    """The precise calibration. At 20 000 reps the CI must contain nominal."""
    n = 16
    hits = 0
    for i in range(REPS_CALIBRATION):
        rng = np.random.default_rng(7 * 10_000_019 + i)
        a, ca = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)
        b, cb = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)
        r = (ancova(np.concatenate([a, b]), np.array([1] * n + [0] * n),
                    np.vstack([ca, cb]))
             if use_ancova else welch(a, b))
        if r.p_value < NOMINAL:
            hits += 1

    lo, hi = clopper_pearson(hits, REPS_CALIBRATION)
    print(f"\nsingle-stage FPR = {hits/REPS_CALIBRATION:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    assert lo <= NOMINAL <= hi, (
        f"FPR {hits/REPS_CALIBRATION:.4f} CI [{lo:.4f}, {hi:.4f}] excludes "
        f"nominal {NOMINAL}"
    )


def test_full_pipeline_false_confirmation_rate(domain):
    """SUPPORTED -> CONFIRMED on a claim with no true effect.

    Stage 1 (SUPPORTED): the author's discovery-stratum evidence must clear
    alpha and the effect-size floor.
    Stage 2 (CONFIRMED): a DIFFERENT agent's confirmation-stratum evidence must
    independently clear confirmation_alpha in the same direction.

    The conjunction is what makes the knowledge base robust to an agent
    spamming hypotheses. This test measures that robustness rather than
    assuming it.
    """
    v = domain.verification
    n = v.min_trials_per_group
    supported = confirmed = 0

    for i in range(REPS_PIPELINE):
        rng = np.random.default_rng(31 * 10_000_019 + i)
        # No true effect anywhere: every arm is the same condition.
        a, ca = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)
        b, cb = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)

        s1 = ancova(np.concatenate([a, b]), np.array([1] * n + [0] * n),
                    np.vstack([ca, cb]))
        if not (s1.p_value < v.alpha and s1.hedges_g >= v.min_effect_size):
            continue
        supported += 1

        # A different agent replicates on fresh evidence.
        a2, ca2 = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)
        b2, cb2 = draw(domain, rng, companion="CLOVER", spacings=(4, 5), n=n)
        s2 = ancova(np.concatenate([a2, b2]), np.array([1] * n + [0] * n),
                    np.vstack([ca2, cb2]))
        if s2.p_value < v.confirmation_alpha and s2.effect > 0:
            confirmed += 1

    s_rate = supported / REPS_PIPELINE
    c_rate = confirmed / REPS_PIPELINE
    lo, hi = clopper_pearson(confirmed, REPS_PIPELINE)

    print(f"\nfalse SUPPORTED  = {s_rate:.5f}  ({supported}/{REPS_PIPELINE})")
    print(f"false CONFIRMED  = {c_rate:.5f}  95% CI [{lo:.5f}, {hi:.5f}]")
    print(f"expected approx alpha*conf_alpha = {v.alpha * v.confirmation_alpha:.5f}")

    assert hi < PIPELINE_CEILING, (
        f"false-confirmation rate CI upper bound {hi:.5f} exceeds the "
        f"pre-registered ceiling {PIPELINE_CEILING}"
    )
    # The replication stage must actually be doing work.
    assert c_rate < s_rate / 5, (
        "requiring independent replication barely reduced the false rate; "
        "the confirmation stage is not filtering"
    )
