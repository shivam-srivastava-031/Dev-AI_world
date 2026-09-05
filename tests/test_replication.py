"""Replication kinds, covariate totality, and the shadow verifier."""

from __future__ import annotations

import dataclasses

import pytest

from aiciv.ids import AgentId, ClaimId
from aiciv.knowledge.causal import COVARIATE_CLASS, CovariateClass, block_by_skill
from aiciv.knowledge.replication import (
    CHANNEL_PUBLIC, CHANNEL_TEACH, CommunicationLog, ReplicationKind,
    classify_replication, independence_profile, is_independently_corroborated,
)
from aiciv.knowledge.shadow import ShadowVerifier, adoption_metrics
from aiciv.knowledge.trials import Trial
from aiciv.world.domains.synthetic import SyntheticDomain
from tests.support.evidence import TrialFactory, rows_of

CID = ClaimId("clm_00001")
SECRET = b"secret"


def classify(log, *, first_trial_tick, author=0, replicator=1):
    return classify_replication(
        author=AgentId(author), replicator=AgentId(replicator), claim_id=CID,
        first_matching_trial_tick=first_trial_tick, log=log)


# --- the four kinds -------------------------------------------------------

def test_never_exposed_is_independent_rediscovery():
    assert classify(CommunicationLog(), first_trial_tick=50) \
        is ReplicationKind.INDEPENDENT_REDISCOVERY


def test_trials_predating_exposure_are_blind_replication():
    """Evidence gathered before the hypothesis existed cannot be contaminated
    by it."""
    log = CommunicationLog()
    log.record(80, AgentId(1), CID, CHANNEL_TEACH)
    assert classify(log, first_trial_tick=50) is ReplicationKind.BLIND_REPLICATION


def test_taught_before_running_is_taught_replication():
    log = CommunicationLog()
    log.record(30, AgentId(1), CID, CHANNEL_TEACH)
    assert classify(log, first_trial_tick=50) is ReplicationKind.TAUGHT_REPLICATION


def test_heard_publicly_before_running_is_informed_replication():
    log = CommunicationLog()
    log.record(30, AgentId(1), CID, CHANNEL_PUBLIC)
    assert classify(log, first_trial_tick=50) is ReplicationKind.INFORMED_REPLICATION


def test_earliest_exposure_decides_the_channel():
    """Being told publicly and later taught directly is still 'informed': what
    matters is how the agent first came to know."""
    log = CommunicationLog()
    log.record(40, AgentId(1), CID, CHANNEL_TEACH)
    log.record(20, AgentId(1), CID, CHANNEL_PUBLIC)
    assert classify(log, first_trial_tick=60) is ReplicationKind.INFORMED_REPLICATION


def test_author_cannot_replicate_own_claim():
    assert classify(CommunicationLog(), first_trial_tick=50, replicator=0) is None


def test_only_untainted_kinds_count_as_corroboration():
    """A claim confirmed only by taught replications shows DIFFUSION, which is
    a different result from independent corroboration and must not be reported
    as one."""
    assert is_independently_corroborated(ReplicationKind.INDEPENDENT_REDISCOVERY)
    assert is_independently_corroborated(ReplicationKind.BLIND_REPLICATION)
    assert not is_independently_corroborated(ReplicationKind.TAUGHT_REPLICATION)
    assert not is_independently_corroborated(ReplicationKind.INFORMED_REPLICATION)
    assert not is_independently_corroborated(None)


def test_independence_profile_counts_every_kind():
    profile = independence_profile([
        ReplicationKind.TAUGHT_REPLICATION,
        ReplicationKind.TAUGHT_REPLICATION,
        ReplicationKind.BLIND_REPLICATION,
        None,
    ])
    assert profile["taught_replication"] == 2
    assert profile["blind_replication"] == 1
    assert profile["independent_rediscovery"] == 0
    assert set(profile) == {k.value for k in ReplicationKind}


def test_exposure_log_is_per_agent_and_per_claim():
    log = CommunicationLog()
    log.record(10, AgentId(1), CID, CHANNEL_TEACH)
    assert log.knows(AgentId(1), CID, by_tick=10)
    assert not log.knows(AgentId(1), CID, by_tick=9)
    assert not log.knows(AgentId(2), CID, by_tick=100)
    assert not log.knows(AgentId(1), ClaimId("clm_00002"), by_tick=100)


# --- covariate classification --------------------------------------------

def test_covariate_classification_is_total():
    """Every field a trial carries must be classified.

    Without this, adding a field to Trial silently creates a covariate with no
    causal status, and the post-treatment guard stops being exhaustive.
    """
    fields = {f.name for f in dataclasses.fields(Trial)}
    unclassified = sorted(fields - set(COVARIATE_CLASS))
    assert not unclassified, f"trial fields with no covariate class: {unclassified}"


def test_skill_is_classified_post_treatment():
    """The single most important entry in the table. See causal.py's docstring:
    skill is pre-treatment for one trial but treatment-affected across a
    sequence, so adjusting for it can absorb the effect being measured."""
    assert COVARIATE_CLASS["skill_at_plant"] is CovariateClass.POST_TREATMENT
    assert COVARIATE_CLASS["soil_band"] is CovariateClass.PRE_TREATMENT
    assert COVARIATE_CLASS["crop_health"] is CovariateClass.MEDIATOR
    assert COVARIATE_CLASS["yield_kg"] is CovariateClass.OUTCOME


def test_skill_blocking_uses_fixed_cut_points():
    """Quantile blocks would be a function of treatment assignment and would
    reintroduce the dependence blocking exists to remove."""
    rows = [{"skill_at_plant": s} for s in (0.0, 0.1, 0.4, 0.5, 0.8, 0.99)]
    blocks = block_by_skill(rows, n_blocks=3)
    assert [len(blocks[i]) for i in range(3)] == [2, 2, 2]
    # Shifting the distribution must not move the boundaries.
    skewed = [{"skill_at_plant": s} for s in (0.0, 0.01, 0.02, 0.03)]
    assert len(block_by_skill(skewed, n_blocks=3)[0]) == 4


# --- the shadow verifier --------------------------------------------------

@pytest.fixture(scope="module")
def shadow_rows():
    domain = SyntheticDomain()
    f = TrialFactory(domain, SECRET, seed=101)
    trials = []
    for start in (0, 100):
        trials += f.batch(agent=0, companion="CLOVER", n=40, start_tick=start)
        trials += f.batch(agent=1, companion="NONE", n=40, start_tick=start,
                          tiles=(14, 15, 16, 17))
    return domain, rows_of(trials)


def test_shadow_verifier_finds_a_real_effect_nobody_claimed(shadow_rows):
    """The point of the shadow verifier: evidence exists in the world whether
    or not any agent filed paperwork about it."""
    domain, rows = shadow_rows
    findings = ShadowVerifier(domain.verification).scan(rows)
    assert findings, "shadow verifier found nothing in strongly-signalled data"
    companions = [f for f in findings if f.variable == "companion"]
    assert companions
    best = max(companions, key=lambda f: f.discovery.effect)
    assert best.spec.conditions["companion"]["value"] == "CLOVER"
    assert best.discovery.effect > 0


def test_shadow_verifier_does_not_dredge(shadow_rows):
    """It is held to the standard it measures: hypotheses come from an early
    slice and are tested on a disjoint later slice."""
    domain, rows = shadow_rows
    sv = ShadowVerifier(domain.verification)
    findings = sv.scan(rows)
    for f in findings:
        assert f.stage in ("supported", "confirmed")
    # Pure noise must not produce confirmed findings.
    import numpy as np
    rng = np.random.default_rng(0)
    noise = [dict(r, yield_kg=float(rng.normal(3.0, 0.5))) for r in rows]
    assert not [f for f in ShadowVerifier(domain.verification).scan(noise)
                if f.stage == "confirmed"]


def test_adoption_rate_is_reported_not_assumed():
    assert adoption_metrics(2, 8)["protocol_adoption_rate"] == 0.25
    assert adoption_metrics(0, 5)["protocol_adoption_rate"] == 0.0
    # Nothing discoverable: the ratio is undefined, not zero.
    assert adoption_metrics(0, 0)["protocol_adoption_rate"] is None
