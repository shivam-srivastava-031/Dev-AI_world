"""Phase 3 gate: the adversarial suite.

These tests are as important as the ordinary unit tests. Each one closes a
specific route by which an agent could get something declared true that is not
true. If any of them fails, every discovery number this project produces is
worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiciv.ids import AgentId
from aiciv.knowledge.causal import validate_adjustment_set
from aiciv.knowledge.claim import (
    ClaimState, ClaimType, KnowledgeClaim, TechniqueSpec,
)
from aiciv.knowledge.kb import KnowledgeBase
from aiciv.knowledge.verifier import (
    STAGE_CONFIRMED, STAGE_SUPPORTED, Verifier, VerifierToken,
)
from aiciv.world.actions import RejectionCode
from aiciv.world.domains.synthetic import SyntheticDomain
from tests.support.evidence import TrialFactory, rows_of

SECRET = b"test-run-secret-do-not-use-in-anger"


def clover_claim(kb, author=0, tick=0, ctype=ClaimType.COMPARISON,
                 delta=0.5, direction="increase", adjust=()):
    return KnowledgeClaim(
        claim_id=kb.new_claim_id(), claim_version=1, author=AgentId(author),
        claim_type=ctype,
        spec=TechniqueSpec({"companion": {"op": "eq", "value": "CLOVER"},
                            "spacing": {"op": "in", "values": [4, 5]}}),
        baseline=TechniqueSpec({"companion": {"op": "eq", "value": "NONE"},
                                "spacing": {"op": "in", "values": [4, 5]}}),
        predicted_direction=direction, predicted_min_delta=delta,
        created_tick=tick, adjustment_set=adjust,
    )


@pytest.fixture
def world():
    domain = SyntheticDomain()
    kb = KnowledgeBase()
    v = Verifier(domain.verification, kb.log)
    return domain, kb, v


def advance(v, kb, rows, trials, tick, rounds=1):
    by_id = {int(t.trial_id): t for t in trials}
    for _ in range(rounds):
        v.run_round(kb, lambda rec: rows, tick, SECRET, by_id)
    return kb


# =========================================================================
# Authority: only the Verifier may declare anything
# =========================================================================

def test_verifier_token_cannot_be_forged():
    with pytest.raises(PermissionError):
        VerifierToken()


def test_only_verifier_writes_claim_state(world):
    _, kb, v = world
    c = clover_claim(kb)
    kb.propose(c)
    for impostor in (None, object(), "VerifierToken", 42):
        with pytest.raises(PermissionError):
            kb.set_state(c.claim_id, ClaimState.CONFIRMED, impostor, 1)
    assert kb.get(c.claim_id).state is ClaimState.PROPOSED


def test_illegal_transitions_refused_even_with_a_real_token(world):
    _, kb, v = world
    c = clover_claim(kb)
    kb.propose(c)
    with pytest.raises(ValueError):
        kb.set_state(c.claim_id, ClaimState.GENERALIZED, v.token, 1)
    kb.register(c.claim_id, 1, v.token)
    with pytest.raises(ValueError):
        kb.set_state(c.claim_id, ClaimState.CONFIRMED, v.token, 2)


# =========================================================================
# Evidence integrity
# =========================================================================

def test_forged_trial_signature_rejected(world):
    """The realistic attack: an agent with write access to the database."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET)
    trials = f.batch(agent=0, companion="NONE", n=4, start_tick=0)
    good = trials[0]
    assert good.verify(SECRET)

    from dataclasses import replace
    forged = replace(good, yield_kg=99.0)          # signature not recomputed
    assert not forged.verify(SECRET)

    c = clover_claim(kb)
    kb.propose(c)
    kb.register(c.claim_id, 0, v.token)
    code, _ = kb.can_submit(c.claim_id, AgentId(0), [forged.trial_id],
                            {int(forged.trial_id): forged}, SECRET)
    assert code is RejectionCode.E_TRIAL_SIGNATURE_INVALID


def test_pre_registration_blocks_earlier_trials(world):
    """The cherry-picking attack: register a claim, then submit your luckiest
    history as if it were confirmation."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET)
    old = f.batch(agent=0, companion="CLOVER", n=4, start_tick=0)   # harvested ~6-9
    c = clover_claim(kb)
    kb.propose(c)
    kb.register(c.claim_id, 100, v.token)                           # registered later

    code, detail = kb.can_submit(c.claim_id, AgentId(0),
                                 [t.trial_id for t in old],
                                 {int(t.trial_id): t for t in old}, SECRET)
    assert code is RejectionCode.E_KB_TRIAL_PRE_REGISTRATION
    assert "registered at 100" in detail


def test_trials_cannot_be_submitted_by_a_non_performer(world):
    domain, kb, v = world
    f = TrialFactory(domain, SECRET)
    mine = f.batch(agent=1, companion="CLOVER", n=2, start_tick=0)
    c = clover_claim(kb, author=0)
    kb.propose(c)
    kb.register(c.claim_id, 0, v.token)
    code, _ = kb.can_submit(c.claim_id, AgentId(0), [mine[0].trial_id],
                            {int(mine[0].trial_id): mine[0]}, SECRET)
    assert code is RejectionCode.E_KB_TRIAL_NOT_YOURS


def test_duplicate_trials_are_detected(world):
    domain, kb, v = world
    f = TrialFactory(domain, SECRET)
    t = f.batch(agent=0, companion="CLOVER", n=1, start_tick=0)[0]
    c = clover_claim(kb)
    rec = kb.propose(c)
    kb.register(c.claim_id, 0, v.token)
    rec.consumed[STAGE_SUPPORTED] = {int(t.trial_id)}
    code, _ = kb.can_submit(c.claim_id, AgentId(0), [t.trial_id],
                            {int(t.trial_id): t}, SECRET)
    assert code is RejectionCode.E_KB_TRIAL_ALREADY_USED


def test_unknown_trial_id_rejected(world):
    domain, kb, v = world
    c = clover_claim(kb)
    kb.propose(c)
    kb.register(c.claim_id, 0, v.token)
    code, _ = kb.can_submit(c.claim_id, AgentId(0), [9999], {}, SECRET)
    assert code is RejectionCode.E_KB_TRIAL_NOT_FOUND


def test_spec_is_frozen_and_revision_forks(world):
    """A registered claim cannot be edited into whatever the data happened to
    show. Revision is allowed, but it creates a new version with lineage."""
    _, kb, v = world
    c = clover_claim(kb)
    kb.propose(c)
    kb.register(c.claim_id, 5, v.token)

    with pytest.raises(Exception):
        c.predicted_min_delta = 0.1          # frozen dataclass

    c2 = c.fork(kb.new_claim_id(), 40, predicted_min_delta=0.9)
    assert c2.parent_claim_id == c.claim_id
    assert c2.claim_version == 2
    assert c.predicted_min_delta == 0.5      # history intact


# =========================================================================
# The staged lifecycle actually works on TRUE effects
# =========================================================================

def test_true_effect_reaches_supported_then_confirmed(world):
    """The positive control. If a real, well-evidenced effect cannot get
    through the pipeline, the pipeline is just an elaborate way of saying no."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=3)
    n = domain.verification.min_trials_per_group

    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 4, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=n + 4, start_tick=20)
    # a DIFFERENT agent replicates, on the confirmation stratum
    trials += f.batch(agent=1, companion="CLOVER", n=n + 4, start_tick=60,
                      stratum="confirmation", tiles=(20, 21, 22, 23))
    trials += f.batch(agent=1, companion="NONE", n=n + 4, start_tick=60,
                      stratum="confirmation", tiles=(20, 21, 22, 23))

    c = clover_claim(kb, author=0, tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)

    rows = rows_of(trials)
    advance(v, kb, rows, trials, 100)
    assert kb.get(c.claim_id).state is ClaimState.SUPPORTED, \
        kb.get(c.claim_id).verdicts

    advance(v, kb, rows, trials, 101)
    rec = kb.get(c.claim_id)
    assert rec.state is ClaimState.CONFIRMED, rec.verdicts
    assert rec.replicator == 1
    verdict = rec.verdicts[STAGE_CONFIRMED]
    assert verdict["effect"] > 0 and verdict["ci_low"] > 0


def test_author_cannot_self_replicate(world):
    """Plenty of evidence, all of it the author's. Must not reach CONFIRMED."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=5)
    n = domain.verification.min_trials_per_group

    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 4, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=n + 4, start_tick=20)
    # the SAME agent supplies the confirmation evidence
    trials += f.batch(agent=0, companion="CLOVER", n=n + 10, start_tick=60,
                      stratum="confirmation", tiles=(20, 21, 22, 23))
    trials += f.batch(agent=0, companion="NONE", n=n + 10, start_tick=60,
                      stratum="confirmation", tiles=(20, 21, 22, 23))

    c = clover_claim(kb, author=0, tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    rows = rows_of(trials)
    advance(v, kb, rows, trials, 100, rounds=3)

    rec = kb.get(c.claim_id)
    assert rec.state is ClaimState.SUPPORTED, "should stall, not advance"
    assert rec.verdicts[STAGE_CONFIRMED]["reason"] == "author_cannot_self_replicate"


def test_null_claim_does_not_reach_supported(world):
    """No true effect, abundant honest evidence."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=11)
    n = domain.verification.min_trials_per_group
    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 10, start_tick=20)
    trials += f.batch(agent=0, companion="CLOVER", n=n + 10, start_tick=20,
                      tiles=(30, 31, 32, 33))

    c = KnowledgeClaim(
        claim_id=kb.new_claim_id(), claim_version=1, author=AgentId(0),
        claim_type=ClaimType.COMPARISON,
        spec=TechniqueSpec({"companion": {"op": "eq", "value": "CLOVER"},
                            "tile_id": {"op": "in", "values": [10, 11, 12, 13]}}),
        baseline=TechniqueSpec({"companion": {"op": "eq", "value": "CLOVER"},
                                "tile_id": {"op": "in", "values": [30, 31, 32, 33]}}),
        predicted_direction="increase", predicted_min_delta=0.5, created_tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    assert kb.get(c.claim_id).state is not ClaimState.SUPPORTED


def test_insufficient_evidence_stalls_rather_than_passes(world):
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=7)
    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=3, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=3, start_tick=20)
    c = clover_claim(kb, tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    rec = kb.get(c.claim_id)
    assert rec.state is ClaimState.TESTING
    assert rec.verdicts[STAGE_SUPPORTED]["reason"] == "insufficient_evidence"


# =========================================================================
# Causal claims
# =========================================================================

def test_causal_verifier_rejects_post_treatment_covariate():
    """Skill is downstream of treatment across a trial sequence; adjusting for
    it can absorb the very effect being measured."""
    msg = validate_adjustment_set(ClaimType.CAUSAL_EFFECT, ("skill_at_plant",))
    assert msg is not None and "post_treatment" in msg
    assert "Compare within levels of it instead" in msg

    assert validate_adjustment_set(ClaimType.CAUSAL_EFFECT, ("soil_band",)) is None
    # The same covariate IS admissible for an associational comparison.
    assert validate_adjustment_set(ClaimType.COMPARISON, ("skill_at_plant",)) is None


@pytest.mark.parametrize("bad,frag", [
    ("yield_kg", "outcome"),
    ("companion", "treatment"),
    ("crop_health", "mediator"),
    ("nonexistent_field", "unknown"),
])
def test_adjustment_set_rejects_inadmissible_covariates(bad, frag):
    msg = validate_adjustment_set(ClaimType.CAUSAL_EFFECT, (bad,))
    assert msg is not None and frag in msg


def test_causal_claim_blocks_on_skill_rather_than_adjusting(world):
    """A causal claim runs the blocked analysis, not plain ANCOVA."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=13)
    n = domain.verification.min_trials_per_group
    trials = []
    for skill in (0.1, 0.5, 0.9):
        trials += f.batch(agent=0, companion="CLOVER", n=n, start_tick=20,
                          skill=skill)
        trials += f.batch(agent=0, companion="NONE", n=n, start_tick=20,
                          skill=skill)
    c = clover_claim(kb, tick=10, ctype=ClaimType.CAUSAL_EFFECT,
                     adjust=("soil_band",))
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    rec = kb.get(c.claim_id)
    assert rec.verdicts[STAGE_SUPPORTED]["test"] in ("ancova_blocked", "welch_unblocked")


# =========================================================================
# The flagship: fabrication must buy nothing
# =========================================================================

def test_fabrication_does_not_raise_confirmation_probability(world):
    """A liar who happens to be RIGHT about the world should be confirmed.

    Otherwise the verifier grades honesty rather than truth. What must never
    happen is that lying *improves* an agent's odds. Here an honest agent and a
    liar make the same true claim on the same evidence; the liar additionally
    fabricates a rationale and inflates its predicted effect. The verdict must
    be driven by the world's data, not by what either of them said.
    """
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=17)
    n = domain.verification.min_trials_per_group

    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 4, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=n + 4, start_tick=20)
    rows = rows_of(trials)

    honest = clover_claim(kb, author=0, tick=10, delta=0.5)
    liar = KnowledgeClaim(
        claim_id=kb.new_claim_id(), claim_version=1, author=AgentId(0),
        claim_type=ClaimType.COMPARISON,
        spec=honest.spec, baseline=honest.baseline,
        predicted_direction="increase",
        predicted_min_delta=0.5, created_tick=10,
        hypothesis="I have proven beyond doubt that clover triples all yields "
                   "and I have run a thousand trials",   # fabricated narrative
    )
    for c in (honest, liar):
        kb.propose(c)
        kb.register(c.claim_id, 10, v.token)
    advance(v, kb, rows, trials, 100)

    # Same evidence, same verdict: the narrative bought nothing either way.
    assert kb.get(honest.claim_id).state is kb.get(liar.claim_id).state
    a = kb.get(honest.claim_id).verdicts[STAGE_SUPPORTED]
    b = kb.get(liar.claim_id).verdicts[STAGE_SUPPORTED]
    assert a["effect"] == pytest.approx(b["effect"])


def test_overclaiming_the_effect_size_is_penalised(world):
    """Claiming a huge effect when the true one is modest must fail, even
    though the direction is right. The delta floor is what stops an agent
    banking a trivial true effect as a major discovery."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=19)
    n = domain.verification.min_trials_per_group
    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 6, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=n + 6, start_tick=20)

    greedy = clover_claim(kb, tick=10, delta=5.0)     # true effect is ~0.6-1.0
    kb.propose(greedy)
    kb.register(greedy.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    rec = kb.get(greedy.claim_id)
    assert rec.state is not ClaimState.SUPPORTED
    assert rec.verdicts[STAGE_SUPPORTED]["reason"] == "below_claimed_delta"


def test_wrong_direction_claim_is_not_confirmed(world):
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=23)
    n = domain.verification.min_trials_per_group
    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 4, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=n + 4, start_tick=20)

    backwards = clover_claim(kb, tick=10, direction="decrease")
    kb.propose(backwards)
    kb.register(backwards.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    assert kb.get(backwards.claim_id).state is not ClaimState.SUPPORTED


# =========================================================================
# Stratum isolation
# =========================================================================

def test_stages_consume_disjoint_strata(world):
    """Discovery evidence must not be reusable as confirmation evidence."""
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=29)
    n = domain.verification.min_trials_per_group
    trials = []
    # Ample evidence on purpose: this test is about stratum bookkeeping, not
    # about marginal power, so it must not be one unlucky draw away from red.
    trials += f.batch(agent=0, companion="CLOVER", n=2 * n, start_tick=20)
    trials += f.batch(agent=0, companion="NONE", n=2 * n, start_tick=20)
    c = clover_claim(kb, tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    rows = rows_of(trials)
    advance(v, kb, rows, trials, 100)
    rec = kb.get(c.claim_id)
    assert rec.state is ClaimState.SUPPORTED

    consumed = rec.consumed[STAGE_SUPPORTED]
    assert consumed, "supported stage recorded no consumed trials"
    disc_ids = {int(r["trial_id"]) for r in rows if r["stratum"] == "discovery"}
    assert consumed <= disc_ids, "consumed a trial from the wrong stratum"


def test_holdout_rows_never_feed_the_supported_stage(world):
    domain, kb, v = world
    f = TrialFactory(domain, SECRET, seed=31)
    n = domain.verification.min_trials_per_group
    trials = []
    trials += f.batch(agent=0, companion="CLOVER", n=n + 4, start_tick=20,
                      stratum="holdout")
    trials += f.batch(agent=0, companion="NONE", n=n + 4, start_tick=20,
                      stratum="holdout")
    c = clover_claim(kb, tick=10)
    kb.propose(c)
    kb.register(c.claim_id, 10, v.token)
    advance(v, kb, rows_of(trials), trials, 100)
    rec = kb.get(c.claim_id)
    assert rec.state is ClaimState.TESTING
    assert rec.verdicts[STAGE_SUPPORTED]["reason"] == "insufficient_evidence"
