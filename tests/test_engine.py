"""Phase 1 gate: the deterministic world engine.

Success condition from the plan: the world runs hundreds of ticks with zero LLM
calls, and the same seed reproduces identical state at every tick.
"""

from __future__ import annotations

import pytest

import aiciv.agents.policies.random_policy  # noqa: F401  (registers the policy)
from aiciv.config import RunConfig
from aiciv.ids import AgentId
from aiciv.world.actions import ActionProposal, Rejection, RejectionCode, Verb
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine
from aiciv.world.validator import validate

TICKS = 120


def build(seed: int = 42, **kw) -> Engine:
    cfg = RunConfig(seed=seed, ticks=TICKS, n_agents=5, policy="random", **kw)
    return Engine(cfg, SyntheticDomain())


@pytest.fixture(scope="module")
def run():
    e = build()
    e.run()
    return e


# --- determinism ----------------------------------------------------------

def test_same_seed_same_hash_sequence():
    a, b = build(42), build(42)
    a.run(); b.run()
    assert a.hash_sequence() == b.hash_sequence()
    assert len(a.hash_sequence()) == TICKS


def test_different_seed_diverges():
    a, b = build(42), build(43)
    a.run(); b.run()
    assert a.hash_sequence() != b.hash_sequence()


def test_same_seed_same_evidence():
    """Trials must match exactly, not just the state hash.

    The state hash excludes trial contents, so a bug that made yields
    nondeterministic could hide behind a matching hash sequence.
    """
    a, b = build(42), build(42)
    a.run(); b.run()
    assert len(a.trials) == len(b.trials)
    for ta, tb in zip(a.trials, b.trials):
        assert ta.agent_view() == tb.agent_view()


def test_agent_processing_order_is_by_id(run):
    for rec in run.history:
        ids = [a for a, _ in rec.proposals]
        assert ids == sorted(ids)


# --- the engine actually does something -----------------------------------

def test_engine_produces_evidence_without_any_llm(run):
    """A floor policy must still close the farming loop, or every downstream
    metric has nothing to measure."""
    assert len(run.trials) > 100, f"only {len(run.trials)} trials in {TICKS} ticks"
    assert run.state.tick == TICKS


def test_no_agent_starves(run):
    """Basic survival is guaranteed in v0.1 (docs/protocol.md section 2)."""
    assert all(a.alive for a in run.state.agents.values())
    assert all(a.energy > 0 for a in run.state.agents.values())


def test_no_livelock_on_a_single_rejection_code(run):
    """One code dominating means agents are stuck in a loop, not learning.

    The first run of this engine spent 1680 of its ticks on rejected EAT
    actions by starving agents -- 18 trials in 400 ticks. This guards that.
    """
    summary = run.rejection_summary()
    total_actions = TICKS * len(run.state.agents)
    for code, count in summary.items():
        assert count < 0.25 * total_actions, (
            f"{code} is {count}/{total_actions} of all actions: likely a livelock"
        )


def test_trials_are_signed_and_verifiable(run):
    assert all(t.verify(run.run_secret) for t in run.trials)
    assert not run.trials[0].verify(b"a different secret")


def test_trials_span_all_three_strata(run):
    """If agents only ever produce discovery-stratum evidence, CONFIRMED and
    GENERALIZED are unreachable and the lifecycle is decorative."""
    strata = {t.stratum for t in run.trials}
    assert strata == {"discovery", "confirmation", "holdout"}


# --- rejections never mutate the world ------------------------------------

@pytest.mark.parametrize("proposal,expected", [
    (ActionProposal(Verb.MOVE, {"tile": [99, 99]}),
     RejectionCode.E_BOUNDS_TILE_OUT_OF_GRID),
    (ActionProposal(Verb.MOVE, {"tile": "over there"}),
     RejectionCode.E_SCHEMA_TYPE),
    (ActionProposal(Verb.PLANT, {"tile": [0, 0]}),
     RejectionCode.E_SCHEMA_MISSING_PARAM),
    (ActionProposal(Verb.REST, {"vigorously": True}),
     RejectionCode.E_SCHEMA_EXTRA_PARAM),
    (ActionProposal(Verb.PLANT, {"tile": [0, 0], "spacing": 99,
                                 "water": 1, "companion": "NONE"}),
     RejectionCode.E_BOUNDS_PARAM_RANGE),
    (ActionProposal(Verb.PLANT, {"tile": [0, 0], "spacing": 3,
                                 "water": 1, "companion": "OAK"}),
     RejectionCode.E_BOUNDS_PARAM_RANGE),
    (ActionProposal(Verb.TEACH, {"to": 0, "claim_id": "clm_00001"}),
     RejectionCode.E_TEACH_SELF),
])
def test_rejection_codes_and_state_is_untouched(proposal, expected):
    e = build()
    e.run(3)
    before = e.state.state_hash()

    verdict = validate(proposal, AgentId(0), e.state, e.domain)
    assert isinstance(verdict, Rejection), f"expected rejection, got {verdict}"
    assert verdict.code is expected, f"got {verdict.code.value}"
    assert verdict.hint                       # a rejection must teach
    assert e.state.state_hash() == before, "a rejected action mutated the world"


def test_harvest_rejections_distinguish_empty_from_immature():
    """Two different failures that both look like "you cannot harvest that".

    Tiles are chosen at runtime: hardcoding [0,0] silently tested the wrong
    branch, because agent 0 starts there and has planted by tick 3.
    """
    e = build()
    e.run(3)
    a = e.state.agents[AgentId(0)]
    before = e.state.state_hash()

    empty = next(t for t in e.state.grid.neighbors(a.x, a.y)
                 if t.arable and t.tile_id not in e.state.crops)
    v = validate(ActionProposal(Verb.HARVEST, {"tile": [empty.x, empty.y]}),
                 AgentId(0), e.state, e.domain)
    assert isinstance(v, Rejection)
    assert v.code is RejectionCode.E_RES_TILE_NOT_PLANTED

    planted = [t for t in [e.state.grid.at(a.x, a.y),
                           *e.state.grid.neighbors(a.x, a.y)]
               if t.tile_id in e.state.crops
               and e.state.crops[t.tile_id].stage.value == "growing"]
    if planted:
        v = validate(ActionProposal(Verb.HARVEST,
                                    {"tile": [planted[0].x, planted[0].y]}),
                     AgentId(0), e.state, e.domain)
        assert isinstance(v, Rejection)
        assert v.code is RejectionCode.E_RES_CROP_NOT_MATURE
        assert "ready at tick" in v.detail        # the hint must be actionable

    assert e.state.state_hash() == before


def test_teaching_can_be_disabled_for_the_control_arm():
    e = build(teaching_enabled=False)
    e.run(2)
    # Put a second agent adjacent so the spatial stage passes and we reach the
    # epistemic stage where the arm's switch lives.
    a0, a1 = e.state.agents[AgentId(0)], e.state.agents[AgentId(1)]
    a1.x, a1.y = a0.x, a0.y
    verdict = validate(
        ActionProposal(Verb.TEACH, {"to": 1, "claim_id": "clm_00001"}),
        AgentId(0), e.state, e.domain)
    assert isinstance(verdict, Rejection)
    assert verdict.code is RejectionCode.E_TEACH_DISABLED


def test_observation_carries_no_hidden_or_metrics_fields(run):
    """build_observation runs assert_no_leak internally; this pins the
    behaviour so a future field addition cannot quietly bypass it."""
    from aiciv.agents.observation import build_observation
    for aid in run.state.agent_ids():
        obs = build_observation(run.state, aid)
        d = obs.to_dict()
        flat = repr(d)
        for banned in ("stratum", "true_mu", "synergy", "signature", "run_secret"):
            assert banned not in flat, f"{banned} leaked into an Observation"
