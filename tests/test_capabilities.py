"""Phase 9 gate: knowledge becomes capability becomes technology.

The claim this suite has to make good on is narrow and testable: an artifact
built by one agent must measurably expand what ANOTHER agent can attempt. A
"technology" that changes nothing about what is possible is a badge.
"""

from __future__ import annotations

import pytest

import aiciv.agents.policies  # noqa: F401  registers every policy
from aiciv.capabilities.procedure import (
    CAPABILITIES, Artifact, CapabilityId, CapabilityRegistry,
    claim_supports_capability,
)
from aiciv.config import RunConfig
from aiciv.ids import AgentId
from aiciv.knowledge.claim import ClaimType, KnowledgeClaim, TechniqueSpec
from aiciv.world.actions import ActionProposal, Rejection, RejectionCode, Verb
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine
from aiciv.world.validator import validate

TICKS = 1400


def water_claim(author=0, hi=3, lo=0):
    return KnowledgeClaim(
        claim_id=f"clm_w{author}", claim_version=1, author=AgentId(author),
        claim_type=ClaimType.COMPARISON,
        spec=TechniqueSpec({"water": {"op": "eq", "value": hi}}),
        baseline=TechniqueSpec({"water": {"op": "eq", "value": lo}}),
        predicted_direction="increase", predicted_min_delta=0.3,
        created_tick=1)


def companion_claim():
    return KnowledgeClaim(
        claim_id="clm_c", claim_version=1, author=AgentId(0),
        claim_type=ClaimType.COMPARISON,
        spec=TechniqueSpec({"companion": {"op": "eq", "value": "CLOVER"}}),
        baseline=TechniqueSpec({"companion": {"op": "eq", "value": "NONE"}}),
        predicted_direction="increase", predicted_min_delta=0.3,
        created_tick=1)


@pytest.fixture(scope="module")
def built():
    """A full run of the Builder archetype."""
    e = Engine(RunConfig(seed=42, ticks=TICKS, n_agents=5, policy="builder"),
               SyntheticDomain())
    e.run()
    return e


# --- knowledge -> procedure ----------------------------------------------

def test_only_relevant_knowledge_compiles_into_a_procedure():
    """A procedure is compiled from specific verified knowledge, not granted
    for any confirmed claim whatsoever."""
    assert claim_supports_capability(water_claim()) is CapabilityId.IRRIGATION
    assert claim_supports_capability(companion_claim()) is None


def test_compiling_is_idempotent():
    reg = CapabilityRegistry()
    a = reg.compile_procedure(water_claim(), 10)
    b = reg.compile_procedure(water_claim(), 20)
    assert a is b and len(reg.procedures) == 1


# --- procedure + practice + materials -> capability ----------------------

def test_capability_needs_knowledge_skill_and_materials():
    """Holding the knowledge is necessary and not sufficient."""
    reg = CapabilityRegistry()
    spec = CAPABILITIES[CapabilityId.IRRIGATION]

    assert "know" in reg.can_build(0, CapabilityId.IRRIGATION, 1.0, 10)
    reg.grant(0, CapabilityId.IRRIGATION, 1)
    assert reg.can_build(0, CapabilityId.IRRIGATION, 1.0, 10) is None
    assert "practised" in reg.can_build(
        0, CapabilityId.IRRIGATION, spec.min_skill - 0.01, 10)
    assert "spare" in reg.can_build(0, CapabilityId.IRRIGATION, 1.0, 0)


def test_the_world_gates_building_not_the_policy():
    """A policy that tries to build too early gets a hint back, rather than
    the policy quietly deciding for itself what it is allowed to do."""
    e = Engine(RunConfig(seed=42, ticks=20, n_agents=5, policy="random"),
               SyntheticDomain())
    e.run()
    a = e.state.agents[AgentId(0)]
    target = next(t for t in e.state.grid.neighbors(a.x, a.y) if t.arable)

    v = validate(ActionProposal(Verb.BUILD_CHANNEL, {"tile": [target.x, target.y]}),
                 AgentId(0), e.state, e.domain, capabilities=e.capabilities)
    assert isinstance(v, Rejection)
    assert v.code is RejectionCode.E_CAP_NOT_HELD
    assert v.hint


# --- the chain actually fires in a real run ------------------------------

def test_confirmed_knowledge_becomes_a_built_artifact(built):
    """The end-to-end slice: claim -> procedure -> capability -> artifact."""
    rep = built.capabilities.report()
    assert rep["procedures"] > 0, "no confirmed claim compiled into a procedure"
    assert rep["agents_with_capability"], "nobody unlocked the capability"
    assert rep["artifacts"] > 0, "nothing was built"

    kinds = [e["kind"] for e in rep["events"]]
    assert kinds.index("procedure_compiled") < kinds.index("artifact_built"), (
        "something was built before any knowledge was compiled")


def test_artifacts_come_only_from_confirmed_claims(built):
    banked = {str(r.claim.claim_id) for r in built.kb
              if r.state.value in ("confirmed", "generalized")}
    for proc in built.capabilities.procedures.values():
        assert str(proc.from_claim) in banked, (
            "a procedure was compiled from knowledge the world never confirmed")


# --- artifact -> a genuinely changed action space ------------------------

def test_an_artifact_expands_what_a_DIFFERENT_agent_can_attempt():
    """The property that makes it technology rather than a private skill.

    An agent who learned nothing about water, and could not have built the
    channel, can attempt a recipe next to it that it could not attempt away
    from it.
    """
    e = Engine(RunConfig(seed=42, ticks=30, n_agents=5, policy="random"),
               SyntheticDomain())
    e.run()

    stranger = e.state.agents[AgentId(1)]
    stranger.water_stock = 1            # cannot afford a thirsty planting
    tile = next(t for t in e.state.grid.neighbors(stranger.x, stranger.y)
                if t.arable and t.tile_id not in e.state.crops
                and not e.state.is_fallow(t.tile_id))

    thirsty = ActionProposal(Verb.PLANT, {
        "tile": [tile.x, tile.y], "spacing": 5, "water": 4, "companion": "NONE"})

    before = validate(thirsty, AgentId(1), e.state, e.domain,
                      capabilities=e.capabilities)
    assert isinstance(before, Rejection)
    assert before.code is RejectionCode.E_RES_INSUFFICIENT_WATER

    # Someone else builds a channel next door. The stranger holds no
    # capability, no procedure, and no belief about water.
    e.capabilities.build(agent=0, tile_id=int(tile.tile_id), tick=e.state.tick)
    assert not e.capabilities.holds(1, CapabilityId.IRRIGATION)

    after = validate(thirsty, AgentId(1), e.state, e.domain,
                     capabilities=e.capabilities)
    assert not isinstance(after, Rejection), (
        "the channel did not widen what a neighbour could attempt")


def test_the_bonus_is_local_not_global():
    """A channel is a place, not a rule change. Standing far from it must not
    help, or 'built the world' would collapse into 'edited the world'."""
    e = Engine(RunConfig(seed=42, ticks=10, n_agents=5, policy="random"),
               SyntheticDomain())
    e.run()
    art = e.capabilities.build(agent=0, tile_id=0, tick=0)
    grid = e.state.grid
    near = e.capabilities.water_bonus_at(grid, 0, 0)
    far = e.capabilities.water_bonus_at(grid, 15, 15)
    assert near > 0
    assert far == 0
    assert art.radius >= 1


def test_two_channels_do_not_stack_without_limit():
    """Overlapping channels take the best, not the sum: otherwise a corner of
    the map becomes arbitrarily wet and the water axis stops mattering."""
    reg = CapabilityRegistry()

    class FakeGrid:
        def by_id(self, tid):
            class T:
                x, y = 5, 5
            return T()

    reg.artifacts[1] = Artifact(1, "channel", 0, AgentId(0), 0)
    reg.artifacts[2] = Artifact(2, "channel", 1, AgentId(1), 0)
    assert reg.water_bonus_at(FakeGrid(), 5, 5) == Artifact(
        3, "channel", 0, AgentId(0), 0).water_bonus


# --- the covariate bug this slice uncovered ------------------------------

def test_a_claim_is_never_adjusted_for_its_own_variable():
    """Adjusting for the treatment removes the treatment.

    A claim about water, analysed with water in the covariate set, is perfectly
    collinear with its own contrast: the fitted effect comes back as exactly
    0.0 however strong the real effect is. That is a silent total failure that
    looks like the agent being wrong rather than us being wrong, and it blocked
    the entire capability chain until it was found.
    """
    import numpy as np

    from aiciv.knowledge.verifier import _analyse_comparison, claim_variables
    from aiciv.world.domains.synthetic import SyntheticDomain

    claim = water_claim(hi=3, lo=0)
    assert "water" in claim_variables(claim)

    rng = np.random.default_rng(0)
    def rows(water, mu):
        return [{"water": water, "soil_band": int(rng.integers(0, 5)),
                 "plant_day": int(rng.integers(0, 12)),
                 "skill_at_plant": 0.2,
                 "yield_kg": float(mu + rng.normal(0, 0.4))} for _ in range(30)]

    res = _analyse_comparison(claim, rows(3, 4.0), rows(0, 3.0),
                              SyntheticDomain.verification)
    assert res.effect > 0.5, (
        f"the treatment effect was absorbed by its own covariate "
        f"(effect {res.effect})")
