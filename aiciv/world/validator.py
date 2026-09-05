"""Validation: the only thing standing between a proposal and the world.

Six ordered stages; the first failure wins and nothing is executed. A rejection
NEVER mutates state -- tests assert the state hash is unchanged after every
rejection code.

The stage order is not arbitrary. Cheap structural checks run before expensive
semantic ones, and the language gate runs before epistemic checks so that a
garbled message is reported as a language problem rather than as a confusing
knowledge-base error.
"""

from __future__ import annotations

from typing import Any, Callable

from ..ids import AgentId, TileId
from .actions import (
    MESSAGE_VERBS, Accepted, ActionProposal, Rejection, RejectionCode, Verb,
)
from .domains.base import Domain
from .state import CropStage, WorldState

# verb -> (required params, optional params)
PARAM_SPEC: dict[Verb, tuple[frozenset[str], frozenset[str]]] = {
    Verb.MOVE: (frozenset({"tile"}), frozenset()),
    Verb.INSPECT_TILE: (frozenset({"tile"}), frozenset()),
    # PLANT is filled in from the domain at validation time; see plant_params().
    Verb.PLANT: (frozenset({"tile"}), frozenset()),
    Verb.TEND: (frozenset({"tile"}), frozenset()),
    Verb.HARVEST: (frozenset({"tile"}), frozenset()),
    Verb.BUILD_CHANNEL: (frozenset({"tile"}), frozenset()),
    Verb.EAT: (frozenset(), frozenset()),
    Verb.REST: (frozenset(), frozenset()),
    Verb.NOOP: (frozenset(), frozenset()),
    Verb.SAY: (frozenset(), frozenset({"to"})),
    Verb.ASK: (frozenset(), frozenset({"to"})),
    Verb.GOAL_SET: (frozenset(), frozenset()),
    Verb.TEACH: (frozenset({"to", "claim_id"}), frozenset()),
    Verb.PREDICT: (frozenset({"tile", "expected_yield"}), frozenset({"ci"})),
    Verb.PROPOSE_CLAIM: (frozenset({"spec", "baseline", "direction", "min_delta"}),
                         frozenset({"claim_type", "adjustment_set"})),
    Verb.REGISTER_CLAIM: (frozenset({"claim_id"}), frozenset()),
    Verb.SUBMIT_TRIALS: (frozenset({"claim_id", "trial_ids", "arm"}), frozenset()),
    Verb.CHALLENGE_CLAIM: (frozenset({"claim_id"}), frozenset()),
    Verb.WITHDRAW_CLAIM: (frozenset({"claim_id"}), frozenset()),
}

#: Verbs that touch a tile, and whether that tile must be reachable.
TILE_VERBS = frozenset({Verb.MOVE, Verb.INSPECT_TILE, Verb.PLANT,
                        Verb.TEND, Verb.HARVEST, Verb.PREDICT,
                        Verb.BUILD_CHANNEL})


def plant_params(domain: Domain) -> frozenset[str]:
    """Parameters PLANT must carry in this world.

    The schedule axis is excluded: it is set by WHEN the agent acts, so asking
    it to supply one would be asking it to choose something it cannot.
    """
    return frozenset(n for n in domain.param_space if n != domain.schedule_axis)


def water_cost(domain: Domain, params: dict) -> int:
    """What planting costs from the agent's own water store.

    Domains express irrigation on their own scale (units, or millimetres), so
    the cost is normalised against that axis' range rather than assumed.
    """
    axis = domain.calibration_axis
    if not axis or axis not in params:
        return 0
    values = domain.param_space[axis].values
    span = max(values) - min(values)
    if span <= 0:
        return 0
    from ..config import WATER_CARRY_MAX
    frac = (float(params[axis]) - min(values)) / span
    return int(round(frac * (WATER_CARRY_MAX // 2)))


def _reject(code: RejectionCode, detail: str = "") -> Rejection:
    return Rejection(code, detail)


def _as_tile(state: WorldState, raw: Any) -> tuple[TileId | None, Rejection | None]:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        return None, _reject(RejectionCode.E_SCHEMA_TYPE,
                             f"tile must be [x, y], got {raw!r}")
    x, y = raw
    if not (isinstance(x, int) and isinstance(y, int)) or isinstance(x, bool):
        return None, _reject(RejectionCode.E_SCHEMA_TYPE,
                             f"tile coordinates must be integers, got {raw!r}")
    if not state.grid.in_bounds(x, y):
        return None, _reject(RejectionCode.E_BOUNDS_TILE_OUT_OF_GRID,
                             f"({x},{y}) is outside the {state.grid.width}x"
                             f"{state.grid.height} world")
    return TileId(y * state.grid.width + x), None


def validate(
    proposal: ActionProposal,
    agent_id: AgentId,
    state: WorldState,
    domain: Domain,
    *,
    language_gate: Callable[[str], Rejection | None] | None = None,
    claimed_tiles: set[TileId] | None = None,
    capabilities=None,
) -> Accepted | Rejection:
    """Validate one proposal against a FROZEN snapshot.

    ``claimed_tiles`` accumulates tiles already taken this tick, so two agents
    planting the same tile is resolved deterministically by ascending agent_id
    in the EXECUTE phase; the loser gets E_SPATIAL_TILE_OCCUPIED.
    """
    agent = state.agents.get(agent_id)
    if agent is None:
        return _reject(RejectionCode.E_SCHEMA_UNKNOWN_ACTION, "no such agent")

    # ---- stage 1: SCHEMA -------------------------------------------------
    if not isinstance(proposal.verb, Verb):
        return _reject(RejectionCode.E_SCHEMA_UNKNOWN_ACTION, str(proposal.verb))
    if proposal.verb not in PARAM_SPEC:
        return _reject(RejectionCode.E_SCHEMA_UNKNOWN_ACTION, proposal.verb.value)

    required, optional = PARAM_SPEC[proposal.verb]
    if proposal.verb is Verb.PLANT:
        required = required | plant_params(domain)
    supplied = set(proposal.params)
    missing = required - supplied
    if missing:
        return _reject(RejectionCode.E_SCHEMA_MISSING_PARAM,
                       f"{proposal.verb.value} needs {sorted(missing)}")
    extra = supplied - required - optional
    if extra:
        return _reject(RejectionCode.E_SCHEMA_EXTRA_PARAM,
                       f"{proposal.verb.value} does not take {sorted(extra)}")

    # ---- stage 2: BOUNDS -------------------------------------------------
    tile_id: TileId | None = None
    if proposal.verb in TILE_VERBS:
        tile_id, rej = _as_tile(state, proposal.params["tile"])
        if rej is not None:
            return rej

    if proposal.verb is Verb.PLANT:
        recipe = {k: proposal.params[k] for k in plant_params(domain)}
        recipe[domain.schedule_axis] = state.day_of_cycle  # committed by WHEN you act
        detail = domain.validate_recipe(recipe)
        if detail is not None:
            return _reject(RejectionCode.E_BOUNDS_PARAM_RANGE, detail)

    if proposal.verb is Verb.PROPOSE_CLAIM:
        md = proposal.params.get("min_delta")
        if not isinstance(md, (int, float)) or isinstance(md, bool) or md <= 0:
            return _reject(RejectionCode.E_BOUNDS_DELTA_NONPOSITIVE,
                           f"min_delta must be > 0, got {md!r}")
        if not proposal.params.get("baseline"):
            return _reject(RejectionCode.E_KB_MISSING_BASELINE,
                           "a claim needs a baseline to be compared against")

    # ---- stage 3: SPATIAL ------------------------------------------------
    if tile_id is not None:
        tile = state.grid.by_id(tile_id)
        if max(abs(tile.x - agent.x), abs(tile.y - agent.y)) > 1:
            return _reject(RejectionCode.E_SPATIAL_NOT_ADJACENT,
                           f"({tile.x},{tile.y}) is not next to you at "
                           f"({agent.x},{agent.y})")
        if proposal.verb in (Verb.MOVE, Verb.PLANT) and not tile.arable:
            return _reject(RejectionCode.E_SPATIAL_TILE_NOT_ARABLE,
                           f"({tile.x},{tile.y}) is {tile.terrain.value}")
        if proposal.verb is Verb.PLANT and claimed_tiles is not None \
                and tile_id in claimed_tiles:
            return _reject(RejectionCode.E_SPATIAL_TILE_OCCUPIED,
                           f"({tile.x},{tile.y}) was taken this tick")

    if proposal.verb in (Verb.TEACH, Verb.SAY, Verb.ASK):
        target = proposal.params.get("to")
        if target is not None:
            if proposal.verb is Verb.TEACH and target == int(agent_id):
                return _reject(RejectionCode.E_TEACH_SELF, "that is you")
            other = state.agents.get(AgentId(target))
            if other is None:
                return _reject(RejectionCode.E_SCHEMA_TYPE,
                               f"no agent {target}")
            if max(abs(other.x - agent.x), abs(other.y - agent.y)) > 1:
                return _reject(RejectionCode.E_SPATIAL_TARGET_TOO_FAR,
                               f"{other.name} is not within earshot")

    # ---- stage 4: RESOURCE / STATE ---------------------------------------
    if agent.energy < 1.0 and proposal.verb not in (Verb.REST, Verb.NOOP, Verb.EAT):
        return _reject(RejectionCode.E_RES_ENERGY,
                       f"energy {agent.energy:.1f} is too low to act")

    if proposal.verb is Verb.PLANT:
        if agent.seeds < 1:
            return _reject(RejectionCode.E_RES_NO_SEEDS, "you carry no seeds")
        if tile_id in state.crops:
            return _reject(RejectionCode.E_RES_TILE_ALREADY_PLANTED,
                           "something already grows there")
        if state.is_fallow(tile_id):
            return _reject(
                RejectionCode.E_RES_TILE_FALLOW,
                f"rested until tick {state.fallow_until[tile_id]}, "
                f"it is {state.tick}")
        if len(agent.plots) >= state.config.max_concurrent_plots:
            return _reject(RejectionCode.E_RES_TOO_MANY_PLOTS,
                           f"you already tend {len(agent.plots)} plots "
                           f"(max {state.config.max_concurrent_plots})")
        cost = water_cost(domain, proposal.params)
        # The built environment counts toward what an agent can water. This
        # is where technology changes the action space: a channel makes
        # recipes affordable that were previously impossible to attempt.
        available = agent.water_stock
        if capabilities is not None:
            available += capabilities.water_bonus_at(
                state.grid, agent.x, agent.y)
        if available < cost:
            return _reject(RejectionCode.E_RES_INSUFFICIENT_WATER,
                           f"you can water {available}, need {cost}")

    if proposal.verb in (Verb.HARVEST, Verb.TEND):
        crop = state.crops.get(tile_id)
        if crop is None:
            return _reject(RejectionCode.E_RES_TILE_NOT_PLANTED,
                           "nothing grows there")
        if proposal.verb is Verb.HARVEST and crop.stage is not CropStage.MATURE:
            ready = crop.planted_tick + state.config.maturation_ticks
            return _reject(RejectionCode.E_RES_CROP_NOT_MATURE,
                           f"ready at tick {ready}, it is {state.tick}")

    if proposal.verb is Verb.EAT and agent.food <= 0:
        return _reject(RejectionCode.E_RES_NO_FOOD, "you have nothing to eat")

    if proposal.verb is Verb.BUILD_CHANNEL:
        from ..capabilities.procedure import CapabilityId
        if capabilities is None:
            return _reject(RejectionCode.E_CAP_NOT_HELD,
                           "nothing can be built in this world")
        if int(tile_id) in capabilities.by_tile:
            return _reject(RejectionCode.E_CAP_ALREADY_BUILT, str(tile_id))
        detail = capabilities.can_build(
            int(agent_id), CapabilityId.IRRIGATION,
            agent.skill_farming, agent.seeds)
        if detail is not None:
            code = (RejectionCode.E_CAP_NOT_HELD
                    if "know" in detail else RejectionCode.E_CAP_INSUFFICIENT)
            return _reject(code, detail)

    # ---- stage 5: LANGUAGE ------------------------------------------------
    if proposal.verb in MESSAGE_VERBS and language_gate is not None:
        rej = language_gate(proposal.message)
        if rej is not None:
            return rej

    # ---- stage 6: SOCIAL / EPISTEMIC --------------------------------------
    if proposal.verb is Verb.TEACH and not state.config.teaching_enabled:
        return _reject(RejectionCode.E_TEACH_DISABLED,
                       "teaching is not possible in this world")

    return Accepted(proposal)
