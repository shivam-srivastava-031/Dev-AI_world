"""Execution: the ONLY module permitted to mutate WorldState.

Every mutation happens here, after validation has already passed. Keeping this
in one place is what makes "a rejected action never changes the world" a
checkable property rather than a hope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import (
    ENERGY_MAX, ENERGY_PER_ACTION, ENERGY_PER_REST, SKILL_PER_HARVEST,
)
from ..hashing import q
from ..ids import AgentId, TileId
from ..knowledge.trials import Trial, make_trial
from .actions import ActionProposal, Verb
from .domains.base import Domain, TileContext
from .state import Crop, CropStage, WorldState


@dataclass
class Event:
    tick: int
    kind: str
    agent_id: int
    payload: dict[str, Any]


def execute(
    proposal: ActionProposal,
    agent_id: AgentId,
    state: WorldState,
    domain: Domain,
    rng: np.random.Generator,
    *,
    run_secret: bytes,
) -> tuple[list[Event], list[Trial]]:
    """Apply a validated proposal. Returns (events, newly-recorded trials)."""
    agent = state.agents[agent_id]
    events: list[Event] = []
    trials: list[Trial] = []
    verb = proposal.verb

    def ev(kind: str, **payload: Any) -> None:
        events.append(Event(state.tick, kind, int(agent_id), payload))

    # Every action but resting costs energy.
    if verb is not Verb.REST:
        agent.energy = q(max(0.0, agent.energy - ENERGY_PER_ACTION))

    if verb is Verb.MOVE:
        tile = state.grid.by_id(_tile(state, proposal))
        agent.x, agent.y = tile.x, tile.y
        ev("move", x=tile.x, y=tile.y)

    elif verb is Verb.INSPECT_TILE:
        tile = state.grid.by_id(_tile(state, proposal))
        # AGENT_OBSERVABLE: soil band is revealed, stratum is NOT.
        ev("inspect", tile_id=int(tile.tile_id), soil_band=tile.soil_band)

    elif verb is Verb.PLANT:
        tid = _tile(state, proposal)
        tile = state.grid.by_id(tid)
        from .validator import plant_params, water_cost
        recipe = {k: proposal.params[k] for k in plant_params(domain)}
        recipe[domain.schedule_axis] = state.day_of_cycle
        agent.seeds -= 1
        agent.water_stock -= water_cost(domain, proposal.params)
        agent.plots.add(tid)
        state.crops[tid] = Crop(
            tile_id=tid, planter=agent_id, recipe=recipe,
            planted_tick=state.tick,
        )
        # skill_at_plant is stamped now: PRE_TREATMENT for this trial.
        state.crops[tid].recipe["_skill_at_plant"] = q(agent.skill_farming)
        ev("plant", tile_id=int(tid), **recipe)

    elif verb is Verb.BUILD_CHANNEL:
        tid = _tile(state, proposal)
        agent.seeds -= 2
        ev("build_channel", tile_id=int(tid))

    elif verb is Verb.TEND:
        ev("tend", tile_id=int(_tile(state, proposal)))

    elif verb is Verb.HARVEST:
        tid = _tile(state, proposal)
        tile = state.grid.by_id(tid)
        crop = state.crops.pop(tid)
        agent.plots.discard(tid)
        # The tile rests. Uniform across strata: this rotates agents around
        # the map without revealing which tiles count for which stage.
        state.fallow_until[tid] = state.tick + state.config.fallow_ticks

        recipe = {k: v for k, v in crop.recipe.items() if not k.startswith("_")}
        skill_at_plant = float(crop.recipe.get("_skill_at_plant", 0.0))

        outcome = domain.evaluate(
            recipe, TileContext(int(tid), tile.soil_band), skill_at_plant,
            # Key-derived: the same plot planted the same way yields the same
            # thing regardless of what else happened this tick.
            rng,
        )

        trial = make_trial(
            trial_id=state.next_trial_id,
            agent_id=int(agent_id),
            tile_id=int(tid),
            planted_tick=crop.planted_tick,
            harvested_tick=state.tick,
            recipe=recipe,
            soil_band=tile.soil_band,
            skill_at_plant=skill_at_plant,
            yield_kg=outcome.observed_yield,
            crop_health=outcome.crop_health,
            stratum=tile.stratum.value,
            true_mu=outcome.true_mu,
            secret=run_secret,
        )
        state.next_trial_id += 1
        trials.append(trial)

        agent.food = q(agent.food + outcome.observed_yield)
        agent.seeds += 2                      # a harvest returns seed
        agent.skill_farming = q(min(1.0, agent.skill_farming + SKILL_PER_HARVEST))
        # NOTE: crop_health and yield are PUBLIC; true_mu is METRICS_ONLY and
        # deliberately absent from the event payload.
        ev("harvest", tile_id=int(tid), trial_id=int(trial.trial_id),
           yield_kg=trial.yield_kg, crop_health=trial.crop_health)

    elif verb is Verb.EAT:
        eaten = min(1.0, agent.food)
        agent.food = q(agent.food - eaten)
        agent.energy = q(min(ENERGY_MAX, agent.energy + 2.0 * eaten))
        ev("eat", amount=eaten)

    elif verb is Verb.REST:
        agent.energy = q(min(ENERGY_MAX, agent.energy + ENERGY_PER_REST))
        ev("rest", energy=agent.energy)

    elif verb in (Verb.SAY, Verb.ASK):
        ev(verb.value.lower(), to=proposal.params.get("to"),
           message=proposal.message)

    elif verb is Verb.GOAL_SET:
        # Logged, never enforced, never scored. See docs/protocol.md 2.1.
        ev("goal_set", text=proposal.message)

    elif verb is Verb.NOOP:
        ev("noop")

    else:
        # Epistemic verbs are handled by the knowledge base in a later phase.
        ev("unhandled", verb=verb.value)

    return events, trials


def _tile(state: WorldState, proposal: ActionProposal) -> TileId:
    x, y = proposal.params["tile"]
    return TileId(y * state.grid.width + x)


def mature_crops(state: WorldState) -> list[Event]:
    """UPKEEP: advance crop stages. Deterministic, order-independent."""
    events: list[Event] = []
    for tid in sorted(state.crops, key=int):
        crop = state.crops[tid]
        if crop.stage is CropStage.GROWING and \
                state.tick - crop.planted_tick >= state.config.maturation_ticks:
            crop.stage = CropStage.MATURE
            events.append(Event(state.tick, "crop_mature", int(crop.planter),
                                {"tile_id": int(tid)}))
    return events
