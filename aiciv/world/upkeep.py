"""Per-tick upkeep: growth, metabolism, and the guaranteed-survival floor.

Basic survival is GUARANTEED in v0.1 (docs/protocol.md section 2). Agents can
go hungry, which costs energy and therefore actions, but they cannot die. That
keeps hunger a real selection pressure without letting a run collapse into a
starvation cascade before any knowledge accumulates.
"""

from __future__ import annotations

from ..config import (
    ENERGY_MAX, FOOD_PER_TICK, REPUTATION_MAX, REPUTATION_REGEN,
    REPUTATION_REGEN_EVERY, SURVIVAL_FLOOR, WATER_CARRY_MAX,
)
from ..hashing import q
from .executor import Event, mature_crops
from .state import WorldState


def upkeep(state: WorldState) -> list[Event]:
    events = list(mature_crops(state))

    for aid in state.agent_ids():
        a = state.agents[aid]

        a.food = q(a.food - FOOD_PER_TICK)
        if a.food < 0.0:
            # The survival floor: hunger bites, but nobody starves in v0.1.
            a.food = 0.0
            a.energy = q(max(SURVIVAL_FLOOR, a.energy - 1.0))
            events.append(Event(state.tick, "hunger", int(aid), {}))

        # Water replenishes slowly; standing next to water fills you faster.
        near_water = any(t.terrain.value == "water"
                         for t in state.grid.neighbors(a.x, a.y))
        a.water_stock = min(WATER_CARRY_MAX, a.water_stock + (2 if near_water else 1))

        a.energy = q(min(ENERGY_MAX, a.energy + 0.5))

        # Without regeneration the knowledge base deadlocks once every agent has
        # spent its stake on refuted claims -- observed around tick 150.
        if state.tick % REPUTATION_REGEN_EVERY == 0 and state.tick > 0:
            a.reputation = q(min(REPUTATION_MAX, a.reputation + REPUTATION_REGEN))

    return events
