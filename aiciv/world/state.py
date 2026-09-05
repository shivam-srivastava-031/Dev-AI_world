"""Mutable world state, plus the state hash that makes determinism checkable.

Only ``executor.py`` and ``upkeep.py`` may mutate this. Policies never see it --
they get an Observation (agents/observation.py) and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import (
    ENERGY_MAX, REPUTATION_START, RunConfig, WATER_CARRY_MAX,
)
from ..hashing import blake2b_hex, fmt_float
from ..ids import AgentId, TileId
from .grid import Grid, generate_world


class CropStage(str, Enum):
    GROWING = "growing"
    MATURE = "mature"


@dataclass
class Crop:
    tile_id: TileId
    planter: AgentId
    recipe: dict[str, Any]        # the full parameter set, committed at PLANT
    planted_tick: int
    stage: CropStage = CropStage.GROWING

    def hash_parts(self) -> list[Any]:
        return [int(self.tile_id), int(self.planter), self.recipe,
                self.planted_tick, self.stage.value]


@dataclass
class Agent:
    agent_id: AgentId
    name: str
    x: int
    y: int
    food: float = 5.0
    energy: float = ENERGY_MAX
    water_stock: int = WATER_CARRY_MAX
    seeds: int = 20
    skill_farming: float = 0.0
    reputation: float = REPUTATION_START
    alive: bool = True
    plots: set[TileId] = field(default_factory=set)
    cohort: int = 0

    def hash_parts(self) -> list[Any]:
        return [
            int(self.agent_id), self.x, self.y,
            fmt_float(self.food), fmt_float(self.energy),
            self.water_stock, self.seeds,
            fmt_float(self.skill_farming), fmt_float(self.reputation),
            self.alive, sorted(int(p) for p in self.plots),
        ]


@dataclass
class WorldState:
    config: RunConfig
    grid: Grid
    tick: int = 0
    agents: dict[AgentId, Agent] = field(default_factory=dict)
    crops: dict[TileId, Crop] = field(default_factory=dict)
    next_trial_id: int = 1

    # ---- ordering ---------------------------------------------------------

    def agent_ids(self) -> list[AgentId]:
        """ALWAYS iterate agents through this. Dict order must never leak into
        results; conflicts are resolved by ascending agent_id."""
        return sorted(self.agents)

    @property
    def day_of_cycle(self) -> int:
        return self.tick % 12

    # ---- hashing ----------------------------------------------------------

    def state_hash(self) -> str:
        """Covers everything the world engine owns. Asserted by replay_actions.

        Deliberately excludes memory and knowledge-base contents; those are
        covered by full_hash, so a pure-engine replay can be verified without
        reconstructing agent cognition.
        """
        return blake2b_hex(
            self.tick,
            [self.agents[a].hash_parts() for a in self.agent_ids()],
            [self.crops[t].hash_parts() for t in sorted(self.crops, key=int)],
            self.next_trial_id,
        )

    def snapshot(self) -> "WorldState":
        """A frozen copy for the DECIDE phase.

        Every agent decides against the same snapshot, so no agent's decision
        can depend on another's action within the same tick. This is also what
        makes parallelising LLM calls safe later without touching determinism.
        """
        return WorldState(
            config=self.config,
            grid=self.grid,
            tick=self.tick,
            agents={
                a: Agent(
                    agent_id=ag.agent_id, name=ag.name, x=ag.x, y=ag.y,
                    food=ag.food, energy=ag.energy, water_stock=ag.water_stock,
                    seeds=ag.seeds, skill_farming=ag.skill_farming,
                    reputation=ag.reputation, alive=ag.alive,
                    plots=set(ag.plots), cohort=ag.cohort,
                )
                for a, ag in self.agents.items()
            },
            crops={
                t: Crop(tile_id=c.tile_id, planter=c.planter,
                        recipe=dict(c.recipe), planted_tick=c.planted_tick,
                        stage=c.stage)
                for t, c in self.crops.items()
            },
            next_trial_id=self.next_trial_id,
        )


AGENT_NAMES = ("Ada", "Bo", "Cy", "Del", "Eli", "Fen", "Gil", "Hana",
               "Ira", "Jo", "Kit", "Lev")


def initial_state(config: RunConfig) -> WorldState:
    """Deterministic initial state. Pure function of the config."""
    grid = generate_world(config.seed, width=config.width, height=config.height)
    state = WorldState(config=config, grid=grid)

    # Agents are placed on arable tiles spread across the grid, deterministically.
    arable = grid.arable_tiles()
    if len(arable) < config.n_agents:
        raise ValueError("not enough arable tiles for the requested agents")
    stride = max(1, len(arable) // config.n_agents)

    for i in range(config.n_agents):
        tile = arable[(i * stride) % len(arable)]
        aid = AgentId(i)
        state.agents[aid] = Agent(
            agent_id=aid,
            name=AGENT_NAMES[i % len(AGENT_NAMES)],
            x=tile.x, y=tile.y,
        )
    return state
