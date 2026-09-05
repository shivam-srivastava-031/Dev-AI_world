"""Observation: the ONLY read channel from the world into a policy.

If a datum is not here, an agent cannot know it. Every Observation is passed
through ``assert_no_leak`` before it reaches a policy, so adding a field
classified HIDDEN or METRICS_ONLY fails loudly rather than silently
invalidating every run made afterwards.

This module must never import aiciv.world.domains.*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..information import assert_no_leak
from ..world.state import WorldState


@dataclass(frozen=True)
class TileView:
    x: int
    y: int
    terrain: str
    planted: bool
    crop_ready: bool
    soil_band: int | None = None      # None until INSPECT_TILE reveals it


@dataclass(frozen=True)
class Observation:
    tick: int
    day_of_cycle: int
    agent_id: int
    name: str
    x: int
    y: int
    food: float
    energy: float
    water_stock: int
    seeds: int
    skill_farming: float
    reputation: float
    plots: list[int]
    nearby_tiles: list[dict[str, Any]]
    nearby_agents: list[dict[str, Any]]
    my_plots: list[dict[str, Any]] = field(default_factory=list)
    my_claims: list[dict[str, Any]] = field(default_factory=list)
    public_claims: list[dict[str, Any]] = field(default_factory=list)
    last_action_result: dict[str, Any] | None = None
    recent_harvests: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_observation(
    state: WorldState,
    agent_id: int,
    *,
    known_soil: dict[int, int] | None = None,
    my_claims: list[dict[str, Any]] | None = None,
    public_claims: list[dict[str, Any]] | None = None,
    last_result: dict[str, Any] | None = None,
    recent_harvests: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> Observation:
    """Project the frozen snapshot down to what one agent may perceive.

    ``known_soil`` holds tiles this agent has actually INSPECTed; soil band is
    AGENT_OBSERVABLE, not PUBLIC, so it is withheld until the agent spends an
    action to look.
    """
    agent = state.agents[agent_id]
    known_soil = known_soil or {}

    tiles: list[dict[str, Any]] = []
    for tile in [state.grid.at(agent.x, agent.y), *state.grid.neighbors(agent.x, agent.y)]:
        crop = state.crops.get(tile.tile_id)
        tiles.append({
            "tile": [tile.x, tile.y],
            "terrain": tile.terrain.value,
            "planted": crop is not None,
            "crop_ready": crop is not None and crop.stage.value == "mature",
            "mine": crop is not None and int(crop.planter) == int(agent_id),
            "soil_band": known_soil.get(int(tile.tile_id)),
        })

    others: list[dict[str, Any]] = []
    for other_id in state.agent_ids():
        if int(other_id) == int(agent_id):
            continue
        other = state.agents[other_id]
        if max(abs(other.x - agent.x), abs(other.y - agent.y)) <= 1:
            others.append({"agent_id": int(other_id), "name": other.name,
                           "tile": [other.x, other.y]})

    # An agent always knows where it planted and whether it is ready; that is
    # its own crop, not privileged information about the world.
    my_plots = []
    for tid in sorted(agent.plots, key=int):
        crop = state.crops.get(tid)
        if crop is None:
            continue
        tile = state.grid.by_id(tid)
        my_plots.append({
            "tile": [tile.x, tile.y],
            "ready": crop.stage.value == "mature",
            "planted_tick": crop.planted_tick,
        })

    obs = Observation(
        tick=state.tick,
        day_of_cycle=state.day_of_cycle,
        agent_id=int(agent_id),
        name=agent.name,
        x=agent.x, y=agent.y,
        food=agent.food, energy=agent.energy,
        water_stock=agent.water_stock, seeds=agent.seeds,
        skill_farming=agent.skill_farming, reputation=agent.reputation,
        plots=sorted(int(p) for p in agent.plots),
        nearby_tiles=tiles,
        nearby_agents=others,
        my_plots=my_plots,
        my_claims=list(my_claims or []),
        public_claims=list(public_claims or []),
        last_action_result=last_result,
        recent_harvests=list(recent_harvests or []),
        messages=list(messages or []),
    )

    # The guard that makes the information boundary real rather than aspirational.
    assert_no_leak(obs.to_dict(), where="Observation")
    return obs
