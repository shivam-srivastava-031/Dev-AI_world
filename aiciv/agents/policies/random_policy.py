"""RandomPolicy: the absolute floor.

Picks a legal-looking action uniformly. Exists so the engine can be exercised
for hundreds of ticks with zero LLM calls, and so every metric has a floor to
be compared against. If a sophisticated policy cannot beat this, it is not
doing anything.
"""

from __future__ import annotations

from ...world.actions import ActionProposal, Verb
from ..policy import PolicyContext, register

COMPANIONS = ("NONE", "CLOVER", "BEANS", "MARIGOLD", "THISTLE")


@register
class RandomPolicy:
    name = "random"

    def reset(self, ctx: PolicyContext) -> None:
        pass

    def observe_result(self, result: dict, ctx: PolicyContext) -> None:
        pass

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal:
        rng = ctx.rng

        # Harvest anything ready and in reach: otherwise the world fills with
        # mature crops and no evidence is ever recorded.
        ready_here = [t["tile"] for t in obs.nearby_tiles
                      if t["crop_ready"] and t["mine"]]
        if ready_here:
            return ActionProposal(Verb.HARVEST, {"tile": list(ready_here[0])})

        if obs.energy < 2.0:
            return ActionProposal(Verb.REST)

        # Only eat if there is something to eat. Without this guard a starving
        # agent burns every remaining tick on a rejected EAT -- a livelock that
        # produced 1680 rejections and 18 trials in the first 400-tick run.
        if obs.food >= 1.0 and obs.energy < 6.0:
            return ActionProposal(Verb.EAT)

        # Walk back toward a plot that is ready. A policy that wanders off and
        # never returns records nothing, which makes it useless as a floor.
        ready_plots = [p for p in obs.my_plots if p["ready"]]
        if ready_plots:
            step = _step_toward(obs, ready_plots[0]["tile"])
            if step is not None:
                return ActionProposal(Verb.MOVE, {"tile": step})

        free = [t["tile"] for t in obs.nearby_tiles
                if t["terrain"] == "arable" and not t["planted"]]
        can_plant = (free and obs.seeds > 0 and obs.water_stock >= 4
                     and len(obs.plots) < ctx.config.max_concurrent_plots)
        if can_plant and rng.random() < 0.7:
            tile = free[int(rng.integers(0, len(free)))]
            return ActionProposal(Verb.PLANT, {
                "tile": list(tile),
                "spacing": int(rng.integers(1, 6)),
                "water": int(rng.integers(0, 5)),
                "companion": COMPANIONS[int(rng.integers(0, len(COMPANIONS)))],
            })

        arable = [t["tile"] for t in obs.nearby_tiles
                  if t["terrain"] == "arable" and t["tile"] != [obs.x, obs.y]]
        if arable:
            tile = arable[int(rng.integers(0, len(arable)))]
            return ActionProposal(Verb.MOVE, {"tile": list(tile)})
        return ActionProposal(Verb.NOOP)


def _step_toward(obs, target) -> list[int] | None:
    """One Chebyshev step toward target, restricted to legal adjacent tiles."""
    tx, ty = target
    best, best_d = None, None
    for t in obs.nearby_tiles:
        x, y = t["tile"]
        if t["terrain"] != "arable" or [x, y] == [obs.x, obs.y]:
            continue
        d = max(abs(tx - x), abs(ty - y))
        if best_d is None or d < best_d:
            best, best_d = [x, y], d
    if best is None:
        return None
    if best_d >= max(abs(tx - obs.x), abs(ty - obs.y)):
        return None          # no progress available; do something else
    return best
