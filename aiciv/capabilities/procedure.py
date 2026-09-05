"""Knowledge -> Capability -> Artifact -> Technology.

This is the loop that separates a civilization from a farming-statistics
simulator. Without it, agents can get better and better at measuring the same
fixed world forever and nothing they learn ever changes what is possible.

    CONFIRMED claim -> Procedure -> (skill + inputs) -> Capability
        -> BUILD -> Artifact (persists in the world)
        -> the action space changes -> a previously unreachable region of the
           parameter space becomes searchable -> Technology (transmissible)

The vertical slice implemented here is deliberately small and complete rather
than broad and notional: irrigation. An agent with a confirmed claim about
water compiles it into a Procedure, which unlocks the CHANNEL capability, which
lets it BUILD a channel on a tile. The channel is a real object in the world:
it persists, anyone can use it, and it raises the water available on nearby
tiles -- which makes recipes that were previously impossible to try, testable.

That last clause is the whole point. A "technology" that does not change what
can be attempted is a badge, not a technology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..ids import AgentId, ClaimId, TileId


class CapabilityId(str, Enum):
    IRRIGATION = "irrigation"


@dataclass(frozen=True)
class Procedure:
    """A confirmed claim compiled into something repeatable.

    Compiled from public knowledge, not asserted: a procedure can only exist
    where a claim already survived verification.
    """

    procedure_id: str
    from_claim: ClaimId
    author: AgentId
    created_tick: int
    capability: CapabilityId
    conditions: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.capability.value} procedure from {self.from_claim}"


@dataclass(frozen=True)
class CapabilitySpec:
    """What it takes to hold a capability, and what holding it permits."""

    capability: CapabilityId
    min_skill: float
    inputs: dict[str, int]
    unlocks_action: str
    description: str


CAPABILITIES: dict[CapabilityId, CapabilitySpec] = {
    CapabilityId.IRRIGATION: CapabilitySpec(
        capability=CapabilityId.IRRIGATION,
        min_skill=0.25,
        inputs={"seeds": 2},        # stands in for materials in v0.1
        unlocks_action="BUILD_CHANNEL",
        description="dig a channel that waters the tiles around it",
    ),
}

#: A claim qualifies as the seed of an irrigation procedure when it is about
#: the water axis. Deliberately narrow: procedures are compiled from specific
#: verified knowledge, not granted for any confirmed claim whatsoever.
WATER_AXES = frozenset({"water", "irrigation"})


def claim_supports_capability(claim) -> CapabilityId | None:
    """Which capability, if any, this claim could be compiled into."""
    touched = set(claim.spec.conditions) | set(claim.baseline.conditions)
    if touched & WATER_AXES:
        return CapabilityId.IRRIGATION
    return None


@dataclass
class Artifact:
    """A thing built in the world. Persists, and is usable by anyone.

    "Usable by anyone" is what makes it technology rather than a private skill:
    an agent who never learned anything about water still benefits from
    standing next to someone else's channel.
    """

    artifact_id: int
    kind: str
    tile_id: TileId
    built_by: AgentId
    built_tick: int
    radius: int = 2
    water_bonus: int = 3

    def covers(self, x: int, y: int, ax: int, ay: int) -> bool:
        return max(abs(x - ax), abs(y - ay)) <= self.radius


@dataclass
class CapabilityRegistry:
    """Who holds what, and what has been built.

    Capabilities are WORLD-RECOGNISED: the validator consults this, so holding
    one genuinely changes what an agent may do rather than merely being noted
    in its memory.
    """

    procedures: dict[str, Procedure] = field(default_factory=dict)
    held: dict[int, set[CapabilityId]] = field(default_factory=dict)
    artifacts: dict[int, Artifact] = field(default_factory=dict)
    by_tile: dict[int, int] = field(default_factory=dict)
    _next_artifact: int = 1
    events: list[dict[str, Any]] = field(default_factory=list)

    # -- knowledge -> procedure -------------------------------------------

    def compile_procedure(self, claim, tick: int) -> Procedure | None:
        """Turn a confirmed claim into a repeatable procedure.

        Returns None when the claim is not about anything a capability is built
        on. Compiling is automatic on confirmation, but it is NOT a reward: it
        is the mechanical consequence of the world now knowing something.
        """
        cap = claim_supports_capability(claim)
        if cap is None:
            return None
        pid = f"proc_{claim.claim_id}"
        if pid in self.procedures:
            return self.procedures[pid]
        proc = Procedure(
            procedure_id=pid, from_claim=claim.claim_id, author=claim.author,
            created_tick=tick, capability=cap,
            conditions=dict(claim.spec.conditions),
        )
        self.procedures[pid] = proc
        self.events.append({"tick": tick, "kind": "procedure_compiled",
                            "procedure": pid, "capability": cap.value,
                            "agent_id": int(claim.author)})
        return proc

    # -- procedure + skill + inputs -> capability --------------------------

    def grant(self, agent: int, cap: CapabilityId, tick: int) -> None:
        self.held.setdefault(agent, set())
        if cap not in self.held[agent]:
            self.held[agent].add(cap)
            self.events.append({"tick": tick, "kind": "capability_unlocked",
                                "agent_id": agent, "capability": cap.value})

    def holds(self, agent: int, cap: CapabilityId) -> bool:
        return cap in self.held.get(agent, set())

    def can_build(self, agent: int, cap: CapabilityId, skill: float,
                  seeds: int) -> str | None:
        """Rejection detail, or None if the agent may build.

        Holding the knowledge is necessary and not sufficient: capability needs
        the procedure AND the competence AND the materials.
        """
        if not self.holds(agent, cap):
            return "you do not know how to do that"
        spec = CAPABILITIES[cap]
        if skill < spec.min_skill:
            return (f"you are not practised enough "
                    f"({skill:.2f} < {spec.min_skill})")
        if seeds < spec.inputs.get("seeds", 0):
            return f"you need {spec.inputs['seeds']} to spare"
        return None

    # -- capability -> artifact -------------------------------------------

    def build(self, *, agent: int, tile_id: int, tick: int,
              kind: str = "channel") -> Artifact:
        art = Artifact(artifact_id=self._next_artifact, kind=kind,
                       tile_id=TileId(tile_id), built_by=AgentId(agent),
                       built_tick=tick)
        self.artifacts[art.artifact_id] = art
        self.by_tile[int(tile_id)] = art.artifact_id
        self._next_artifact += 1
        self.events.append({"tick": tick, "kind": "artifact_built",
                            "agent_id": agent, "tile_id": int(tile_id),
                            "artifact": art.artifact_id})
        return art

    # -- artifact -> changed action space ---------------------------------

    def water_bonus_at(self, grid, x: int, y: int) -> int:
        """How much extra water the built environment supplies here.

        This is the payoff of the whole chain: it enters the validator's water
        check, so a channel makes recipes affordable that an agent could not
        previously attempt at all.
        """
        bonus = 0
        for art in self.artifacts.values():
            tile = grid.by_id(art.tile_id)
            if art.covers(tile.x, tile.y, x, y):
                bonus = max(bonus, art.water_bonus)
        return bonus

    def report(self) -> dict[str, Any]:
        return {
            "procedures": len(self.procedures),
            "agents_with_capability": {
                str(a): sorted(c.value for c in caps)
                for a, caps in sorted(self.held.items()) if caps
            },
            "artifacts": len(self.artifacts),
            "artifact_tiles": sorted(self.by_tile),
            "events": self.events,
            # Technology proper: agents benefiting from an artifact they did
            # not build and could not have built themselves.
            "builders": sorted({int(a.built_by) for a in self.artifacts.values()}),
        }
