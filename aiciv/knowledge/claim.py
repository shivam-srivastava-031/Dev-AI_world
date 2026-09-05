"""Claims: falsifiable, versioned, and frozen once registered.

A claim is the unit of public knowledge. Three properties make it more than a
sentence an agent asserted:

* It carries a MANDATORY baseline. "Clover is good" is not verifiable;
  "clover at spacing 4-5 beats no companion at spacing 4-5 by at least 0.5" is.
* Its type binds a specific analysis. A CAUSAL_EFFECT cannot be settled with
  associational evidence just because the numbers came out nicely.
* Registration freezes it. Revision forks a new version with lineage intact, so
  knowledge can be refined without history being rewritten.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

from ..hashing import blake2b_hex
from ..ids import AgentId, ClaimId


class ClaimType(str, Enum):
    OBSERVATION = "observation"        # "yield varies with companion"
    COMPARISON = "comparison"          # "A yields more than B"      (associational)
    CAUSAL_EFFECT = "causal_effect"    # "changing A raises yield"   (causal)
    OPTIMAL_RANGE = "optimal_range"    # "this region is best"
    INTERACTION = "interaction"        # "the effect of A depends on B"
    GENERALIZATION = "generalization"  # "it holds where we did not look"


class ClaimState(str, Enum):
    PROPOSED = "proposed"
    REGISTERED = "registered"
    TESTING = "testing"
    SUPPORTED = "supported"        # passed on discovery-stratum evidence
    CONFIRMED = "confirmed"        # independently replicated on confirmation
    GENERALIZED = "generalized"    # also held on untouched holdout tiles
    CONTESTED = "contested"
    REFUTED = "refuted"
    WITHDRAWN = "withdrawn"


#: States that count against an agent's open-claim budget. A CONFIRMED claim
#: is finished work and must NOT occupy a registration slot.
OPEN_STATES = frozenset({ClaimState.PROPOSED, ClaimState.REGISTERED,
                         ClaimState.TESTING, ClaimState.SUPPORTED,
                         ClaimState.CONTESTED})

#: States the Verifier must keep re-examining each round.
#: CONFIRMED belongs here and not in OPEN_STATES: a confirmed claim is still
#: eligible to advance to GENERALIZED on holdout evidence, but it no longer
#: costs its author a registration slot. Conflating the two made GENERALIZED
#: unreachable in practice -- the stage was never evaluated even when ample
#: holdout evidence existed.
ADVANCEABLE_STATES = OPEN_STATES | {ClaimState.CONFIRMED}

#: States that count as public knowledge an agent may teach.
TEACHABLE_STATES = frozenset({ClaimState.TESTING, ClaimState.SUPPORTED,
                              ClaimState.CONFIRMED, ClaimState.GENERALIZED,
                              ClaimState.CONTESTED})

TERMINAL_STATES = frozenset({ClaimState.REFUTED, ClaimState.WITHDRAWN})

#: Legal transitions. The Verifier is the only thing allowed to walk these,
#: and only via a VerifierToken it alone can construct.
LEGAL_TRANSITIONS: dict[ClaimState, frozenset[ClaimState]] = {
    ClaimState.PROPOSED:    frozenset({ClaimState.REGISTERED, ClaimState.WITHDRAWN}),
    ClaimState.REGISTERED:  frozenset({ClaimState.TESTING, ClaimState.WITHDRAWN}),
    ClaimState.TESTING:     frozenset({ClaimState.SUPPORTED, ClaimState.REFUTED,
                                       ClaimState.WITHDRAWN}),
    ClaimState.SUPPORTED:   frozenset({ClaimState.CONFIRMED, ClaimState.REFUTED,
                                       ClaimState.CONTESTED}),
    ClaimState.CONFIRMED:   frozenset({ClaimState.GENERALIZED, ClaimState.CONTESTED,
                                       ClaimState.REFUTED}),
    ClaimState.GENERALIZED: frozenset({ClaimState.CONTESTED, ClaimState.REFUTED}),
    ClaimState.CONTESTED:   frozenset({ClaimState.SUPPORTED, ClaimState.CONFIRMED,
                                       ClaimState.GENERALIZED, ClaimState.REFUTED}),
    ClaimState.REFUTED:     frozenset(),
    ClaimState.WITHDRAWN:   frozenset(),
}


@dataclass(frozen=True)
class TechniqueSpec:
    """A region of parameter space, as constraints on trial fields.

    An empty condition set matches everything, which is legal but useless: a
    claim whose spec and baseline both match everything has zero effect by
    construction and will be refuted.
    """

    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def matches(self, row: dict[str, Any]) -> bool:
        for name, cond in self.conditions.items():
            if name not in row:
                return False
            v, op = row[name], cond.get("op")
            if op == "eq":
                if v != cond["value"]:
                    return False
            elif op == "in":
                if v not in cond["values"]:
                    return False
            elif op == "range":
                if not (cond["lo"] <= v <= cond["hi"]):
                    return False
            else:
                return False
        return True

    @property
    def hash(self) -> str:
        return blake2b_hex(self.conditions)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted(self.conditions))

    def describe(self) -> str:
        if not self.conditions:
            return "anything"
        parts = []
        for k in sorted(self.conditions):
            c = self.conditions[k]
            if c["op"] == "eq":
                parts.append(f"{k}={c['value']}")
            elif c["op"] == "in":
                parts.append(f"{k} in {list(c['values'])}")
            else:
                parts.append(f"{k} in [{c['lo']},{c['hi']}]")
        return " and ".join(parts)

    def overlaps(self, other: "TechniqueSpec") -> bool:
        """Do spec and baseline describe overlapping regions?

        An overlapping baseline makes a claim partly a comparison of a set with
        itself, which biases the estimate toward zero and wastes trials.
        """
        for k, a in self.conditions.items():
            b = other.conditions.get(k)
            if b is None:
                continue
            if not _cond_overlap(a, b):
                return False
        return True


def _values(cond: dict[str, Any]) -> set | None:
    if cond["op"] == "eq":
        return {cond["value"]}
    if cond["op"] == "in":
        return set(cond["values"])
    return None      # range: handled numerically


def _cond_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    va, vb = _values(a), _values(b)
    if va is not None and vb is not None:
        return bool(va & vb)
    if a["op"] == "range" and b["op"] == "range":
        return a["lo"] <= b["hi"] and b["lo"] <= a["hi"]
    rng, val = (a, vb) if a["op"] == "range" else (b, va)
    if val is None:
        return True
    return any(rng["lo"] <= v <= rng["hi"] for v in val
               if isinstance(v, (int, float)))


@dataclass(frozen=True)
class KnowledgeClaim:
    """Immutable once registered. Revision forks a new version."""

    claim_id: ClaimId
    claim_version: int
    author: AgentId
    claim_type: ClaimType
    spec: TechniqueSpec
    baseline: TechniqueSpec
    predicted_direction: str          # "increase" | "decrease"
    predicted_min_delta: float
    created_tick: int
    hypothesis: str = ""              # free English, for humans; never analysed
    adjustment_set: tuple[str, ...] = ()
    parent_claim_id: ClaimId | None = None

    def __post_init__(self) -> None:
        if self.predicted_min_delta <= 0:
            raise ValueError("predicted_min_delta must be > 0")
        if self.predicted_direction not in ("increase", "decrease"):
            raise ValueError(f"bad direction: {self.predicted_direction}")

    @property
    def spec_hash(self) -> str:
        """Identical techniques from different agents collide here.

        That collision is exactly how metrics separate INDEPENDENT_REDISCOVERY
        from transfer: same hash, no communication path, two authors.
        """
        return blake2b_hex(self.claim_type.value, self.spec.conditions,
                           self.baseline.conditions, self.predicted_direction)

    @property
    def analysis_plan(self) -> str:
        return ANALYSIS_PLAN[self.claim_type]

    def fork(self, new_id: ClaimId, tick: int, **changes: Any) -> "KnowledgeClaim":
        """Revise without rewriting history: a new version, lineage intact."""
        return replace(self, claim_id=new_id, claim_version=self.claim_version + 1,
                       parent_claim_id=self.claim_id, created_tick=tick, **changes)

    def describe(self) -> str:
        verb = "more than" if self.predicted_direction == "increase" else "less than"
        return (f"[{self.claim_type.value}] {self.spec.describe()} yields {verb} "
                f"{self.baseline.describe()} by at least "
                f"{self.predicted_min_delta:.2f}")


ANALYSIS_PLAN: dict[ClaimType, str] = {
    ClaimType.OBSERVATION: "omnibus F-test across the levels of the named variable",
    ClaimType.COMPARISON: "one-sided Welch or ANCOVA on treat vs baseline",
    ClaimType.CAUSAL_EFFECT: "ANCOVA on the treat coefficient, PRE_TREATMENT covariates only",
    ClaimType.OPTIMAL_RANGE: "region must beat every adjacent region",
    ClaimType.INTERACTION: "factorial design, tests the interaction term",
    ClaimType.GENERALIZATION: "re-test on the holdout stratum",
}


@dataclass
class ClaimRecord:
    """Mutable status. Only the Verifier may change ``state``."""

    claim: KnowledgeClaim
    state: ClaimState = ClaimState.PROPOSED
    registered_tick: int | None = None
    decided_tick: int | None = None
    verdicts: dict[str, Any] = field(default_factory=dict)
    replication_kind: str | None = None
    replicator: int | None = None
    consumed: dict[str, set[int]] = field(default_factory=dict)
    duplicate_of: ClaimId | None = None
    challenged_by: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    def summary(self) -> dict[str, Any]:
        d = {
            "claim_id": str(self.claim.claim_id),
            "version": self.claim.claim_version,
            "author": int(self.claim.author),
            "type": self.claim.claim_type.value,
            "state": self.state.value,
            "spec_hash": self.claim.spec_hash,
            "describes": self.claim.describe(),
            "registered_tick": self.registered_tick,
            "decided_tick": self.decided_tick,
            "replication_kind": self.replication_kind,
            "replicator": self.replicator,
        }
        for stage, v in self.verdicts.items():
            d[f"verdict_{stage}"] = v
        return d
