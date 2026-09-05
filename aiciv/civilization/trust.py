"""Belief, trust, and the epistemic consequences of being taught.

Two things this module keeps strictly apart, because conflating them is the
easiest way to fake a result:

  BELIEF   what an agent holds to be true, and on whose word
  KNOWLEDGE what the world has verified

Teaching moves a claim into a student's head as ``hearsay`` and nothing more.
No skill changes hands, no world state moves. The student upgrades its own
belief only by running its own trials -- so "knowledge transferred" and
"competence transferred" stay separately measurable.

Trust is endogenous: it moves when a taught claim turns out to hold up or not,
never by assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

TRUST_START = 0.5
TRUST_CONFIRMED = 0.10       # a teacher whose claim held up
TRUST_REFUTED = -0.15        # asymmetric: being misled costs more than being helped
PERSONAL_TRIALS_TO_JUDGE = 4


class BeliefStatus(str, Enum):
    NONE = "none"
    HEARSAY = "hearsay"                        # told, not tested
    PERSONALLY_CONFIRMED = "personally_confirmed"
    PERSONALLY_REFUTED = "personally_refuted"


@dataclass
class Belief:
    claim_id: str
    status: BeliefStatus = BeliefStatus.HEARSAY
    strength: float = 0.5
    source: int | None = None                  # who told them, if anyone
    learned_tick: int | None = None
    personal_trials: int = 0
    personal_treat: list[float] = field(default_factory=list)
    personal_ctrl: list[float] = field(default_factory=list)
    world_status: str = "unknown"
    adopted_tick: int | None = None            # first action matching the claim

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "status": self.status.value,
            "strength": round(self.strength, 3), "source": self.source,
            "learned_tick": self.learned_tick,
            "personal_trials": self.personal_trials,
            "world_status": self.world_status,
            "adopted_tick": self.adopted_tick,
        }


class Beliefs:
    """One agent's private epistemic state. Never world state."""

    def __init__(self) -> None:
        self._by_claim: dict[str, Belief] = {}

    def __contains__(self, cid: str) -> bool:
        return cid in self._by_claim

    def get(self, cid: str) -> Belief | None:
        return self._by_claim.get(cid)

    def all(self) -> list[Belief]:
        return [self._by_claim[c] for c in sorted(self._by_claim)]

    def learn(self, cid: str, source: int, tick: int, world_status: str) -> Belief:
        """Receive a claim from another agent. Hearsay, and nothing more."""
        b = self._by_claim.get(cid)
        if b is None:
            b = Belief(claim_id=cid, source=source, learned_tick=tick,
                       world_status=world_status)
            self._by_claim[cid] = b
        return b

    def observe(self, cid: str, *, in_treat: bool, yield_kg: float,
                tick: int) -> Belief:
        """Fold one of the agent's OWN trials into its belief.

        This is the only route from hearsay to conviction. An agent that is told
        something and never tests it stays at hearsay forever, which is what
        makes blind adoption measurable.
        """
        b = self._by_claim.setdefault(cid, Belief(claim_id=cid,
                                                  status=BeliefStatus.NONE))
        b.personal_trials += 1
        (b.personal_treat if in_treat else b.personal_ctrl).append(float(yield_kg))
        if b.adopted_tick is None and in_treat:
            b.adopted_tick = tick

        nt, nc = len(b.personal_treat), len(b.personal_ctrl)
        if nt >= PERSONAL_TRIALS_TO_JUDGE and nc >= PERSONAL_TRIALS_TO_JUDGE:
            mt = sum(b.personal_treat) / nt
            mc = sum(b.personal_ctrl) / nc
            # A deliberately weaker bar than world verification. The asymmetry
            # between what an agent will believe and what the world will certify
            # is itself something worth measuring.
            if mt > mc:
                b.status = BeliefStatus.PERSONALLY_CONFIRMED
                b.strength = min(1.0, b.strength + 0.2)
            else:
                b.status = BeliefStatus.PERSONALLY_REFUTED
                b.strength = max(0.0, b.strength - 0.3)
        return b


@dataclass
class Relationships:
    """Directed trust, updated only by outcomes."""

    trust: dict[tuple[int, int], float] = field(default_factory=dict)
    teach_events: dict[tuple[int, int], int] = field(default_factory=dict)
    useful_teach: dict[tuple[int, int], int] = field(default_factory=dict)
    misleading_teach: dict[tuple[int, int], int] = field(default_factory=dict)

    def get(self, a: int, b: int) -> float:
        return self.trust.get((a, b), TRUST_START)

    def record_teach(self, teacher: int, student: int) -> None:
        k = (student, teacher)
        self.teach_events[k] = self.teach_events.get(k, 0) + 1
        self.trust.setdefault(k, TRUST_START)

    def settle(self, student: int, teacher: int, confirmed: bool) -> None:
        """The student found out. Trust moves on evidence, not on assertion."""
        k = (student, teacher)
        delta = TRUST_CONFIRMED if confirmed else TRUST_REFUTED
        self.trust[k] = max(0.0, min(1.0, self.get(*k) + delta))
        target = self.useful_teach if confirmed else self.misleading_teach
        target[k] = target.get(k, 0) + 1

    def report(self) -> dict:
        taught = sum(self.teach_events.values())
        useful = sum(self.useful_teach.values())
        misled = sum(self.misleading_teach.values())
        return {
            "teach_events": taught,
            "settled": useful + misled,
            "useful_teach_events": useful,
            "misleading_teach_events": misled,
            "trust_pairs": {f"{s}->{t}": round(v, 3)
                            for (s, t), v in sorted(self.trust.items())},
            "mean_trust": (round(sum(self.trust.values()) / len(self.trust), 3)
                           if self.trust else None),
        }
