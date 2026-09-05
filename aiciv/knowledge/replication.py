"""Replication kinds, resolved by communication-path analysis.

"Replicated by an agent other than the author" is NOT independence. Consider:

    A discovers X  ->  A tells B exactly X  ->  B runs X  ->  B gets X

That is social replication. It demonstrates that knowledge *travelled*, which is
a valuable result, but it is not independent corroboration and must never be
reported as such.

The question this module answers is narrow and checkable: for replicator R and
claim C whose first matching trial by R was at tick T, had any message carrying
C reached R before T?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ids import AgentId, ClaimId


class ReplicationKind(str, Enum):
    INDEPENDENT_REDISCOVERY = "independent_rediscovery"  # never heard of it
    BLIND_REPLICATION = "blind_replication"              # ran it before hearing
    INFORMED_REPLICATION = "informed_replication"        # heard publicly first
    TAUGHT_REPLICATION = "taught_replication"            # was taught it directly


#: Kinds that constitute genuine independent corroboration.
INDEPENDENT_KINDS = frozenset({
    ReplicationKind.INDEPENDENT_REDISCOVERY,
    ReplicationKind.BLIND_REPLICATION,
})

#: Channels, ordered by how directly they transmit a claim.
CHANNEL_TEACH = "teach"
CHANNEL_PUBLIC = "public"     # SAY within earshot, or browsing the public KB


@dataclass(frozen=True)
class Exposure:
    """One moment at which a claim became available to an agent."""

    tick: int
    agent: AgentId
    claim_id: ClaimId
    channel: str


class CommunicationLog:
    """Who learned what, when, and through which channel.

    Deliberately append-only and queried rather than mutated: a replication kind
    computed at tick 300 must give the same answer when recomputed from a replay.
    """

    def __init__(self) -> None:
        self._by_agent: dict[tuple[int, str], list[Exposure]] = {}

    def record(self, tick: int, agent: AgentId, claim_id: ClaimId,
               channel: str) -> None:
        key = (int(agent), str(claim_id))
        self._by_agent.setdefault(key, []).append(
            Exposure(tick, agent, claim_id, channel))

    def exposures(self, agent: AgentId, claim_id: ClaimId) -> list[Exposure]:
        return sorted(self._by_agent.get((int(agent), str(claim_id)), []),
                      key=lambda e: (e.tick, e.channel))

    def first_exposure(self, agent: AgentId, claim_id: ClaimId) -> Exposure | None:
        ex = self.exposures(agent, claim_id)
        return ex[0] if ex else None

    def knows(self, agent: AgentId, claim_id: ClaimId, by_tick: int) -> bool:
        first = self.first_exposure(agent, claim_id)
        return first is not None and first.tick <= by_tick


def classify_replication(
    *,
    author: AgentId,
    replicator: AgentId,
    claim_id: ClaimId,
    first_matching_trial_tick: int,
    log: CommunicationLog,
) -> ReplicationKind | None:
    """Classify how a replicator came to run the claim's condition.

    Returns None if the replicator IS the author -- an author cannot replicate
    their own claim, and the verifier rejects that separately with a specific
    code rather than silently ignoring it.
    """
    if int(author) == int(replicator):
        return None

    first = log.first_exposure(replicator, claim_id)

    if first is None:
        # Never heard of the claim, yet ran its condition anyway.
        return ReplicationKind.INDEPENDENT_REDISCOVERY

    if first_matching_trial_tick < first.tick:
        # Ran the condition before learning the claim existed. The evidence was
        # gathered without the hypothesis in mind, so it cannot be contaminated
        # by it.
        return ReplicationKind.BLIND_REPLICATION

    if first.channel == CHANNEL_TEACH:
        return ReplicationKind.TAUGHT_REPLICATION
    return ReplicationKind.INFORMED_REPLICATION


def independence_profile(kinds: list[ReplicationKind | None]) -> dict[str, int]:
    """Counts by kind, for the report's confirmation_independence_profile."""
    out = {k.value: 0 for k in ReplicationKind}
    for k in kinds:
        if k is not None:
            out[k.value] += 1
    return out


def is_independently_corroborated(kind: ReplicationKind | None) -> bool:
    return kind in INDEPENDENT_KINDS
