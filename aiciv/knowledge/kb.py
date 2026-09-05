"""The knowledge base: the public record, and the only writer of claim state.

``set_state`` demands a VerifierToken, which only ``Verifier.__init__`` can
construct. Everything an agent can do -- propose, register, submit, challenge,
withdraw -- goes through the ordinary validator and can only ever *request* a
transition. This is the structural reason an agent cannot declare itself right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..config import MAX_OPEN_CLAIMS, REGISTER_COST
from ..ids import AgentId, ClaimId, claim_id as make_claim_id
from ..world.actions import RejectionCode
from .claim import (
    ADVANCEABLE_STATES, LEGAL_TRANSITIONS, OPEN_STATES, TEACHABLE_STATES,
    ClaimRecord, ClaimState, KnowledgeClaim,
)
from .replication import CHANNEL_PUBLIC, CHANNEL_TEACH, CommunicationLog


@dataclass
class KnowledgeBase:
    records: dict[ClaimId, ClaimRecord] = field(default_factory=dict)
    log: CommunicationLog = field(default_factory=CommunicationLog)
    _next: int = 1
    transitions: list[tuple[int, str, str, str]] = field(default_factory=list)

    # -- reads --------------------------------------------------------------

    def get(self, cid: ClaimId) -> ClaimRecord | None:
        return self.records.get(cid)

    def advanceable_records(self) -> list[ClaimRecord]:
        """Claims the Verifier should re-examine. Sorted so a round is
        order-independent."""
        return [self.records[c] for c in sorted(self.records, key=str)
                if self.records[c].state in ADVANCEABLE_STATES]

    # Back-compat alias; prefer advanceable_records.
    def open_records(self) -> list[ClaimRecord]:
        return self.advanceable_records()

    def by_state(self, state: ClaimState) -> list[ClaimRecord]:
        return [r for r in self.records.values() if r.state is state]

    def open_count(self, agent: AgentId) -> int:
        return sum(1 for r in self.records.values()
                   if int(r.claim.author) == int(agent) and r.is_open)

    def find_duplicate(self, claim: KnowledgeClaim) -> ClaimRecord | None:
        """Same technique, different author: the signature of independent
        rediscovery rather than transfer."""
        for r in self.records.values():
            if r.claim.spec_hash == claim.spec_hash and \
                    int(r.claim.author) != int(claim.author):
                return r
        return None

    def teachable(self, cid: ClaimId) -> bool:
        r = self.get(cid)
        return r is not None and r.state in TEACHABLE_STATES

    # -- agent-facing requests (all can fail; none can force a state) --------

    def new_claim_id(self) -> ClaimId:
        cid = make_claim_id(self._next)
        self._next += 1
        return cid

    def propose(self, claim: KnowledgeClaim) -> ClaimRecord:
        record = ClaimRecord(claim=claim, state=ClaimState.PROPOSED)
        dup = self.find_duplicate(claim)
        if dup is not None:
            record.duplicate_of = dup.claim.claim_id
        self.records[claim.claim_id] = record
        return record

    def can_register(self, cid: ClaimId, agent: AgentId,
                     reputation: float) -> RejectionCode | None:
        r = self.get(cid)
        if r is None:
            return RejectionCode.E_KB_CLAIM_NOT_FOUND
        if int(r.claim.author) != int(agent):
            return RejectionCode.E_KB_CLAIM_NOT_YOURS
        if r.state is not ClaimState.PROPOSED:
            return RejectionCode.E_KB_CLAIM_WRONG_STATE
        if self.open_count(agent) > MAX_OPEN_CLAIMS:
            return RejectionCode.E_KB_TOO_MANY_OPEN_CLAIMS
        if reputation < REGISTER_COST:
            return RejectionCode.E_KB_INSUFFICIENT_REPUTATION
        return None

    def register(self, cid: ClaimId, tick: int, token) -> None:
        """Freeze the claim and start the pre-registration clock.

        After this, only trials harvested LATER than ``tick`` can support it.
        """
        r = self.records[cid]
        r.registered_tick = tick
        self.set_state(cid, ClaimState.REGISTERED, token, tick)

    def can_submit(self, cid: ClaimId, agent: AgentId, trial_ids, trials_by_id,
                   run_secret: bytes) -> tuple[RejectionCode | None, str]:
        """Check a SUBMIT_TRIALS request. Every branch here is an attack that
        the adversarial suite fires at us."""
        r = self.get(cid)
        if r is None:
            return RejectionCode.E_KB_CLAIM_NOT_FOUND, str(cid)
        if r.registered_tick is None:
            return RejectionCode.E_KB_CLAIM_WRONG_STATE, "not registered yet"
        for tid in trial_ids:
            t = trials_by_id.get(int(tid))
            if t is None:
                return RejectionCode.E_KB_TRIAL_NOT_FOUND, str(tid)
            if not t.verify(run_secret):
                return RejectionCode.E_TRIAL_SIGNATURE_INVALID, str(tid)
            if int(t.agent_id) != int(agent):
                return RejectionCode.E_KB_TRIAL_NOT_YOURS, str(tid)
            if t.harvested_tick <= r.registered_tick:
                return (RejectionCode.E_KB_TRIAL_PRE_REGISTRATION,
                        f"trial {tid} harvested at tick {t.harvested_tick}, "
                        f"claim registered at {r.registered_tick}")
            if any(int(tid) in s for s in r.consumed.values()):
                return RejectionCode.E_KB_TRIAL_ALREADY_USED, str(tid)
        return None, ""

    def challenge(self, cid: ClaimId, agent: AgentId, tick: int, token) -> None:
        r = self.records[cid]
        r.challenged_by.append(int(agent))
        if r.state in (ClaimState.SUPPORTED, ClaimState.CONFIRMED,
                       ClaimState.GENERALIZED):
            self.set_state(cid, ClaimState.CONTESTED, token, tick)

    # -- exposure tracking ---------------------------------------------------

    def record_teach(self, tick: int, learner: AgentId, cid: ClaimId) -> None:
        self.log.record(tick, learner, cid, CHANNEL_TEACH)

    def record_public_mention(self, tick: int, hearer: AgentId, cid: ClaimId) -> None:
        self.log.record(tick, hearer, cid, CHANNEL_PUBLIC)

    # -- the guarded write ---------------------------------------------------

    def set_state(self, cid: ClaimId, new: ClaimState, token, tick: int) -> None:
        """The ONLY mutation of claim state. Requires a VerifierToken."""
        from .verifier import VerifierToken

        if not isinstance(token, VerifierToken):
            raise PermissionError(
                "claim state is writable only by the Verifier; "
                f"got {type(token).__name__}"
            )
        r = self.records[cid]
        if new is r.state:
            return
        if new not in LEGAL_TRANSITIONS[r.state]:
            raise ValueError(f"illegal transition {r.state.value} -> {new.value}")

        self.transitions.append((tick, str(cid), r.state.value, new.value))
        r.state = new
        if new not in OPEN_STATES:
            r.decided_tick = tick

    # -- reporting -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in ClaimState}
        for r in self.records.values():
            out[r.state.value] += 1
        return out

    def __iter__(self) -> Iterator[ClaimRecord]:
        return iter(self.records.values())

    def __len__(self) -> int:
        return len(self.records)
