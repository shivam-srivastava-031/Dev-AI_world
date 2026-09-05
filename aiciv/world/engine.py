"""The tick loop.

    FREEZE - OBSERVE - REMEMBER - PROPOSE - VALIDATE - EXECUTE
           - WORLD OUTCOME - RECORD EVIDENCE - UPDATE BELIEFS
           - VERIFY CLAIMS - TEACH/COMMUNICATE - UPKEEP - CHECKPOINT

Two properties the phase separation buys, both load-bearing:

* All agents decide against ONE frozen snapshot, so no agent's decision can
  depend on another's action within the same tick. Conflicts resolve
  deterministically by ascending agent_id.
* Because DECIDE is pure with respect to world state, LLM calls can later be
  parallelised across agents without changing a single result.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agents.observation import build_observation
from ..agents.policy import PolicyContext, build_policy
from ..config import REGISTER_COST, RunConfig
from ..ids import AgentId
from ..knowledge.claim import (
    TEACHABLE_STATES, ClaimState, ClaimType, KnowledgeClaim, TechniqueSpec,
)
from ..knowledge.causal import validate_adjustment_set
from ..knowledge.kb import KnowledgeBase
from ..knowledge.trials import Trial
from ..knowledge.verifier import Verifier
from ..rng import derive_rng
from .actions import ActionProposal, Rejection, RejectionCode, Verb
from .domains.base import Domain
from .executor import Event, execute
from .state import WorldState, initial_state
from .upkeep import upkeep


@dataclass
class TickRecord:
    tick: int
    state_hash: str
    proposals: list[tuple[int, ActionProposal]]
    rejections: list[tuple[int, Rejection]]
    events: list[Event]
    trials: list[Trial]


@dataclass
class Engine:
    config: RunConfig
    domain: Domain
    state: WorldState = field(init=False)
    run_secret: bytes = field(init=False)

    policies: dict[int, Any] = field(default_factory=dict)
    trials: list[Trial] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    history: list[TickRecord] = field(default_factory=list)

    # per-agent scratch that is NOT world state
    known_soil: dict[int, dict[int, int]] = field(default_factory=dict)
    last_result: dict[int, dict[str, Any]] = field(default_factory=dict)
    recent_harvests: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    inbox: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    beliefs: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)
    trials_by_id: dict[int, Trial] = field(default_factory=dict)
    _rows_cache: list[dict] = field(default_factory=list)

    language_gate: Callable[[str], Rejection | None] | None = None
    kb: KnowledgeBase = field(default_factory=KnowledgeBase)
    verifier: Verifier | None = None

    def __post_init__(self) -> None:
        self.state = initial_state(self.config)
        if self.verifier is None:
            self.verifier = Verifier(self.domain.verification, self.kb.log)
        # The secret never leaves this process except into the runs table. It is
        # what makes a trial row unforgeable even with write access to the DB.
        self.run_secret = secrets.token_bytes(32)
        for aid in self.state.agent_ids():
            self.policies[int(aid)] = build_policy(self.config.policy)
            self.known_soil[int(aid)] = {}
            self.last_result[int(aid)] = {}
            self.recent_harvests[int(aid)] = []
            self.inbox[int(aid)] = []
            self.beliefs[int(aid)] = {}

    # ---- the loop ---------------------------------------------------------

    def tick(self) -> TickRecord:
        from .validator import validate

        state = self.state
        t = state.tick

        # 1. FREEZE
        snapshot = state.snapshot()

        # 2-4. OBSERVE / REMEMBER / PROPOSE -- all against the same snapshot
        proposals: list[tuple[int, ActionProposal]] = []
        for aid in snapshot.agent_ids():
            a = int(aid)
            obs = build_observation(
                snapshot, aid,
                known_soil=self.known_soil[a],
                last_result=self.last_result[a] or None,
                recent_harvests=self.recent_harvests[a][-8:],
                messages=self.inbox[a][-5:],
                my_claims=[r.summary() for r in self.kb
                           if int(r.claim.author) == a],
                public_claims=[r.summary() for r in self.kb
                               if int(r.claim.author) != a
                               and r.state.value in ('supported',
                                                     'confirmed',
                                                     'generalized')],
            )
            ctx = PolicyContext(
                agent_id=a, tick=t,
                rng=derive_rng(self.config.seed, "policy", t, a),
                config=self.config,
                param_space={k: v.values for k, v in self.domain.param_space.items()},
            )
            proposals.append((a, self.policies[a].decide(obs, ctx)))

        # 5-7. VALIDATE / EXECUTE / WORLD OUTCOME
        rejections: list[tuple[int, Rejection]] = []
        tick_events: list[Event] = []
        tick_trials: list[Trial] = []
        claimed: set = set()

        for a, proposal in proposals:
            verdict = validate(proposal, AgentId(a), snapshot, self.domain,
                               language_gate=self.language_gate,
                               claimed_tiles=claimed)
            if isinstance(verdict, Rejection):
                rejections.append((a, verdict))
                # A rejection teaches: it lands in memory and is shown next tick.
                self.last_result[a] = {"ok": False, **verdict.as_feedback()}
                continue

            if proposal.verb is Verb.PLANT:
                x, y = proposal.params["tile"]
                claimed.add(y * state.grid.width + x)

            if proposal.verb in EPISTEMIC_VERBS:
                rej = self.handle_epistemic(a, proposal, t)
                if rej is not None:
                    rejections.append((a, rej))
                    self.last_result[a] = {'ok': False, **rej.as_feedback()}
                else:
                    self.last_result[a] = {'ok': True,
                                           'action': proposal.verb.value}
                    ag = state.agents[AgentId(a)]
                    ag.energy = max(0.0, ag.energy - 1.0)
                continue

            evs, trs = execute(proposal, AgentId(a), state, self.domain,
                               derive_rng(self.config.seed, "yield", t, a),
                               run_secret=self.run_secret)
            tick_events.extend(evs)
            tick_trials.extend(trs)
            self.last_result[a] = {"ok": True, "action": proposal.verb.value}

            for e in evs:
                if e.kind == "inspect":
                    self.known_soil[a][e.payload["tile_id"]] = e.payload["soil_band"]
                elif e.kind == "harvest":
                    self.recent_harvests[a].append(e.payload)
                elif e.kind in ("say", "ask") and e.payload.get("to") is not None:
                    target = int(e.payload["to"])
                    if target in self.inbox:
                        self.inbox[target].append(
                            {"from": a, "text": e.payload.get("message", "")})

        # 8-11. RECORD EVIDENCE / UPDATE BELIEFS / VERIFY / COMMUNICATE
        self.trials.extend(tick_trials)
        self.trials_by_id.update({int(x.trial_id): x for x in tick_trials})
        if self.kb.advanceable_records():
            self.verifier.run_round(
                self.kb, lambda rec: self.rows(), t,
                self.run_secret, self.trials_by_id)

        # 12. UPKEEP
        tick_events.extend(upkeep(state))

        # 13. CHECKPOINT
        state.tick += 1
        self.events.extend(tick_events)

        rec = TickRecord(t, state.state_hash(), proposals, rejections,
                         tick_events, tick_trials)
        self.history.append(rec)
        return rec

    def run(self, ticks: int | None = None) -> None:
        for _ in range(ticks if ticks is not None else self.config.ticks):
            self.tick()

    # ---- reporting --------------------------------------------------------

    def hash_sequence(self) -> list[str]:
        return [r.state_hash for r in self.history]

    def rejection_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.history:
            for _, rej in rec.rejections:
                counts[rej.code.value] = counts.get(rej.code.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

EPISTEMIC_VERBS = frozenset({
    Verb.PROPOSE_CLAIM, Verb.REGISTER_CLAIM, Verb.SUBMIT_TRIALS,
    Verb.CHALLENGE_CLAIM, Verb.WITHDRAW_CLAIM, Verb.TEACH,
})


def _rows(self) -> list[dict]:
    """Trial rows for the verifier. Rebuilt only when new evidence lands."""
    from dataclasses import asdict
    if len(self._rows_cache) != len(self.trials):
        self._rows_cache = [asdict(t) for t in self.trials]
    return self._rows_cache


def _handle_epistemic(self, agent, proposal, tick):
    """The OPTIONAL protocol verbs.

    Every branch can fail and none can force an outcome: the knowledge base
    accepts a *request*, and the Verifier alone decides what becomes true.
    """
    kb, verb, p = self.kb, proposal.verb, proposal.params
    ag = self.state.agents[AgentId(agent)]

    if verb is Verb.PROPOSE_CLAIM:
        adjust = tuple(p.get("adjustment_set", ()))
        try:
            ctype = ClaimType(p.get("claim_type", "comparison"))
        except ValueError:
            return Rejection(RejectionCode.E_SCHEMA_TYPE,
                             str(p.get("claim_type")))
        bad = validate_adjustment_set(ctype, adjust)
        if bad:
            return Rejection(RejectionCode.E_KB_POST_TREATMENT_COVARIATE, bad)
        try:
            claim = KnowledgeClaim(
                claim_id=kb.new_claim_id(), claim_version=1,
                author=AgentId(agent), claim_type=ctype,
                spec=TechniqueSpec(dict(p["spec"] or {})),
                baseline=TechniqueSpec(dict(p["baseline"] or {})),
                predicted_direction=str(p["direction"]),
                predicted_min_delta=float(p["min_delta"]),
                created_tick=tick, hypothesis=proposal.message,
                adjustment_set=adjust)
        except (ValueError, TypeError) as e:
            return Rejection(RejectionCode.E_BOUNDS_DELTA_NONPOSITIVE, str(e))
        kb.propose(claim)
        return None

    if verb is Verb.REGISTER_CLAIM:
        cid = p["claim_id"]
        code = kb.can_register(cid, AgentId(agent), ag.reputation)
        if code is not None:
            return Rejection(code, str(cid))
        ag.reputation = round(ag.reputation - REGISTER_COST, 3)
        kb.register(cid, tick, self.verifier.token)
        return None

    if verb is Verb.SUBMIT_TRIALS:
        code, detail = kb.can_submit(p["claim_id"], AgentId(agent),
                                     p["trial_ids"], self.trials_by_id,
                                     self.run_secret)
        if code is not None:
            return Rejection(code, detail)
        # Accepted, but note: the Verifier reads world-recorded evidence
        # directly. Submitting is a signal of intent, never a data transfer.
        return None

    if verb is Verb.CHALLENGE_CLAIM:
        if kb.get(p["claim_id"]) is None:
            return Rejection(RejectionCode.E_KB_CLAIM_NOT_FOUND,
                             str(p["claim_id"]))
        kb.challenge(p["claim_id"], AgentId(agent), tick, self.verifier.token)
        return None

    if verb is Verb.WITHDRAW_CLAIM:
        rec = kb.get(p["claim_id"])
        if rec is None:
            return Rejection(RejectionCode.E_KB_CLAIM_NOT_FOUND,
                             str(p["claim_id"]))
        if int(rec.claim.author) != agent:
            return Rejection(RejectionCode.E_KB_CLAIM_NOT_YOURS,
                             str(p["claim_id"]))
        kb.set_state(p["claim_id"], ClaimState.WITHDRAWN,
                     self.verifier.token, tick)
        return None

    if verb is Verb.TEACH:
        cid, target = p["claim_id"], int(p["to"])
        rec = kb.get(cid)
        if rec is None or rec.state not in TEACHABLE_STATES:
            return Rejection(RejectionCode.E_TEACH_UNKNOWN_CLAIM, str(cid))
        if str(cid) not in self.beliefs[agent] and int(rec.claim.author) != agent:
            return Rejection(RejectionCode.E_TEACH_UNKNOWN_CLAIM,
                             "you do not hold that claim")
        # Knowledge transfers; competence does not. No skill changes hands and
        # no world state moves -- the student receives a belief marked hearsay
        # and must run its own trials to upgrade it.
        self.beliefs[target][str(cid)] = {
            "claim_id": str(cid), "belief": 0.5, "source": agent,
            "status": "hearsay", "personal_trials": 0,
            "world_status": rec.state.value,
        }
        kb.record_teach(tick, AgentId(target), cid)
        return None

    return None


Engine.rows = _rows
Engine.handle_epistemic = _handle_epistemic
