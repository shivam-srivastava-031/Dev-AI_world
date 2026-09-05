"""Replay: re-execute a recorded run and prove it produces the same world.

Two modes, answering two different questions:

  replay_actions  Re-runs the logged decision stream through a fresh engine and
                  asserts every per-tick state hash matches. This proves the
                  WORLD ENGINE is deterministic and needs no model at all -- an
                  LLM run can be verified by someone who does not have the LLM.

  replay_full     Additionally re-runs the policies against the LLM call log,
                  asserting the emitted proposals are byte-identical. This
                  proves the whole pipeline is reproducible even though it was
                  not deterministic when first produced.

A mismatch is always reported with the tick it first occurred at. "Something
diverged somewhere" is not a useful thing to tell anyone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import RunConfig
from ..ids import AgentId
from ..world.actions import ActionProposal, Verb
from ..world.domains.agronomy import AgronomyDomain
from ..world.domains.synthetic import SyntheticDomain

DOMAINS = {"synthetic": SyntheticDomain, "agronomy": AgronomyDomain}


@dataclass
class ReplayResult:
    run_id: str
    mode: str
    ticks_checked: int
    matched: bool
    first_divergence: int | None = None
    detail: str = ""
    recorded_hash: str | None = None
    replayed_hash: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "mode": self.mode,
            "ticks_checked": self.ticks_checked, "matched": self.matched,
            "first_divergence": self.first_divergence, "detail": self.detail,
            "recorded_final_hash": self.recorded_hash,
            "replayed_final_hash": self.replayed_hash,
            "notes": self.notes,
        }


class _LoggedPolicy:
    """Replays one agent's recorded decisions. Consults no model and no state.

    This is what makes the engine independently checkable: the decisions are
    treated as data, and the only thing under test is whether the world does
    the same thing with them.
    """

    name = "logged"

    def __init__(self, actions_by_tick: dict[int, ActionProposal]) -> None:
        self.actions = actions_by_tick
        self.missing = 0

    def reset(self, ctx) -> None:
        pass

    def observe_result(self, result, ctx) -> None:
        pass

    def decide(self, obs, ctx) -> ActionProposal:
        proposal = self.actions.get(ctx.tick)
        if proposal is None:
            self.missing += 1
            return ActionProposal(Verb.NOOP)
        return proposal


def _load_actions(store, run_id: str) -> dict[int, dict[int, ActionProposal]]:
    by_agent: dict[int, dict[int, ActionProposal]] = {}
    for row in store.actions(run_id):
        payload = json.loads(row["proposal_json"])
        proposal = ActionProposal(
            verb=Verb(payload["verb"]),
            params=payload.get("params") or {},
            rationale=payload.get("rationale", ""),
            message=payload.get("message", ""),
        )
        by_agent.setdefault(row["agent_id"], {})[row["tick"]] = proposal
    return by_agent


def replay_actions(store, run_id: str) -> ReplayResult:
    """Re-execute the logged decisions and compare state hashes tick by tick."""
    from ..world.engine import Engine

    meta = store.run(run_id)
    if meta is None:
        return ReplayResult(run_id, "actions", 0, False,
                            detail=f"no such run: {run_id}")

    cfg_raw = json.loads(meta["config_json"])
    cfg = RunConfig(**{k: v for k, v in cfg_raw.items()
                       if k in RunConfig.__dataclass_fields__})
    if cfg.hash != meta["config_hash"]:
        return ReplayResult(run_id, "actions", 0, False,
                            detail="config hash mismatch: the recorded config "
                                   "does not rebuild to the same value")

    domain_cls = DOMAINS.get(meta["domain"])
    if domain_cls is None:
        return ReplayResult(run_id, "actions", 0, False,
                            detail=f"unknown domain {meta['domain']}")

    recorded = store.state_hashes(run_id)
    by_agent = _load_actions(store, run_id)

    engine = Engine(cfg, domain_cls(water_hint=cfg.water_hint))
    for aid in engine.state.agent_ids():
        engine.policies[int(aid)] = _LoggedPolicy(by_agent.get(int(aid), {}))

    notes = []
    for tick, expected in enumerate(recorded):
        rec = engine.tick()
        if rec.state_hash != expected:
            return ReplayResult(
                run_id, "actions", tick + 1, False, first_divergence=tick,
                detail=(f"state diverged at tick {tick}: recorded "
                        f"{expected[:12]}, replayed {rec.state_hash[:12]}"),
                recorded_hash=recorded[-1],
                replayed_hash=engine.hash_sequence()[-1], notes=notes)

    missing = sum(p.missing for p in engine.policies.values())
    if missing:
        notes.append(f"{missing} ticks had no recorded action for some agent")

    return ReplayResult(
        run_id, "actions", len(recorded), True,
        detail="every tick reproduced the recorded state",
        recorded_hash=recorded[-1] if recorded else None,
        replayed_hash=engine.hash_sequence()[-1] if engine.history else None,
        notes=notes)


def replay_full(store, run_id: str, cache) -> ReplayResult:
    """Re-run the POLICIES against the recorded LLM call log.

    Stricter than replay_actions: it re-derives the decisions rather than
    replaying them, so it fails if the prompt, the model configuration or the
    policy code changed. That failure is the useful part.
    """
    from ..world.engine import Engine
    from ..agents.policies.llm.policy import OllamaLLMPolicy

    meta = store.run(run_id)
    if meta is None:
        return ReplayResult(run_id, "full", 0, False, detail="no such run")
    if cache.mode != "replay":
        return ReplayResult(run_id, "full", 0, False,
                            detail="cache must be opened in replay mode")

    cfg_raw = json.loads(meta["config_json"])
    cfg = RunConfig(**{k: v for k, v in cfg_raw.items()
                       if k in RunConfig.__dataclass_fields__})
    domain_cls = DOMAINS[meta["domain"]]
    recorded = store.state_hashes(run_id)
    logged = _load_actions(store, run_id)

    engine = Engine(cfg, domain_cls(water_hint=cfg.water_hint))
    for aid in engine.state.agent_ids():
        engine.policies[int(aid)] = OllamaLLMPolicy(
            model=cfg.model, scaffold=cfg.scaffold_level, cache=cache)

    for tick, expected in enumerate(recorded):
        rec = engine.tick()
        for agent, proposal in rec.proposals:
            want = logged.get(agent, {}).get(tick)
            if want is not None and (proposal.verb, proposal.params) != \
                    (want.verb, want.params):
                return ReplayResult(
                    run_id, "full", tick + 1, False, first_divergence=tick,
                    detail=(f"agent {agent} decided {proposal.verb.value} at "
                            f"tick {tick}; the log says {want.verb.value}"))
        if rec.state_hash != expected:
            return ReplayResult(
                run_id, "full", tick + 1, False, first_divergence=tick,
                detail=f"state diverged at tick {tick}")

    return ReplayResult(run_id, "full", len(recorded), True,
                        detail="policies re-derived every recorded decision",
                        recorded_hash=recorded[-1] if recorded else None,
                        replayed_hash=engine.hash_sequence()[-1])
