"""Command line entry point.

Only the subcommands whose phase has landed are implemented. The rest name the
phase that will provide them rather than failing with a stack trace, so the
gap between the plan and the code stays visible.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter

from . import __name__ as _pkg  # noqa: F401
from .config import RunConfig
from .world.domains.synthetic import SyntheticDomain
from .knowledge.shadow import ShadowVerifier, adoption_metrics
from .metrics.adoption import adoption_report
from .world.engine import Engine

NOT_YET = {
    "replay": "Phase 1 (persistence) -- needs the SQLite action log",
    "experiment": "Phase 8 -- needs arms and the paired-seed runner",
    "metrics": "Phase 5 -- needs the metrics package",
    "compare": "Phase 8 -- needs the bootstrap comparison",
    "prior-probe": "Phase 7 -- needs the Ollama policy",
}


def _git_commit() -> tuple[str, bool]:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5)
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=5)
        return (c.stdout.strip() or "unknown", bool(d.stdout.strip()))
    except Exception:
        return ("unknown", False)


def build_domain(name: str, water_hint: bool):
    if name == "synthetic":
        return SyntheticDomain(water_hint=water_hint)
    raise SystemExit(f"unknown domain {name!r} (agronomy lands in Phase 6)")


def manifest(cfg: RunConfig, domain, engine: Engine) -> dict:
    """Everything needed to reproduce this run months from now."""
    commit, dirty = _git_commit()
    return {
        "git_commit": commit,
        "dirty": dirty,
        "config_hash": cfg.hash,
        "code_version": "0.1.0",
        "domain": domain.name,
        "domain_version": domain.version,
        "verification_spec_hash": domain.verification.hash,
        "verification_calibrated_by": domain.verification.calibrated_by,
        "seed": cfg.seed,
        "ticks": cfg.ticks,
        "agents": cfg.n_agents,
        "policy": cfg.policy,
        "scaffold_level": cfg.scaffold_level,
        "teaching_enabled": cfg.teaching_enabled,
        "water_hint": cfg.water_hint,
        "arm": cfg.arm,
        "python": platform.python_version(),
        # prior_probe_id and lexicon_hash are added when Phases 5 and 7 land.
        "prior_probe_id": None,
        "lexicon_hash": None,
    }


def cmd_run(args: argparse.Namespace) -> int:
    import aiciv.agents.policies.random_policy  # noqa: F401  registers
    import aiciv.agents.policies.scripted  # noqa: F401  registers

    cfg = RunConfig(
        seed=args.seed, ticks=args.ticks, n_agents=args.agents,
        policy=args.policy, domain=args.domain, water_hint=not args.no_water_hint,
        teaching_enabled=not args.no_teaching, arm=args.arm,
    )
    domain = build_domain(args.domain, cfg.water_hint)
    engine = Engine(cfg, domain)
    engine.run()

    hashes = engine.hash_sequence()
    strata = Counter(t.stratum for t in engine.trials)

    # What the civilization actually banked as public knowledge.
    claims = []
    for r in sorted(engine.kb, key=lambda r: str(r.claim.claim_id)):
        row = r.summary()
        v = r.verdicts.get("supported", {})
        row["effect"] = v.get("effect")
        row["ci"] = ([v.get("ci_low"), v.get("ci_high")]
                     if v.get("ci_low") is not None else None)
        row["p_value"] = v.get("p_value")
        row["hedges_g"] = v.get("hedges_g")
        row["n"] = ([v.get("n_treat"), v.get("n_ctrl")]
                    if v.get("n_treat") is not None else None)
        row["reason"] = v.get("reason")
        claims.append(row)

    # The shadow verifier measures what the EVIDENCE supports, whether or not
    # any agent filed paperwork. Its ratio to agent-confirmed claims is
    # protocol adoption -- a finding, not an assumption.
    # Adoption graded on each agent's own information state: an agent that
    # never had a formulable comparison is excluded, not scored zero.
    rows_by_agent: dict[int, list[dict]] = {}
    for row in engine.rows():
        rows_by_agent.setdefault(int(row["agent_id"]), []).append(row)
    claims_by_agent: dict[int, list] = {}
    for r in engine.kb:
        if r.registered_tick is not None:
            claims_by_agent.setdefault(int(r.claim.author), []).append(r.claim)
    adoption = adoption_report(rows_by_agent, claims_by_agent)

    shadow = ShadowVerifier(domain.verification)
    findings = shadow.scan(engine.rows())
    shadow_confirmed = sum(1 for f in findings if f.stage == "confirmed")
    agent_confirmed = sum(
        1 for r in engine.kb
        if r.state.value in ("confirmed", "generalized"))
    out = {
        "manifest": manifest(cfg, domain, engine),
        "final_state_hash": hashes[-1] if hashes else None,
        "hash_sequence_digest": hashes[-1] if hashes else None,
        "trials": len(engine.trials),
        "trials_by_stratum": dict(strata),
        "events": len(engine.events),
        "rejections": engine.rejection_summary(),
        "claim_states": {k: v for k, v in engine.kb.counts().items() if v},
        "claims": claims,
        "adoption": adoption,
        "shadow": {
            **adoption_metrics(agent_confirmed, shadow_confirmed),
            "shadow_findings": [f.describe() for f in findings],
        },
        "agents": {
            engine.state.agents[a].name: {
                "skill": engine.state.agents[a].skill_farming,
                "food": engine.state.agents[a].food,
                "reputation": engine.state.agents[a].reputation,
            }
            for a in engine.state.agent_ids()
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aiciv", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a simulation headlessly")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--ticks", type=int, default=400)
    r.add_argument("--agents", type=int, default=5)
    r.add_argument("--policy", default="random")
    r.add_argument("--domain", default="synthetic")
    r.add_argument("--no-water-hint", action="store_true",
                   help="ablation: remove the crop_health water signal")
    r.add_argument("--no-teaching", action="store_true",
                   help="control arm: TEACH is rejected")
    r.add_argument("--arm", default="default")
    r.set_defaults(func=cmd_run)

    for name, phase in NOT_YET.items():
        s = sub.add_parser(name, help=f"not yet implemented -- {phase}")
        s.set_defaults(func=lambda a, _n=name, _p=phase: (
            print(f"'aiciv {_n}' is not implemented yet: {_p}", file=sys.stderr) or 1))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
