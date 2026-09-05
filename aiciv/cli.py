"""Command line entry point.

Every subcommand here is implemented. Where a capability is genuinely absent --
no model server, say -- the command says so plainly rather than failing with a
stack trace or, worse, quietly producing something that looks like a result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
from collections import Counter
from typing import Any

from .config import RunConfig
from .knowledge.shadow import ShadowVerifier, adoption_metrics
from .metrics.adoption import adoption_report
from .metrics.definitions import civilization_report
from .persistence.replay import replay_actions
from .persistence.store import RunStore
from .world.domains.agronomy import AgronomyDomain
from .world.domains.synthetic import SyntheticDomain

RUNS = pathlib.Path("runs")
DOMAINS = {"synthetic": SyntheticDomain, "agronomy": AgronomyDomain}


def _git() -> tuple[str, bool]:
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5)
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=5)
        return (c.stdout.strip() or "unknown", bool(d.stdout.strip()))
    except Exception:
        return ("unknown", False)


def build_domain(name: str, water_hint: bool):
    if name not in DOMAINS:
        raise SystemExit(f"unknown domain {name!r}; have {sorted(DOMAINS)}")
    return DOMAINS[name](water_hint=water_hint)


def manifest(cfg: RunConfig, domain, engine) -> dict[str, Any]:
    """Everything needed to reproduce this run months from now."""
    commit, dirty = _git()
    lexicon = getattr(engine, "_lexicon", None)
    return {
        "git_commit": commit, "dirty": dirty,
        "config_hash": cfg.hash, "code_version": "0.1.0",
        "domain": domain.name, "domain_version": domain.version,
        "verification_spec_hash": domain.verification.hash,
        "verification_calibrated_by": domain.verification.calibrated_by,
        "lexicon_hash": lexicon.hash if lexicon else None,
        "prior_probe_id": None,
        "seed": cfg.seed, "ticks": cfg.ticks, "agents": cfg.n_agents,
        "policy": cfg.policy, "scaffold_level": cfg.scaffold_level,
        "teaching_enabled": cfg.teaching_enabled, "water_hint": cfg.water_hint,
        "arm": cfg.arm, "python": platform.python_version(),
    }


# --- run -------------------------------------------------------------------

def cmd_run(args) -> int:
    import aiciv.agents.policies  # noqa: F401  registers every policy
    from .world.engine import Engine

    cfg = RunConfig(
        seed=args.seed, ticks=args.ticks, n_agents=args.agents,
        policy=args.policy, domain=args.domain,
        water_hint=not args.no_water_hint,
        teaching_enabled=not args.no_teaching, arm=args.arm)
    domain = build_domain(args.domain, cfg.water_hint)
    engine = Engine(cfg, domain)
    engine.run()

    report = civilization_report(engine, domain)

    rows_by_agent: dict[int, list[dict]] = {}
    for row in engine.rows():
        rows_by_agent.setdefault(int(row["agent_id"]), []).append(row)
    claims_by_agent: dict[int, list] = {}
    for r in engine.kb:
        if r.registered_tick is not None:
            claims_by_agent.setdefault(int(r.claim.author), []).append(r.claim)
    report["adoption"] = adoption_report(rows_by_agent, claims_by_agent)

    shadow = ShadowVerifier(domain.verification, domain)
    findings = shadow.scan(engine.rows())
    report["shadow"] = {
        **adoption_metrics(
            sum(1 for r in engine.kb
                if r.state.value in ("confirmed", "generalized")),
            sum(1 for f in findings if f.stage == "confirmed")),
        "findings": [f.describe() for f in findings],
        "caveat": ("the shadow scanner forms marginal single-variable "
                   "hypotheses only, so its count is not like-for-like with "
                   "agent claims; see knowledge/shadow.py"),
    }
    report["capabilities"] = {
        k: v for k, v in engine.capabilities.report().items() if k != "events"}

    run_id = args.run_id or f"{cfg.arm}_s{cfg.seed}_{cfg.policy}"
    if not args.no_save:
        store = RunStore(RUNS / f"{run_id}.sqlite")
        store.save_run(run_id, engine,
                       manifest=manifest(cfg, domain, engine), metrics=report)
        store.close()

    hashes = engine.hash_sequence()
    out = {
        "run_id": run_id,
        "saved": None if args.no_save else str(RUNS / f"{run_id}.sqlite"),
        "manifest": manifest(cfg, domain, engine),
        "final_state_hash": hashes[-1] if hashes else None,
        "trials": len(engine.trials),
        "trials_by_stratum": dict(Counter(t.stratum for t in engine.trials)),
        "rejections": engine.rejection_summary(),
        "claim_states": {k: v for k, v in engine.kb.counts().items() if v},
        "metrics": report,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


# --- replay ----------------------------------------------------------------

def cmd_replay(args) -> int:
    path = (pathlib.Path(args.run) if args.run.endswith(".sqlite")
            else RUNS / f"{args.run}.sqlite")
    if not path.exists():
        print(f"no such run file: {path}", file=sys.stderr)
        return 1
    run_id = path.stem
    store = RunStore(path)
    try:
        if args.mode == "actions":
            result = replay_actions(store, run_id)
        else:
            from .agents.policies.llm.cache import LLMCache
            from .persistence.replay import replay_full
            cache = LLMCache(path=RUNS / f"{run_id}_llm.sqlite", mode="replay")
            result = replay_full(store, run_id, cache)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.matched else 1
    finally:
        store.close()


# --- metrics / experiments -------------------------------------------------

def cmd_metrics(args) -> int:
    path = RUNS / f"{args.run}.sqlite"
    if not path.exists():
        print(f"no such run: {path}", file=sys.stderr)
        return 1
    store = RunStore(path)
    try:
        report = store.metrics(args.run)
        if report is None:
            print("no metrics recorded for this run", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, default=str))
        return 0
    finally:
        store.close()


def cmd_experiment(args) -> int:
    from dataclasses import replace as dc_replace

    from .experiment.runner import DEFAULT_ARMS, Experiment

    arms = list(DEFAULT_ARMS)
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = [a for a in DEFAULT_ARMS if a.name in wanted]
        missing = wanted - {a.name for a in arms}
        if missing:
            raise SystemExit(f"unknown arms: {sorted(missing)}")

    exp = Experiment(
        name=args.name,
        arms=[dc_replace(a, domain=args.domain) for a in arms],
        seeds=[int(s) for s in args.seeds.split(",")],
        ticks=args.ticks, n_agents=args.agents, out_dir=RUNS)

    def progress(arm: str, seed: int) -> None:
        print(f"  {arm} seed {seed} ...", file=sys.stderr, flush=True)

    print(f"running {len(exp.arms)} arms x {len(exp.seeds)} seeds "
          f"x {exp.ticks} ticks", file=sys.stderr)
    report = exp.report(exp.run(progress=progress))
    path = exp.save(report)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nsaved {path}", file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    path = RUNS / f"experiment_{args.experiment}.json"
    if not path.exists():
        print(f"no such experiment report: {path}", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(report.get("contrasts", {}), indent=2))
    return 0


# --- prior probe -----------------------------------------------------------

def cmd_prior_probe(args) -> int:
    from .agents.policies.llm.policy import OllamaClient, OllamaError
    from .experiment.prior_probe import run_probe, save

    domain = build_domain(args.domain, True)
    # A generous default: the prediction task asks for twenty numbers, which
    # is a long generation, and a reasoning model spends most of its budget
    # thinking before it writes anything at all.
    client = OllamaClient(model=args.model, host=args.host,
                          timeout=args.timeout)

    def ask(prompt: str) -> str:
        # structured=False: the probe asks open questions and must receive
        # whatever the model would actually say. Two of the three tasks expect
        # JSON of their own shape, and the third expects prose.
        # Reasoning models spend most of their budget on <think>; 600 tokens
        # left them deliberating with nothing to show for it.
        return client.chat("You are answering questions about farming.",
                           prompt,
                           {"temperature": 0.2, "num_predict": args.max_tokens},
                           structured=False)

    try:
        baseline = run_probe(domain, ask, model=args.model,
                             scaffold=args.scaffold)
    except OllamaError as e:
        print(f"cannot reach a model: {e}", file=sys.stderr)
        print("The prior probe needs a running Ollama. Without it we cannot say "
              "what the model already knew, and discovery counts from any run "
              "would be uninterpretable.", file=sys.stderr)
        return 1

    path = save(baseline, RUNS / "priors")
    print(json.dumps(baseline.to_dict(), indent=2))
    print(f"\nsaved {path}", file=sys.stderr)
    return 0


# --- entry point -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aiciv", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a simulation headlessly")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--ticks", type=int, default=1200)
    r.add_argument("--agents", type=int, default=5)
    r.add_argument("--policy", default="scripted_factorial")
    r.add_argument("--domain", default="synthetic", choices=sorted(DOMAINS))
    r.add_argument("--run-id", default=None)
    r.add_argument("--no-water-hint", action="store_true",
                   help="ablation: remove the crop_health signal")
    r.add_argument("--no-teaching", action="store_true",
                   help="control arm: TEACH is rejected")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--arm", default="default")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="re-execute a recorded run and verify it")
    rp.add_argument("--run", required=True)
    rp.add_argument("--mode", default="actions", choices=["actions", "full"])
    rp.set_defaults(func=cmd_replay)

    m = sub.add_parser("metrics", help="print the recorded metrics for a run")
    m.add_argument("--run", required=True)
    m.set_defaults(func=cmd_metrics)

    e = sub.add_parser("experiment", help="run arms x paired seeds")
    e.add_argument("--name", default="transfer")
    e.add_argument("--arms", default=None,
                   help="comma-separated subset of the default arms")
    e.add_argument("--seeds", default="42,43,44")
    e.add_argument("--ticks", type=int, default=1200)
    e.add_argument("--agents", type=int, default=5)
    e.add_argument("--domain", default="synthetic", choices=sorted(DOMAINS))
    e.set_defaults(func=cmd_experiment)

    c = sub.add_parser("compare", help="print the contrasts from an experiment")
    c.add_argument("--experiment", required=True)
    c.set_defaults(func=cmd_compare)

    pp = sub.add_parser("prior-probe",
                        help="measure what a model knows BEFORE it farms")
    pp.add_argument("--model", default="assistant:latest")
    pp.add_argument("--host", default="http://127.0.0.1:11434")
    pp.add_argument("--domain", default="synthetic", choices=sorted(DOMAINS))
    pp.add_argument("--scaffold", default="rules_only")
    pp.add_argument("--timeout", type=float, default=300.0,
                    help="seconds per call")
    pp.add_argument("--max-tokens", type=int, default=600,
                    help="raise for reasoning models: <think> "
                         "traces consume the budget before the answer")
    pp.set_defaults(func=cmd_prior_probe)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
