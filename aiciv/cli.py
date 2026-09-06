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
from .information import scan_banned
from .knowledge.shadow import ShadowVerifier, adoption_metrics
from .metrics.adoption import adoption_report
from .metrics.definitions import civilization_report
from .persistence.replay import replay_actions
from .persistence.store import RunStore
from .tasks.board import DEFAULT_BOARD, BoardError, TaskBoard
from .tasks.spec import score_board
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
        # Steering, recorded in the run itself rather than only in the task
        # board. A run file has to be able to say on its own that its agents
        # were told what to aim at; a reader six months from now may not have
        # the board, and would otherwise read a steered run as a free one.
        #
        # `steered` is whether anything actually REACHED an agent, which is
        # narrower than whether a brief was passed. Only the llm policy reads a
        # prompt, so a brief handed to a scripted run was configured and never
        # delivered -- `directives` still records what was asked for, and
        # calling that run steered would attribute to its agents an instruction
        # they were never given.
        "steered": bool(cfg.directives) and cfg.policy == "llm",
        "brief_id": cfg.brief_id or None,
        "directives": list(cfg.directives),
        "directives_reached_agents": cfg.policy == "llm",
        "directive_method_terms": sorted({
            t for d in cfg.directives for t in scan_banned(d)}),
    }


# --- run -------------------------------------------------------------------

def _announce_brief(brief_id: str, directives: tuple[str, ...],
                    policy: str) -> None:
    """Say plainly, at launch, that this run is not an unsteered one.

    Printed to stderr before the first tick, because the moment to find out
    that a run was steered is before it costs twenty hours -- not while reading
    its report afterwards.
    """
    if not directives:
        # A brief whose tasks all have empty directives says nothing out loud.
        # The tasks are still scored, but this run is not steered and must not
        # be reported as if it were.
        print(f"BRIEF {brief_id}: no directive text, so nothing is said to the "
              f"agents. Its tasks are scored; the run is not steered.",
              file=sys.stderr)
        return
    print(f"BRIEFED RUN ({brief_id}): the agents are being told:",
          file=sys.stderr)
    for d in directives:
        print(f"  - {d}", file=sys.stderr)
    if policy != "llm":
        # A scripted policy never reads a prompt. Saying nothing here would let
        # someone believe they had steered a run that could not have heard them.
        print("  NOTE: policy is not 'llm', so nothing reads the prompt. The "
              "tasks will be scored, but these agents were not told anything.",
              file=sys.stderr)
    leaked = sorted({t for d in directives for t in scan_banned(d)})
    if leaked:
        print(f"  WARNING: the brief contains method vocabulary {leaked}. "
              f"That is the operator's choice, but this run can no longer be "
              f"compared against a rules_only run that was not given it.",
              file=sys.stderr)


def cmd_run(args) -> int:
    import aiciv.agents.policies  # noqa: F401  registers every policy
    from .world.engine import Engine

    try:
        board = TaskBoard(pathlib.Path(args.task_board))
        directives = board.directives(args.brief) if args.brief else ()
    except BoardError as e:
        # A mistyped brief id must not cost a stack trace, and must not
        # silently start an UNSTEERED run under a name the operator believes
        # was steered -- that run would then be pooled with real ones.
        raise SystemExit(f"cannot start: {e}") from e
    if args.brief:
        _announce_brief(args.brief, directives, args.policy)

    cfg = RunConfig(
        seed=args.seed, ticks=args.ticks, n_agents=args.agents,
        policy=args.policy, domain=args.domain,
        water_hint=not args.no_water_hint,
        teaching_enabled=not args.no_teaching, arm=args.arm,
        brief_id=args.brief or "", directives=directives)
    domain = build_domain(args.domain, cfg.water_hint)
    engine = Engine(cfg, domain)

    if args.policy == "llm":
        # Swap in the model-backed policy. The registry builds policies with no
        # arguments, so the model, scaffold and timeout are wired here.
        from .agents.policies.llm.policy import OllamaLLMPolicy
        for aid in engine.state.agent_ids():
            engine.policies[int(aid)] = OllamaLLMPolicy(
                model=args.model, host=args.host, scaffold=cfg.scaffold_level,
                timeout=args.llm_timeout, directives=cfg.directives)

    run_id = args.run_id or f"{cfg.arm}_s{cfg.seed}_{cfg.policy}"

    if args.brief:
        # Attach before the run starts, not after. A twenty-hour run that is
        # killed at hour twelve must still be on record as having been
        # steered; recording that only on a clean exit would leave the most
        # confusing case -- a partial steered run -- looking unsteered.
        board.attach_brief(run_id, args.brief)

    if args.checkpoint_every > 0 and not args.no_save:
        # A long model-driven run is measured in hours. Without checkpointing,
        # a failure at hour twenty discards everything: the evidence exists
        # only in memory. Partial state is written periodically so a lost run
        # costs the remaining ticks rather than all of them.
        import time as _time
        store = RunStore(RUNS / f"{run_id}.sqlite")
        started = _time.time()
        for tick in range(cfg.ticks):
            engine.tick()
            done = tick + 1
            if done % args.checkpoint_every == 0 or done == cfg.ticks:
                store.save_run(run_id, engine,
                               manifest=manifest(cfg, domain, engine),
                               status="running" if done < cfg.ticks
                               else "complete")
                rate = (_time.time() - started) / done
                print(f"  tick {done}/{cfg.ticks}  "
                      f"trials={len(engine.trials)}  "
                      f"claims={sum(v for v in engine.kb.counts().values())}  "
                      f"{rate:.1f}s/tick  "
                      f"eta {(cfg.ticks - done) * rate / 60:.0f}min",
                      file=sys.stderr, flush=True)
        store.close()
    else:
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
    if not args.no_save:
        # Scored from the file just written, not from the engine in memory:
        # what the dashboard will show has to be what was actually persisted.
        store = RunStore(RUNS / f"{run_id}.sqlite", read_only=True)
        try:
            out["tasks"] = score_board(board.run_tasks(run_id), store, run_id)
        finally:
            store.close()
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


# --- tasks -----------------------------------------------------------------

def cmd_tasks(args) -> int:
    """Read the board. Authoring lives in the dashboard, and deliberately:
    a task is a claim about what you were trying to find out, and it should be
    written where you can see the run it is about."""
    board = TaskBoard(pathlib.Path(args.task_board))
    if args.run is None:
        print(json.dumps({"board": str(board.path),
                          "briefs": board.briefs()}, indent=2))
        return 0
    path = RUNS / f"{args.run}.sqlite"
    if not path.exists():
        print(f"no such run: {path}", file=sys.stderr)
        return 1
    store = RunStore(path, read_only=True)
    try:
        print(json.dumps(score_board(board.run_tasks(args.run), store, args.run),
                         indent=2))
    finally:
        store.close()
    return 0


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
    r.add_argument("--model", default="assistant:latest",
                   help="only used when --policy llm")
    r.add_argument("--host", default="http://127.0.0.1:11434")
    r.add_argument("--llm-timeout", type=float, default=300.0)
    r.add_argument("--brief", default=None,
                   help="brief id whose tasks are put IN THE AGENTS' PROMPT. "
                        "This makes the run steered and not comparable with an "
                        "unsteered one; it is recorded in the manifest as such")
    r.add_argument("--task-board", default=str(DEFAULT_BOARD),
                   help="where the task board JSON lives")
    r.add_argument("--checkpoint-every", type=int, default=0,
                   help="save partial state every N ticks; essential for runs "
                        "measured in hours")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="re-execute a recorded run and verify it")
    rp.add_argument("--run", required=True)
    rp.add_argument("--mode", default="actions", choices=["actions", "full"])
    rp.set_defaults(func=cmd_replay)

    m = sub.add_parser("metrics", help="print the recorded metrics for a run")
    m.add_argument("--run", required=True)
    m.set_defaults(func=cmd_metrics)

    t = sub.add_parser("tasks", help="score the task board for a run")
    t.add_argument("--run", default=None,
                   help="run id to score; omit to list the briefs instead")
    t.add_argument("--task-board", default=str(DEFAULT_BOARD))
    t.set_defaults(func=cmd_tasks)

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
