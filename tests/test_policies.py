"""Phase 4 gate: the control arms behave as the design predicts.

The whole point of the deceptive landscape is that method matters. If a naive
one-factor-at-a-time searcher did as well as a factorial one, there would be
nothing for an LLM to be better or worse at, and every later comparison would
be uninterpretable.

These runs use 1000 ticks rather than 400. The scripted arms cost no LLM
tokens, so buying enough trials to characterise them properly is nearly free;
the 400-tick budget in the plan exists for the LLM arms.
"""

from __future__ import annotations

import pytest

import aiciv.agents.policies.scripted  # noqa: F401  registers both policies
from aiciv.config import RunConfig
from aiciv.knowledge.claim import ClaimState
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine

TICKS = 1000
GLOBAL_OPTIMA = {("CLOVER", 4), ("CLOVER", 5), ("BEANS", 1), ("BEANS", 2)}
TRAP = ("MARIGOLD", 3)


def run(policy: str, seed: int = 42) -> Engine:
    e = Engine(RunConfig(seed=seed, ticks=TICKS, n_agents=5, policy=policy),
               SyntheticDomain())
    e.run()
    return e


def picks(engine: Engine) -> list[tuple]:
    out = []
    for aid in engine.state.agent_ids():
        p = engine.policies[int(aid)]
        out.append(getattr(p, "best", None)
                   or (getattr(p, "best_companion", None),
                       getattr(p, "best_spacing", None)))
    return out


def banked(engine: Engine) -> list:
    """Claims that reached CONFIRMED or better."""
    return [r for r in engine.kb
            if r.state in (ClaimState.CONFIRMED, ClaimState.GENERALIZED)]


@pytest.fixture(scope="module")
def factorial():
    return run("scripted_factorial")


@pytest.fixture(scope="module")
def greedy():
    return run("scripted_greedy")


# --- the factorial arm must actually find the interaction ------------------

def test_factorial_finds_a_global_optimum(factorial):
    hits = sum(1 for p in picks(factorial) if p in GLOBAL_OPTIMA)
    assert hits >= 4, f"only {hits}/5 agents found a global optimum: {picks(factorial)}"


def test_factorial_banks_confirmed_knowledge(factorial):
    """A discovery that never reaches CONFIRMED is not public knowledge.

    This is the end-to-end proof that the whole pipeline works: search ->
    pre-registration -> evidence -> independent replication -> CONFIRMED.
    """
    conf = banked(factorial)
    assert conf, f"nothing confirmed; states were {factorial.kb.counts()}"
    for r in conf:
        cond = r.claim.spec.conditions
        cell = (cond["companion"]["value"], cond["spacing"]["value"])
        assert cell in GLOBAL_OPTIMA, f"confirmed a non-optimal cell {cell}"


def test_factorial_confirmed_claims_are_independently_replicated(factorial):
    """CONFIRMED requires an agent other than the author, by construction."""
    for r in banked(factorial):
        assert r.replicator is not None
        assert r.replicator != int(r.claim.author)


# --- the greedy arm must be trapped, and measurably worse ------------------

def test_greedy_is_substantially_trapped(greedy):
    """Not necessarily every agent: a noisy spacing sweep sometimes lands
    somewhere the companion sweep can escape from. What matters is that a real
    share converge on the trap, which naive method cannot leave."""
    p = picks(greedy)
    trapped = sum(1 for x in p if x == TRAP)
    assert trapped >= 2, f"expected the trap to capture agents, got {p}"


def test_greedy_banks_a_true_but_inferior_claim(greedy):
    """The trap is a REAL effect (+0.25 synergy), so greedy is not wrong -- it
    is merely stuck. Its confirmed claims should be genuine and small."""
    conf = banked(greedy)
    trap_claims = [
        r for r in conf
        if (r.claim.spec.conditions["companion"]["value"],
            r.claim.spec.conditions["spacing"]["value"]) == TRAP
    ]
    assert trap_claims, f"greedy confirmed nothing at the trap: {picks(greedy)}"
    for r in trap_claims:
        eff = r.verdicts["supported"]["effect"]
        assert 0 < eff < 0.7, f"trap effect {eff} is not small-but-real"


def test_factorial_beats_greedy_on_confirmed_effect_size(factorial, greedy):
    """The quantitative statement the landscape exists to produce.

    Both arms bank true knowledge. The factorial arm banks *better* knowledge,
    and the gap is what any LLM policy will be measured against.
    """
    def best_effect(engine):
        effs = [r.verdicts["supported"]["effect"] for r in banked(engine)
                if "supported" in r.verdicts]
        return max(effs) if effs else 0.0

    f_eff, g_eff = best_effect(factorial), best_effect(greedy)
    assert f_eff > g_eff, f"factorial {f_eff:.3f} did not beat greedy {g_eff:.3f}"
    assert f_eff - g_eff > 0.3, (
        f"the arms are too close ({f_eff:.3f} vs {g_eff:.3f}); the landscape "
        f"has stopped discriminating between methods"
    )


# --- both arms are honest about how they got there ------------------------

def test_no_policy_imports_the_domain():
    """A control policy that peeked at the yield function would invalidate
    every comparison made against it."""
    import ast
    import pathlib
    pkg = pathlib.Path(aiciv.agents.policies.scripted.__file__).parent
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            assert "domains" not in mod, f"{path.name} imports {mod}"


def test_policies_produce_evidence_across_all_strata(factorial):
    strata = {t.stratum for t in factorial.trials}
    assert strata == {"discovery", "confirmation", "holdout"}
