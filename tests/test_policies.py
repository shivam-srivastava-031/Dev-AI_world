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


def established(engine: Engine) -> list:
    """Claims that passed at least the first verification gate.

    Deliberately not "CONFIRMED or better": whether a claim has had time to be
    independently replicated depends on run length and on how big its effect
    is, neither of which is what these tests are about. Greedy's trap effect is
    +0.25 against BEANS' +1.00, so it clears SUPPORTED much later.
    """
    return [r for r in engine.kb
            if r.state in (ClaimState.SUPPORTED, ClaimState.CONFIRMED,
                           ClaimState.GENERALIZED, ClaimState.CONTESTED)]


def confirmed(engine: Engine) -> list:
    """Claims that reached CONFIRMED or better -- i.e. independently replicated."""
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
    conf = confirmed(factorial)
    assert conf, f"nothing confirmed; states were {factorial.kb.counts()}"
    for r in conf:
        cond = r.claim.spec.conditions
        cell = (cond["companion"]["value"], cond["spacing"]["value"])
        assert cell in GLOBAL_OPTIMA, f"confirmed a non-optimal cell {cell}"


def test_factorial_confirmed_claims_are_independently_replicated(factorial):
    """CONFIRMED requires an agent other than the author, by construction."""
    for r in confirmed(factorial):
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
    conf = established(greedy)
    trap_claims = [
        r for r in conf
        if (r.claim.spec.conditions["companion"]["value"],
            r.claim.spec.conditions["spacing"]["value"]) == TRAP
    ]
    assert trap_claims, f"greedy established nothing at the trap: {picks(greedy)}"
    for r in trap_claims:
        eff = r.verdicts["supported"]["effect"]
        assert 0 < eff < 0.7, f"trap effect {eff} is not small-but-real"


def _effects(engine) -> list[float]:
    return sorted(r.verdicts["supported"]["effect"] for r in established(engine)
                  if "supported" in r.verdicts
                  and r.verdicts["supported"].get("effect") is not None)


def test_factorial_beats_greedy_on_typical_effect_size(factorial, greedy):
    """The quantitative statement the landscape exists to produce.

    MEDIAN, not max. Max measures the single luckiest agent, and greedy's noisy
    spacing sweep occasionally lands one agent somewhere it can escape from --
    that agent then out-scores every factorial agent, while four of its five
    peers sit in the trap. Measured that way greedy looks better than factorial,
    which is exactly backwards as a statement about method.

    Observed (seed 42, 1000 ticks):
      factorial 5/5 on BEANS@2,     effects 0.899 - 1.127, median 1.072
      greedy    4/5 on MARIGOLD@3,  effects 0.443 - 0.529, plus one escapee
                                    at 1.255,              median 0.514
    """
    import statistics

    f, g = _effects(factorial), _effects(greedy)
    assert f and g
    f_med, g_med = statistics.median(f), statistics.median(g)

    assert f_med > g_med, (
        f"factorial median {f_med:.3f} did not beat greedy median {g_med:.3f}")
    assert f_med - g_med > 0.3, (
        f"the arms are too close ({f_med:.3f} vs {g_med:.3f}); the landscape "
        f"has stopped discriminating between methods"
    )


def test_the_typical_greedy_agent_is_worse_than_every_factorial_agent(factorial, greedy):
    """A sharper form of the same claim, robust to the lucky escapee: greedy's
    median agent must fall below the WORST factorial agent."""
    import statistics
    assert statistics.median(_effects(greedy)) < min(_effects(factorial))


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
