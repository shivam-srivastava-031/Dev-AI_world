"""Phase 8 gate: arms, paired seeds, and honest comparison."""

from __future__ import annotations

import numpy as np
import pytest

from aiciv.experiment.runner import (
    DEFAULT_ARMS, PRIMARY_METRICS, Arm, Experiment, RunResult, compare,
    paired_bootstrap, run_one,
)


# --- pairing --------------------------------------------------------------

def test_bootstrap_is_paired_not_pooled():
    """Pairing is the point. Worlds differ enormously between seeds, and an
    unpaired comparison spends most of its power on that rather than on the
    thing being tested.

    Here every pair differs by exactly +2 while the raw values span 10 to 210.
    Pooled, the difference would vanish into the between-seed spread.
    """
    a = [10.0, 60.0, 110.0, 160.0, 210.0]
    b = [8.0, 58.0, 108.0, 158.0, 208.0]
    res = paired_bootstrap(a, b, reps=2000, seed=0)
    assert res["effect"] == pytest.approx(2.0, abs=1e-9)
    assert res["excludes_zero"]
    assert res["ci"][0] > 0

    pooled_sd = float(np.std(a + b))
    assert pooled_sd > 50, "the between-seed spread should dwarf the effect"


def test_bootstrap_reports_no_effect_honestly():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0, 1, 12))
    b = list(rng.normal(0, 1, 12))
    res = paired_bootstrap(a, b, reps=3000, seed=1)
    assert not res["excludes_zero"]
    assert res["ci"][0] < 0 < res["ci"][1]


def test_bootstrap_declines_to_compare_too_few_pairs():
    res = paired_bootstrap([1.0], [0.0])
    assert res["effect"] is None
    assert "too few" in res["note"]


def test_bootstrap_drops_unpaired_seeds():
    res = paired_bootstrap([1.0, None, 3.0], [0.0, 5.0, 1.0], reps=500)
    assert res["n_pairs"] == 2


def test_bootstrap_is_deterministic():
    a, b = [3.0, 5.0, 9.0, 2.0], [1.0, 4.0, 7.0, 1.0]
    first = paired_bootstrap(a, b, reps=1000, seed=7)
    assert paired_bootstrap(a, b, reps=1000, seed=7) == first


# --- arms -----------------------------------------------------------------

def test_default_arms_cover_the_designed_contrasts():
    """The 2x2 the plan calls for, plus the floor to measure everything from."""
    names = {a.name for a in DEFAULT_ARMS}
    assert {"social_teaching", "social_no_teaching"} <= names   # teaching effect
    assert {"scripted_factorial", "scripted_greedy"} <= names   # method effect
    assert "random" in names                                     # the floor


def test_teaching_arms_differ_only_in_teaching():
    on = next(a for a in DEFAULT_ARMS if a.name == "social_teaching")
    off = next(a for a in DEFAULT_ARMS if a.name == "social_no_teaching")
    assert on.policy == off.policy
    assert on.teaching and not off.teaching
    assert on.domain == off.domain and on.water_hint == off.water_hint


def test_arm_config_carries_its_switches():
    arm = Arm("x", "random", teaching=False, water_hint=False)
    cfg = arm.config(seed=7, ticks=50, n_agents=3)
    assert cfg.seed == 7 and cfg.ticks == 50 and cfg.n_agents == 3
    assert not cfg.teaching_enabled and not cfg.water_hint
    assert cfg.arm == "x"


# --- a real (small) experiment -------------------------------------------

@pytest.fixture(scope="module")
def small_results():
    arms = [Arm("a_random", "random"),
            Arm("b_factorial", "scripted_factorial")]
    return [run_one(arm, seed, ticks=200, n_agents=5)
            for arm in arms for seed in (42, 43)]


def test_run_one_produces_gradeable_metrics(small_results):
    for r in small_results:
        assert r.metrics["knowledge"] is not None
        assert r.metrics["grading"]["banked"] >= 0
        assert r.manifest["final_state_hash"]
        assert r.scalar("collective.trials") > 0


def test_scalar_lookup_handles_missing_paths(small_results):
    r = small_results[0]
    assert r.scalar("knowledge.banked") is not None
    assert r.scalar("nope.nothing") is None
    assert r.scalar("knowledge.by_state") is None      # not a number


def test_arms_run_on_the_same_seeds(small_results):
    by_arm: dict[str, set[int]] = {}
    for r in small_results:
        by_arm.setdefault(r.arm, set()).add(r.seed)
    assert len(set(map(frozenset, by_arm.values()))) == 1, (
        "arms must share seeds or the comparison is not paired")


def test_compare_produces_intervals_for_every_primary_metric(small_results):
    out = compare(small_results, "b_factorial", "a_random")
    assert out["seeds"] == [42, 43]
    for m in PRIMARY_METRICS:
        assert m in out["metrics"]
        entry = out["metrics"][m]
        assert "effect" in entry and "ci" in entry


def test_report_summarises_arms_and_contrasts(small_results):
    exp = Experiment(name="t", arms=[Arm("a_random", "random"),
                                     Arm("b_factorial", "scripted_factorial")],
                     seeds=[42, 43], ticks=200)
    report = exp.report(small_results)
    assert set(report["summary"]) == {"a_random", "b_factorial"}
    for stats in report["summary"]["b_factorial"].values():
        assert {"n", "mean", "median", "min", "max"} <= set(stats)


def test_report_saves_and_reloads(small_results, tmp_path):
    import json
    exp = Experiment(name="t", arms=[Arm("a_random", "random")],
                     seeds=[42, 43], ticks=200, out_dir=tmp_path)
    path = exp.save(exp.report(small_results))
    assert json.loads(path.read_text(encoding="utf-8"))["experiment"] == "t"
