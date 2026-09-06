"""The task board: scoring, and the separation between the two kinds of task.

Two things are being defended here.

The first is arithmetic honesty. A task must not turn "not measured" into 0%,
and must not print a percentage for a target that does not fill up. Both are
easy to get wrong in a way that looks fine on a healthy run and lies on a
degraded one, which is the only kind of run where it matters.

The second is the steering boundary. An observer task is scored from the log
and never reaches an agent; a briefed task was in the prompt. If those two ever
became confusable, a steered run could be pooled with unsteered ones and every
comparison drawn from that pool would be wrong.
"""

from __future__ import annotations

import json

import pytest

import aiciv.agents.policies  # noqa: F401  registers every policy
from aiciv.agents.policies.llm.prompts import (
    BANNED_TERMS, assignment_block, scan_banned, system_prompt,
)
from aiciv.config import RunConfig
from aiciv.persistence.store import RunStore
from aiciv.tasks.board import BoardError, TaskBoard
from aiciv.tasks.spec import (
    METRICS, Task, catalogue, score, score_board, snapshot,
)
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine

SPACE = {"spacing": (2, 4, 6), "water": (0, 1), "plant_day": (0, 1, 2)}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("taskruns")
    e = Engine(RunConfig(seed=42, ticks=120, n_agents=5,
                         policy="scripted_factorial"), SyntheticDomain())
    e.run()
    s = RunStore(tmp / "r.sqlite")
    s.save_run("r1", e, manifest={"seed": 42})
    return s


def task(**kw) -> Task:
    base = dict(task_id="t1", title="t", metric="trials",
                comparator=">=", target=10.0)
    return Task(**{**base, **kw})


# --- the arithmetic must not invent anything -----------------------------

def test_an_unmeasured_quantity_is_none_not_zero():
    """The distinction the whole module exists for.

    A run with no trials has no mean yield. Reporting that as 0.00 kg would say
    the agents grew nothing when in fact they have not harvested yet, and on a
    degraded run -- where the model was silent and nothing happened -- the two
    readings point at opposite diagnoses.
    """
    scored = score(task(metric="mean_yield", comparator=">=", target=3.0),
                   {"mean_yield": None})
    assert scored["value"] is None
    assert scored["progress"] is None
    assert scored["measurable"] is False
    assert scored["done"] is False
    assert "not measurable yet" in scored["detail"]


def test_a_ceiling_gets_no_percentage():
    """Being at 30% of a 15% limit is not "half done"; it is not done.

    Only a counting metric with a floor target has a meaningful fraction, so
    everything else reports met/not-met and the dashboard draws a pill rather
    than a bar. If progress were a number here the bar would fill up as the
    task got FURTHER from being met.
    """
    over = score(task(metric="invalid_action_rate", comparator="<=",
                      target=0.15), {"invalid_action_rate": 0.30})
    assert over["kind"] == "threshold"
    assert over["progress"] is None
    assert over["done"] is False

    under = score(task(metric="invalid_action_rate", comparator="<=",
                       target=0.15), {"invalid_action_rate": 0.11})
    assert under["done"] is True
    assert under["progress"] is None


def test_a_rate_with_a_floor_is_also_a_threshold():
    """A rate does not accumulate. Half of a target productivity rate is not
    half of the way to holding it -- rates move both directions."""
    scored = score(task(metric="productive_action_rate", comparator=">=",
                        target=0.5), {"productive_action_rate": 0.25})
    assert scored["kind"] == "threshold"
    assert scored["progress"] is None


def test_a_counting_floor_gets_a_real_fraction():
    scored = score(task(metric="trials", comparator=">=", target=200),
                   {"trials": 50})
    assert scored["kind"] == "accumulate"
    assert scored["progress"] == pytest.approx(0.25)
    assert scored["done"] is False


def test_progress_is_capped_at_one():
    """Overshooting a target is done, not 340% done. A bar that ran past its
    track would be the visible symptom; the invisible one is a board mean
    dragged above 1.0 by a single task nobody set a ceiling on."""
    scored = score(task(metric="trials", target=100), {"trials": 340})
    assert scored["progress"] == 1.0
    assert scored["done"] is True


def test_a_zero_target_does_not_divide_by_zero():
    scored = score(task(metric="trials", target=0), {"trials": 0})
    assert scored["progress"] == 1.0
    assert scored["done"] is True


def test_board_mean_excludes_thresholds(store):
    """Averaging met/not-met in as 1 and 0 would invent a number.

    A board of one counting task at 50% and four unmet ceilings is not "10%
    complete". The mean covers the counting tasks; the met count covers the
    rest, and they are reported separately.
    """
    tasks = [task(task_id="a", metric="trials", target=1000),
             task(task_id="b", metric="invalid_action_rate",
                  comparator="<=", target=0.0)]
    board = score_board(tasks, store, "r1")
    fractions = [t["progress"] for t in board["tasks"]
                 if t["progress"] is not None]
    assert len(fractions) == 1
    assert board["mean_progress"] == round(fractions[0], 4)


def test_mean_progress_is_none_when_nothing_has_a_fraction(store):
    board = score_board(
        [task(metric="invalid_action_rate", comparator="<=", target=0.5)],
        store, "r1")
    assert board["mean_progress"] is None


# --- the metrics themselves ----------------------------------------------

def test_every_catalogued_metric_is_produced_by_the_snapshot(store):
    """A metric offered in the dashboard that the snapshot never computes would
    read as permanently unmeasurable, which is a silent dead entry in a form."""
    values = snapshot(store, "r1")
    missing = [m.key for m in METRICS if m.key not in values]
    assert not missing, f"catalogued but never computed: {missing}"
    assert {c["key"] for c in catalogue()} == set(values)


def test_the_snapshot_agrees_with_the_run_report(store):
    """The task board and the run report read the same log, so they have to
    agree. Two panels on one page quoting different trial counts is worse than
    either being absent."""
    from aiciv.experiment.report import report as build_report

    values = snapshot(store, "r1")
    rep = build_report(store, "r1")
    assert values["trials"] == rep["exploration"]["trials"]
    assert values["distinct_recipes"] == rep["exploration"]["distinct_recipes"]
    assert values["claims_banked"] == rep["knowledge"]["banked"]
    assert values["ticks"] == rep["ticks_recorded"]


def test_scoring_never_leaks_ground_truth(store):
    """The board walks the trial table, which carries true_mu and the stratum.
    A scorer that passed a row through untouched would put ground truth in a
    dashboard response, and every experiment run after that is invalidated."""
    from aiciv.information import assert_no_leak

    assert_no_leak(score_board([task()], store, "r1"), where="test/tasks")
    assert_no_leak(snapshot(store, "r1"), where="test/snapshot")


def test_a_bad_metric_is_refused():
    with pytest.raises(ValueError):
        task(metric="whatever_i_like")
    with pytest.raises(ValueError):
        task(comparator="~=")
    with pytest.raises(ValueError):
        task(target=-1)


# --- the board: where tasks live -----------------------------------------

def test_the_board_is_not_the_run_file(tmp_path, store):
    """Tasks must never be written into the run's SQLite. That file is hashed
    and replayed, and an observer changing their mind at hour twelve must not
    be able to touch it."""
    board = TaskBoard(tmp_path / "board.json")
    board.add_run_task("r1", task())
    assert (tmp_path / "board.json").exists()
    # The run store is still openable read-only and its tables are untouched.
    assert "tasks" not in json.dumps([dict(r) for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")])


def test_a_task_added_to_a_run_is_always_an_observer_task(tmp_path):
    """Source is decided by the route, never by the caller.

    A run already under way has no channel to its agents, so a task added now
    could not have been in their prompt. Trusting a client-supplied "briefed"
    here would mislabel an unsteered run as steered -- or, worse, let a steered
    one be relabelled.
    """
    board = TaskBoard(tmp_path / "board.json")
    stored = board.add_run_task("r1", task(source="briefed"))
    assert stored.source == "observer"
    assert board.is_steered("r1") is False


def test_a_used_brief_cannot_be_edited(tmp_path):
    """Once a brief has gone into a prompt it is a record of what was said.

    Editing it afterwards would leave the run's manifest describing a brief
    that no longer exists in that form, and the run would look as if it had
    been told something it never was.
    """
    board = TaskBoard(tmp_path / "board.json")
    brief = board.create_brief("grow more")
    board.add_brief_task(brief["brief_id"], task(directive="grow more food"))
    board.attach_brief("run_a", brief["brief_id"])

    with pytest.raises(BoardError):
        board.add_brief_task(brief["brief_id"], task(task_id="t2"))
    with pytest.raises(BoardError):
        board.remove_brief_task(brief["brief_id"], "t1")


def test_attaching_a_brief_marks_the_run_steered(tmp_path):
    board = TaskBoard(tmp_path / "board.json")
    brief = board.create_brief("b")
    board.add_brief_task(brief["brief_id"], task(directive="plant clover"))
    board.attach_brief("run_a", brief["brief_id"])

    assert board.is_steered("run_a") is True
    assert board.run_tasks("run_a")[0].source == "briefed"
    # And re-attaching is idempotent: a resumed run must not double its tasks.
    board.attach_brief("run_a", brief["brief_id"])
    assert len(board.run_tasks("run_a")) == 1


def test_a_brief_with_nothing_to_say_does_not_steer(tmp_path, store):
    """A briefed task with no directive text is scored but never spoken.

    The observer wanted the target watched without saying it out loud, so the
    agents were told nothing and the run is not steered. Marking it STEERED
    would exclude a perfectly ordinary run from every unsteered comparison for
    no reason -- the mirror image of the mistake the flag exists to prevent.
    """
    board = TaskBoard(tmp_path / "board.json")
    brief = board.create_brief("silent")
    board.add_brief_task(brief["brief_id"], task(directive=""))
    board.attach_brief("r1", brief["brief_id"])

    assert board.directives(brief["brief_id"]) == ()
    assert board.is_steered("r1") is False
    scored = score_board(board.run_tasks("r1"), store, "r1")
    assert scored["steered"] is False
    assert scored["briefed_count"] == 1     # it did come from a brief
    assert scored["spoken_count"] == 0      # but nothing was said


def test_a_spoken_brief_does_steer(tmp_path, store):
    board = TaskBoard(tmp_path / "board.json")
    brief = board.create_brief("loud")
    board.add_brief_task(brief["brief_id"], task(directive="grow more food"))
    board.attach_brief("r1", brief["brief_id"])

    assert board.is_steered("r1") is True
    scored = score_board(board.run_tasks("r1"), store, "r1")
    assert scored["steered"] is True
    assert scored["spoken_count"] == 1


def test_directives_are_sorted_and_deduplicated(tmp_path):
    """The prompt must be byte-identical for identical state, or the LLM call
    log keys on noise and replay stops working. Two observers adding the same
    sentence twice must not change the prompt either."""
    board = TaskBoard(tmp_path / "board.json")
    brief = board.create_brief("b")
    for i, text in enumerate(["water every tile", "plant clover",
                              "water every tile"]):
        board.add_brief_task(brief["brief_id"],
                             task(task_id=f"t{i}", directive=text))
    assert board.directives(brief["brief_id"]) == ("plant clover",
                                                   "water every tile")


def test_a_corrupt_board_is_refused_not_silently_emptied(tmp_path):
    """Starting from an empty board would look exactly like a board nobody had
    filled in, and the observer would quietly lose every task they set."""
    path = tmp_path / "board.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BoardError):
        TaskBoard(path)


def test_the_board_survives_a_reload(tmp_path):
    path = tmp_path / "board.json"
    TaskBoard(path).add_run_task("r1", task(title="two hundred trials"))
    assert TaskBoard(path).run_tasks("r1")[0].title == "two hundred trials"


# --- the steering boundary in the prompt ---------------------------------

def test_an_unbriefed_prompt_is_unchanged():
    """The load-bearing test for every run made before briefs existed.

    If the assignment block appeared even empty, every prior run's prompt hash
    would change and nothing recorded before this feature could be compared
    with anything after it.
    """
    kwargs = dict(name="Ada", width=20, height=20, param_space=SPACE,
                  schedule_axis="plant_day", others="#1", scaffold="rules_only")
    assert system_prompt(**kwargs) == system_prompt(**kwargs, directives=())
    assert "You have been asked to" not in system_prompt(**kwargs)
    assert assignment_block(()) == ""
    assert assignment_block(("   ",)) == ""


def test_a_briefed_prompt_says_who_asked_and_that_it_is_optional():
    """The brief states the request and stops. It must not add a reward, an
    urgency, or a method -- those would be extra grants riding in on the back
    of the operator's sentence."""
    p = system_prompt(name="Ada", width=20, height=20, param_space=SPACE,
                      schedule_axis="plant_day", others="#1",
                      scaffold="rules_only",
                      directives=("grow at least 40 kg of food",))
    assert "grow at least 40 kg of food" in p
    assert "your own choice" in p


def test_the_narrative_scan_is_unaffected_by_a_directive():
    """Operator text sits outside `narrative` on purpose. Sweeping it into the
    banned-term scan would either fail an observer's run or, worse, tempt
    someone to loosen the scan that protects every scaffold comparison."""
    from aiciv.agents.policies.llm.prompts import narrative

    text = narrative(name="Ada", width=20, height=20, param_space=SPACE,
                     schedule_axis="plant_day", others="#1",
                     scaffold="rules_only").lower()
    assert not [t for t in BANNED_TERMS if t in text]


def test_method_vocabulary_in_a_directive_is_reported():
    """Not blocked -- what an observer tells the agents is theirs to decide --
    but a brief that says "run a controlled experiment" has handed over the
    exact thing rules_only withholds, and that has to be visible before the
    run costs twenty hours."""
    assert scan_banned("run a controlled experiment and compare") == [
        "compare", "experiment"]
    assert scan_banned("grow more food") == []


def test_an_unsteered_config_hashes_as_it_always_did():
    """A pinned hash, and it earned the pin.

    Adding brief_id and directives to RunConfig changed the hash of every
    unsteered config, so every run recorded before briefs existed failed its
    own replay with "config hash mismatch" -- a divergence that never happened,
    reported by the one check whose job is to tell you when a real one did.
    Empty steering is therefore omitted from the hash rather than folded in.

    The value below was computed from the code as it stood before this feature.
    If a later field breaks it again, the fix is the same: add that field to
    RunConfig._OMITTED_WHEN_EMPTY, or accept that every prior run is now
    unverifiable and say so out loud.
    """
    assert RunConfig().hash == "50aa3c12d19e661a3d42158bb4536cc3"
    assert RunConfig(seed=7, ticks=50, policy="llm").hash == (
        "32d1084a8ffcfdd42b342deb6099041c")


def test_a_brief_changes_the_config_hash():
    """Two runs with the same seed and different briefs are two different
    experiments. Sharing a config hash would make the manifest claim they were
    the same one."""
    plain = RunConfig(seed=42, policy="llm")
    briefed = RunConfig(seed=42, policy="llm", brief_id="brf_1",
                        directives=("grow more food",))
    assert plain.hash != briefed.hash


def test_directives_hash_the_same_whether_list_or_tuple():
    """A config round-tripped through JSON comes back with a list. If that
    hashed differently, reloading a run's own config would say it was a
    different run."""
    a = RunConfig(seed=42, directives=("a", "b"))
    b = RunConfig(seed=42, directives=["a", "b"])
    assert a.hash == b.hash
    assert b.directives == ("a", "b")


def test_the_llm_policy_carries_the_brief_into_its_prompt():
    from aiciv.agents.policies.llm.policy import OllamaLLMPolicy

    p = OllamaLLMPolicy(directives=("plant on the wet tiles",))
    assert p.directives == ("plant on the wet tiles",)
    assert OllamaLLMPolicy().directives == ()
