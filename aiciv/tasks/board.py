"""Where tasks are kept.

Deliberately NOT in the run's SQLite file. A run file is the record of what the
world did, it is hashed and replayed, and an observer changing their mind about
a target at hour twelve must not touch it. Tasks live in their own JSON file,
so the run store keeps its property that nothing writes to it but the engine.

JSON rather than another SQLite: this is a handful of rows edited by one person
through a form, and being able to read and hand-edit the file is worth more
here than transactions. Writes go through a temp file and os.replace, so a
crash mid-write leaves the previous board rather than half of one.
"""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any

from .spec import Task

#: Bumped only when the on-disk shape changes incompatibly.
VERSION = 1

DEFAULT_BOARD = pathlib.Path("tasks") / "board.json"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BoardError(ValueError):
    """A bad request against the board -- unknown id, duplicate name."""


class TaskBoard:
    """The observer's task board, loaded and saved whole.

    Small enough that read-modify-write of the entire file is the simplest
    correct thing. Every mutating method saves before returning, so there is no
    window where the caller holds unsaved state.
    """

    def __init__(self, path: pathlib.Path | str = DEFAULT_BOARD) -> None:
        self.path = pathlib.Path(path)
        self.data = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": VERSION, "briefs": {}, "run_tasks": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            # Refuse to silently start from empty. A board that quietly resets
            # would look identical to a board nobody had filled in yet.
            raise BoardError(
                f"{self.path} is not readable JSON ({e}). Fix or move it; "
                f"starting from an empty board would hide the tasks you set."
            ) from e
        data.setdefault("briefs", {})
        data.setdefault("run_tasks", {})
        data.setdefault("version", VERSION)
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # -- tasks attached to a run -------------------------------------------

    def run_tasks(self, run_id: str) -> list[Task]:
        return [Task.from_dict(d)
                for d in self.data["run_tasks"].get(run_id, [])]

    def add_run_task(self, run_id: str, task: Task) -> Task:
        """Attach a task to an existing run.

        A task added here is always ``observer``: the run either already
        happened or is already going, so nothing added now could have been in
        the agents' prompt. The source is overridden rather than trusted from
        the caller, because a mislabelled steered run is worse than a rejected
        request.
        """
        task = Task.from_dict({**task.to_dict(), "source": "observer",
                               "created_at": task.created_at or now()})
        self.data["run_tasks"].setdefault(run_id, []).append(task.to_dict())
        self.save()
        return task

    def remove_run_task(self, run_id: str, task_id: str) -> None:
        rows = self.data["run_tasks"].get(run_id, [])
        kept = [r for r in rows if r["task_id"] != task_id]
        if len(kept) == len(rows):
            raise BoardError(f"no task {task_id!r} on run {run_id!r}")
        self.data["run_tasks"][run_id] = kept
        self.save()

    # -- briefs: tasks the agents will actually be given ---------------------

    def briefs(self) -> list[dict[str, Any]]:
        return sorted(self.data["briefs"].values(),
                      key=lambda b: b["created_at"], reverse=True)

    def brief(self, brief_id: str) -> dict[str, Any]:
        b = self.data["briefs"].get(brief_id)
        if b is None:
            raise BoardError(f"no brief {brief_id!r}")
        return b

    def create_brief(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise BoardError("a brief needs a name")
        brief = {"brief_id": new_id("brf"), "name": name,
                 "created_at": now(), "tasks": [], "launched_runs": []}
        self.data["briefs"][brief["brief_id"]] = brief
        self.save()
        return brief

    def delete_brief(self, brief_id: str) -> None:
        self.brief(brief_id)
        del self.data["briefs"][brief_id]
        self.save()

    def add_brief_task(self, brief_id: str, task: Task) -> Task:
        brief = self.brief(brief_id)
        if brief["launched_runs"]:
            # The brief has already been read into a prompt. Editing it now
            # would mean the recorded brief no longer matches what those agents
            # were told, and the run's provenance would be quietly wrong.
            raise BoardError(
                f"brief {brief_id!r} has already been used by "
                f"{', '.join(brief['launched_runs'])}; copy it into a new "
                f"brief rather than editing what those agents were told")
        task = Task.from_dict({**task.to_dict(), "source": "briefed",
                               "brief_id": brief_id,
                               "created_at": task.created_at or now()})
        brief["tasks"].append(task.to_dict())
        self.save()
        return task

    def remove_brief_task(self, brief_id: str, task_id: str) -> None:
        brief = self.brief(brief_id)
        if brief["launched_runs"]:
            raise BoardError(
                f"brief {brief_id!r} has already been used by "
                f"{', '.join(brief['launched_runs'])} and is now a record of "
                f"what was said; it cannot be edited")
        kept = [t for t in brief["tasks"] if t["task_id"] != task_id]
        if len(kept) == len(brief["tasks"]):
            raise BoardError(f"no task {task_id!r} in brief {brief_id!r}")
        brief["tasks"] = kept
        self.save()

    def brief_tasks(self, brief_id: str) -> list[Task]:
        return [Task.from_dict(d) for d in self.brief(brief_id)["tasks"]]

    def directives(self, brief_id: str) -> tuple[str, ...]:
        """The sentences that go into the agents' prompt, in a fixed order.

        Sorted, because the prompt must be byte-identical for identical state
        or the LLM call log keys on noise and replay stops working.
        """
        texts = [t.directive.strip() for t in self.brief_tasks(brief_id)
                 if t.directive.strip()]
        return tuple(sorted(set(texts)))

    def attach_brief(self, run_id: str, brief_id: str) -> list[Task]:
        """Record that a run was launched under a brief.

        Copies the brief's tasks onto the run so the dashboard scores them, and
        stamps the run into the brief so the brief becomes read-only from here.
        The copy is what freezes it: later edits to the brief -- were they
        allowed -- could not retroactively change what this run was told.
        """
        brief = self.brief(brief_id)
        tasks = self.brief_tasks(brief_id)
        existing = {t["task_id"] for t in self.data["run_tasks"].get(run_id, [])}
        rows = self.data["run_tasks"].setdefault(run_id, [])
        for t in tasks:
            if t.task_id not in existing:
                rows.append(t.to_dict())
        if run_id not in brief["launched_runs"]:
            brief["launched_runs"].append(run_id)
        self.save()
        return tasks

    def is_steered(self, run_id: str) -> bool:
        """Whether anything was actually said to this run's agents.

        A briefed task with no directive text is scored but never spoken, so it
        does not make the run steered -- see ``score_board``, which draws the
        same line.
        """
        return any(t["source"] == "briefed" and (t.get("directive") or "").strip()
                   for t in self.data["run_tasks"].get(run_id, []))
