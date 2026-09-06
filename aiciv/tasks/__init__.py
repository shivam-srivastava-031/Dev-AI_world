"""Observer-set tasks, and how far along a run is on each.

Two paths, kept apart on purpose:

  * an ``observer`` task is scored from the log and the agents never see it, so
    setting one cannot change what the run does;
  * a ``briefed`` task was written into the agents' prompt when the run was
    launched, which makes that run steered and not comparable with an
    unsteered one.

``spec`` holds the metric catalogue and the scoring; ``board`` holds where the
tasks are kept, which is never the run's own file.
"""

from .board import BoardError, TaskBoard
from .spec import (
    COMPARATORS, METRICS, SOURCES, Metric, Task, catalogue, score, score_board,
    snapshot,
)

__all__ = [
    "BoardError", "COMPARATORS", "METRICS", "Metric", "SOURCES", "Task",
    "TaskBoard", "catalogue", "score", "score_board", "snapshot",
]
