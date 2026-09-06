"""Importing this package registers every policy.

Registration used to depend on each caller importing the right module, which
meant a test could fail with "unknown policy 'random'" for reasons that had
nothing to do with what it was testing. Importing here makes the registry
complete as soon as anything touches the package.

``llm`` is included: it constructs with defaults and contacts nothing until it
is asked to decide, so registering it costs nothing on a scripted run.
"""

from . import archetypes, builder, random_policy, scripted  # noqa: F401
from .llm import policy as llm_policy  # noqa: F401

__all__ = ["archetypes", "builder", "random_policy", "scripted", "llm_policy"]
