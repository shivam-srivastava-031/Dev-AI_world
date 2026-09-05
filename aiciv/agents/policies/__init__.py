"""Importing this package registers every policy.

Registration used to depend on each caller importing the right module, which
meant a test could fail with "unknown policy 'random'" for reasons that had
nothing to do with what it was testing. Importing here makes the registry
complete as soon as anything touches the package.
"""

from . import archetypes, builder, random_policy, scripted  # noqa: F401

__all__ = ["archetypes", "builder", "random_policy", "scripted"]
