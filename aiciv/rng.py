"""Key-derived randomness. The single design decision the whole engine rests on.

Every random draw is addressed by a *key* -- (seed, purpose, tick, agent, ...) --
rather than drawn from a sequential stream. Consequences:

* The ORDER in which random numbers are consumed cannot affect any result.
  Adding a feature that draws an extra number does not shift anyone else's draws.
* Checkpoints carry no RNG state, so a checkpoint can be loaded into a fresh
  process with zero hidden state and the run continues identically.
* Phase 3 (DECIDE) can be parallelised across agents without touching results.

Nothing outside this module may call ``random.*`` or ``np.random.*``; an AST scan
in tests/test_determinism.py enforces that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hashing import blake2b_int


def derive_seed(master_seed: int, *keys: Any) -> int:
    """Derive a 64-bit seed from a master seed and a tuple of naming keys."""
    return blake2b_int(master_seed, list(keys), nbytes=8)


def derive_rng(master_seed: int, *keys: Any) -> np.random.Generator:
    """A Generator uniquely determined by (master_seed, *keys).

    Always name the purpose first, e.g.::

        derive_rng(seed, "yield", tick, agent_id, tile_id)
    """
    if not keys:
        raise ValueError("derive_rng requires at least one naming key")
    return np.random.default_rng(derive_seed(master_seed, *keys))


def derive_int(master_seed: int, *keys: Any, lo: int = 0, hi: int = 2**31 - 1) -> int:
    """A deterministic integer in [lo, hi) without constructing a Generator.

    Used for things like per-call LLM sampling seeds where a full Generator is
    wasteful.
    """
    if hi <= lo:
        raise ValueError(f"empty range [{lo}, {hi})")
    return lo + derive_seed(master_seed, *keys) % (hi - lo)


def derive_unit(master_seed: int, *keys: Any) -> float:
    """A deterministic float in [0, 1). Handy for hash-based partitioning."""
    return derive_seed(master_seed, *keys) / 2**64
