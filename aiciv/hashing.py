"""Canonical serialisation and hashing.

Every hash in the system flows through this module. Two rules make hashes stable
across machines, numpy builds and BLAS versions:

1. Floats are quantised to FLOAT_DP decimal places at the boundary (``q``) and
   formatted with a fixed-width repr (``fmt_float``) before hashing. Trailing-bit
   drift in a dot product therefore cannot change a state hash.
2. Mappings are serialised with sorted keys and no insertion-order dependence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any

FLOAT_DP = 3
_QUANT = 10**FLOAT_DP


def q(x: float) -> float:
    """Quantise a float to FLOAT_DP decimal places (half-away-from-zero).

    Python's round() is banker's rounding, which is fine but surprising; we want
    the same answer a human gets, and more importantly the same answer on every
    platform. Values are also normalised so -0.0 and 0.0 hash identically.
    """
    if not math.isfinite(x):
        raise ValueError(f"non-finite value cannot be quantised: {x!r}")
    scaled = x * _QUANT
    # copysign trick: floor(|v| + 0.5) with the original sign restored.
    r = math.floor(abs(scaled) + 0.5)
    out = math.copysign(r, scaled) / _QUANT
    return out + 0.0  # collapse -0.0 -> 0.0


def fmt_float(x: float) -> str:
    """Fixed-width decimal repr used inside hashes. Never scientific notation."""
    return f"{q(x):.{FLOAT_DP}f}"


def _canon(obj: Any) -> Any:
    """Recursively convert to a JSON-safe, order-stable, float-stable structure."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return fmt_float(obj)
    if isinstance(obj, dict):
        # Keys must be strings so that sorting is total and unambiguous.
        return {str(k): _canon(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        # Sets have no inherent order; sort their canonical forms.
        return sorted((_canon(v) for v in obj), key=json.dumps)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # enum.Enum
        return _canon(obj.value)
    raise TypeError(f"cannot canonicalise {type(obj).__name__}: {obj!r}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for HASHING: sorted keys, quantised floats.

    LOSSY ON PURPOSE. Floats become fixed-width strings, which is exactly
    what makes a hash stable across BLAS versions -- and exactly what makes
    this unsuitable for anything that will be read back. Use
    ``stable_json`` to persist a value you intend to load again.
    """
    return json.dumps(_canon(obj), separators=(",", ":"), ensure_ascii=True)


def stable_json(obj: Any) -> str:
    """Deterministic JSON that ROUND-TRIPS: sorted keys, types preserved.

    Serialising an action payload with canonical_json turned min_delta=0.3
    into the string "0.300", so replaying the run rejected the claim as
    having a non-positive delta and the world quietly diverged. Ordering is
    still deterministic; only the lossy float handling is gone.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def blake2b_hex(*parts: Any, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    for p in parts:
        h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator: prevents ("ab","c") == ("a","bc")
    return h.hexdigest()


def blake2b_int(*parts: Any, nbytes: int = 8) -> int:
    h = hashlib.blake2b(digest_size=nbytes)
    for p in parts:
        h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def sign_row(secret: bytes, row: dict[str, Any]) -> str:
    """HMAC over a trial row, excluding any existing signature field.

    The secret lives only in the engine process and the ``runs`` table. It is
    never placed in an Observation, a memory record or a prompt, so an agent
    cannot forge a row even with full write access to the database file.
    """
    payload = canonical_json({k: v for k, v in row.items() if k != "signature"})
    return hmac.new(secret, payload.encode("utf-8"), hashlib.blake2b).hexdigest()


def verify_row(secret: bytes, row: dict[str, Any]) -> bool:
    got = row.get("signature")
    if not isinstance(got, str):
        return False
    return hmac.compare_digest(got, sign_row(secret, row))
