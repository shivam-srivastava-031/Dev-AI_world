"""LLM call log: how a nondeterministic policy still produces a replayable run.

We cannot make a language model deterministic. Setting temperature and a seed
narrows the variance but guarantees nothing across server builds, quantisation
or batching. So we do not pretend: every call is recorded, keyed by a hash of
everything that determined it, and replay reads the log instead of the network.

The consequence is precise and worth stating plainly: a run is NOT deterministic
when first produced, but it IS exactly replayable afterwards. Anyone can re-run
the analysis and get the same answer without owning the model.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from ....hashing import blake2b_hex

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT,
  tick         INTEGER,
  agent_id     INTEGER,
  retry_index  INTEGER,
  model        TEXT,
  options_json TEXT,
  prompt_version TEXT,
  prompt_text  TEXT,
  prompt_hash  TEXT NOT NULL,
  response_text TEXT,
  duration_ms  INTEGER,
  UNIQUE(prompt_hash, retry_index)
);
CREATE INDEX IF NOT EXISTS idx_llm_hash ON llm_calls(prompt_hash);
"""


class ReplayCacheMiss(RuntimeError):
    """Replay hit a prompt that is not in the log.

    Always a real problem: it means the prompt changed, so the replay would be
    of a different experiment. Failing loudly beats silently calling the model
    and quietly producing a run nobody can reproduce.
    """


@dataclass
class LLMCache:
    path: pathlib.Path
    mode: str = "record"                 # "record" | "replay"
    prompt_version: str = "p1"
    run_id: str = "run"
    hits: int = 0
    misses: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ("record", "replay"):
            raise ValueError(f"bad cache mode: {self.mode}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- keying -------------------------------------------------------------

    def key(self, model: str, options: dict, prompt: str) -> str:
        """Everything that determined the response goes into the key.

        Including PROMPT_VERSION: replaying an old run against a changed
        template must miss rather than silently answer the wrong question.
        """
        return blake2b_hex(model, options, self.prompt_version, prompt)

    # -- the two modes ------------------------------------------------------

    def lookup(self, prompt_hash: str, retry_index: int) -> str | None:
        row = self.conn.execute(
            "SELECT response_text FROM llm_calls "
            "WHERE prompt_hash = ? AND retry_index = ?",
            (prompt_hash, retry_index)).fetchone()
        return row[0] if row else None

    def get_or_call(self, *, model: str, options: dict, prompt: str,
                    tick: int, agent_id: int, retry_index: int,
                    call) -> str:
        h = self.key(model, options, prompt)
        cached = self.lookup(h, retry_index)
        if cached is not None:
            self.hits += 1
            return cached

        if self.mode == "replay":
            self.misses += 1
            raise ReplayCacheMiss(
                f"no logged response for tick={tick} agent={agent_id} "
                f"retry={retry_index} prompt_hash={h[:12]}; the prompt or the "
                f"model configuration changed since this run was recorded"
            )

        started = time.perf_counter()
        response = call(prompt)
        elapsed = int((time.perf_counter() - started) * 1000)

        self.conn.execute(
            "INSERT OR REPLACE INTO llm_calls "
            "(run_id, tick, agent_id, retry_index, model, options_json, "
            " prompt_version, prompt_text, prompt_hash, response_text, duration_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, tick, agent_id, retry_index, model,
             json.dumps(options, sort_keys=True), self.prompt_version,
             prompt, h, response, elapsed))
        self.conn.commit()
        self.misses += 1
        return response

    def report(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
        avg = self.conn.execute(
            "SELECT AVG(duration_ms) FROM llm_calls").fetchone()[0]
        return {
            "mode": self.mode,
            "logged_calls": total,
            "cache_hits": self.hits,
            "network_calls": self.misses if self.mode == "record" else 0,
            "mean_latency_ms": round(avg, 1) if avg else None,
            "prompt_version": self.prompt_version,
            "path": str(self.path),
        }

    def close(self) -> None:
        self.conn.close()
