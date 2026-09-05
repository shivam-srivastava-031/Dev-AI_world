"""The Ollama-backed policy, plus the client and parser it needs.

Three things keep this honest:

  * The model NEVER writes to the world. It returns text; the parser turns that
    into an ActionProposal; the validator decides whether it happens. A
    malformed reply is a rejection like any other, fed back as a hint.
  * The model never writes a claim object either. It emits a structured
    hypothesis and the backend translates it -- so the shape of public
    knowledge stays under our control even when its content does not.
  * Every call goes through the cache, so the run is replayable afterwards even
    though it was not deterministic when produced.
"""

from __future__ import annotations

import json
import pathlib
import re
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ....rng import derive_int
from ....world.actions import ActionProposal, Verb
from ...policy import PolicyContext, register
from .cache import LLMCache
from .prompts import (
    PROMPT_VERSION, observation_block, schema_json, summary_block,
    system_prompt,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "assistant:latest"
MAX_RETRIES = 2

#: Reasoning models emit these; they are not part of the answer.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    timeout: float = 120.0

    def chat(self, system: str, user: str, options: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": options,
            # Structured outputs. The difference between a working 7B agent and
            # a log full of parse failures.
            "format": json.loads(schema_json()),
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {e}") from e
        return body.get("message", {}).get("content", "")


class ParseError(ValueError):
    pass


def parse_action(text: str, param_space: dict[str, tuple]) -> ActionProposal:
    """Strict extraction. Anything ambiguous is a ParseError, not a guess.

    Guessing what a model meant would quietly substitute our judgement for its
    behaviour, which is the thing being measured.
    """
    if not text or not text.strip():
        raise ParseError("empty response")
    cleaned = THINK_RE.sub("", text).strip()
    m = JSON_RE.search(cleaned)
    if not m:
        raise ParseError("no JSON object in response")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseError(f"malformed JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("response was not an object")

    raw = str(obj.get("action", "")).strip().upper()
    try:
        verb = Verb(raw)
    except ValueError:
        raise ParseError(f"unknown action {raw!r}") from None

    if obj.get("params") is not None and not isinstance(obj["params"], dict):
        # `or {}` would have swallowed a list here and invented an empty
        # parameter set the model never sent.
        raise ParseError(
            f"params was {type(obj['params']).__name__}, not an object")
    params = dict(obj.get("params") or {})

    # Coerce the obvious shapes; anything else the validator will reject with a
    # hint the model can learn from.
    if "tile" in params and isinstance(params["tile"], (list, tuple)):
        params["tile"] = [int(params["tile"][0]), int(params["tile"][1])]
    for name, values in param_space.items():
        if name in params and values and isinstance(values[0], int):
            try:
                params[name] = int(params[name])
            except (TypeError, ValueError):
                pass

    return ActionProposal(
        verb=verb, params=params,
        rationale=str(obj.get("rationale", ""))[:400],
        message=str(obj.get("message", ""))[:500],
    )


@register
class OllamaLLMPolicy:
    """One model call per tick per agent, logged and replayable."""

    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 scaffold: str = "rules_only", cache: LLMCache | None = None,
                 temperature: float = 0.7) -> None:
        self.client = OllamaClient(model=model, host=host)
        self.scaffold = scaffold
        self.cache = cache
        self.temperature = temperature
        self.parse_failures = 0
        self.calls = 0
        self.results: dict[tuple, list[float]] = defaultdict(list)
        self.pending: dict[int, tuple] = {}
        self.seen: set = set()
        self._system: str | None = None

    def reset(self, ctx: PolicyContext) -> None:
        pass

    def observe_result(self, result: dict, ctx: PolicyContext) -> None:
        pass

    # -- the engine-computed summary ---------------------------------------

    def _ingest(self, obs, ctx) -> None:
        """Keep a running table of the agent's own results by choice.

        The engine, not the model, computes this. Asking a 7B model to hold
        sixty noisy harvests in its head produces a random walk; a small table
        of means is something it can actually use.
        """
        axes = [n for n in sorted(ctx.param_space) if n != ctx.schedule_axis]
        for h in obs.recent_harvests:
            key = (h.get("tile_id"), h.get("trial_id"))
            if key in self.seen:
                continue
            cell = self.pending.pop(h.get("tile_id"), None)
            if cell is None:
                continue
            self.seen.add(key)
            self.results[cell].append(float(h.get("yield_kg", 0.0)))

    def _summary(self) -> dict[tuple, dict]:
        return {
            k: {"n": len(v), "mean": sum(v) / len(v), "best": max(v)}
            for k, v in self.results.items() if v
        }

    # -- the decision -------------------------------------------------------

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal:
        self._ingest(obs, ctx)

        if self._system is None:
            others = ", ".join(f"#{v['agent_id']}" for v in obs.visible_agents) \
                or "nobody nearby"
            self._system = system_prompt(
                name=obs.name, width=20, height=20,
                param_space=ctx.param_space, schedule_axis=ctx.schedule_axis,
                others=others, scaffold=self.scaffold)

        blocks = [observation_block(obs)]
        summary = summary_block(self._summary())
        if summary:
            blocks.append(summary)
        user = "\n\n".join(blocks)

        for retry in range(MAX_RETRIES + 1):
            options = {
                "temperature": self.temperature,
                # Derived, not random: the same tick and agent always ask for
                # the same sampling seed, which narrows (never eliminates)
                # variance. The log is what actually guarantees replay.
                "seed": derive_int(ctx.config.seed, "llm", ctx.tick,
                                   ctx.agent_id, retry, lo=0, hi=2**31 - 1),
                "num_predict": 400,
            }
            prompt = f"{self._system}\n---\n{user}\n(attempt {retry})"
            try:
                self.calls += 1
                text = self._invoke(prompt, options, ctx, retry)
                proposal = parse_action(text, ctx.param_space)
            except ParseError:
                self.parse_failures += 1
                continue
            self._remember_plant(proposal, ctx)
            return proposal

        # Deterministic fallback. A model that cannot produce valid JSON after
        # its retries loses the day, exactly as a malformed action would.
        return ActionProposal(Verb.NOOP, {}, rationale="unparseable reply")

    def _invoke(self, prompt: str, options: dict, ctx, retry: int) -> str:
        if self.cache is None:
            return self.client.chat(self._system, prompt, options)
        return self.cache.get_or_call(
            model=self.client.model, options=options, prompt=prompt,
            tick=ctx.tick, agent_id=ctx.agent_id, retry_index=retry,
            call=lambda p: self.client.chat(self._system, p, options))

    def _remember_plant(self, proposal: ActionProposal, ctx) -> None:
        if proposal.verb is not Verb.PLANT:
            return
        tile = proposal.params.get("tile")
        if not isinstance(tile, (list, tuple)) or len(tile) != 2:
            return
        axes = [n for n in sorted(ctx.param_space) if n != ctx.schedule_axis]
        cell = tuple(proposal.params.get(a) for a in axes)
        self.pending[tile[1] * 20 + tile[0]] = cell

    def report(self) -> dict[str, Any]:
        return {"calls": self.calls, "parse_failures": self.parse_failures,
                "parse_failure_rate": (round(self.parse_failures / self.calls, 4)
                                       if self.calls else None)}
