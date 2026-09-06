"""Prompt construction, and the scaffold levels that make priors measurable.

The scaffold is the experiment's most sensitive surface. `rules_only` -- the
default -- describes the world and the available actions and NOTHING about
method. If we write "run controlled trials" into the system prompt, we have
handed the agents the scientific method and any later claim about them applying
it is circular.

BANNED_TERMS is scanned by a test against the rules_only prompt. The banned
list is deliberately blunt: it is easier to keep a word out than to argue about
whether a particular sentence implies it.

Prompts must also be byte-identical for identical state, or the call log keys
on noise and replay stops working. Hence: sorted iteration everywhere, fixed
float formatting, no clock, no id().
"""

from __future__ import annotations

import json
from typing import Any

# Both re-exported below: the banned list and its scanner are the information
# boundary's business, but every existing caller looks for them here.
from ....information import BANNED_TERMS as _BANNED_TERMS
from ....information import scan_banned  # noqa: F401  re-export

PROMPT_VERSION = "p1"

SCAFFOLD_LEVELS = ("bare", "rules_only", "rules_plus_method")

#: Never allowed to appear in a rules_only prompt. See
#: docs/agent_prior_knowledge.md section 6.
#:
#: Defined in ``information`` and re-exported here, where the scan has always
#: lived. It moved because the API needs it too -- to report method vocabulary
#: in an observer's brief -- and the API must not import policy machinery, or
#: the engine acquires a dependency on the web server it is meant to outlive.
BANNED_TERMS = _BANNED_TERMS

BARE = """You are {name}, living in a world of tiles.
Reply with a single JSON object choosing one action.
"""

RULES_ONLY = """You are {name}. You live in a walled world of {width}x{height} tiles \
and you grow food to stay fed.

How the world works:
- You can act on the tile you stand on or any of the eight touching it.
- Planting uses a seed and takes time to grow. When it is ready you harvest it.
- A tile you have just harvested needs to rest before it can be planted again.
- You lose a little food every day and can eat what you have stored.
- If an action is not possible you are told why, it costs you the day, and you \
may try something else tomorrow.

When you plant you choose: {param_help}
The day of the cycle is decided by when you plant, not by you.

Other people live here too: {others}. You can speak to someone standing next \
to you, and you can walk toward someone you can see.

Write in ordinary English.
"""

METHOD_EXTRA = """
How to work things out:
- Vary one thing at a time and keep the rest the same, or you will not know \
which change mattered.
- Repeat a promising result several times before believing it; a single good \
harvest can be luck.
- Compare against something. "This is good" means nothing without "compared to \
what".
- Use PROPOSE_CLAIM and REGISTER_CLAIM to record a comparison, then gather \
results afterwards to support it.
"""

CORE_ACTIONS = """
Actions you may take (one per day):
  MOVE          {{"tile": [x, y]}}
  INSPECT_TILE  {{"tile": [x, y]}}
  PLANT         {{"tile": [x, y], {plant_params}}}
  HARVEST       {{"tile": [x, y]}}
  EAT           {{}}
  REST          {{}}
  SAY           {{"to": <agent id>}}  with a message
  GOAL_SET      {{}}  with a message describing what you want
  NOOP          {{}}
"""

#: The optional record-keeping actions.
#:
#: Offering these GRANTS comparison vocabulary. An action called PROPOSE_CLAIM
#: with a "baseline" parameter tells an agent that comparing against something
#: is a thing one does -- that is a real prior, and it is recorded as GRANTED
#: in docs/agent_prior_knowledge.md rather than quietly ignored.
#:
#: What is still withheld: any instruction to use them, and any account of how
#: to gather evidence that would actually support a claim.
#:
#: The protocol_offered=False arm removes them entirely, which is the only way
#: to measure what merely naming them is worth.
PROTOCOL_ACTIONS = """  TEACH         {{"to": <agent id>, "claim_id": "<id>"}}
  PROPOSE_CLAIM {{"spec": {{...}}, "baseline": {{...}}, "direction": "increase", \
"min_delta": <number>}}
  REGISTER_CLAIM {{"claim_id": "<id>"}}
"""

REPLY_FORMAT = """
Reply with JSON only:
{{"action": "<ACTION>", "params": {{...}}, "message": "<optional>", \
"rationale": "<why, briefly>"}}
"""

#: The observer's brief, when a run was launched under one.
#:
#: This block is the ONE place operator text reaches an agent, and a run that
#: carries it is a steered run: the agents were told what to aim at, and any
#: later statement about what they chose to do has to say so. It is absent
#: entirely unless directives were passed, so an ordinary run is byte-identical
#: to what it was before this existed.
#:
#: The wording around the directives is deliberately flat. "You have been asked
#: to" states who wants it without adding urgency, a reward, or a method; a
#: block that said "your goal is to maximise" would be handing over an
#: objective function on top of the operator's sentence.
ASSIGNMENT = """
You have been asked to:
{items}
Nothing forces you to. What you actually do is your own choice.
"""


def assignment_block(directives: tuple[str, ...]) -> str:
    """Render the brief. Empty for an unsteered run, and that emptiness is
    what keeps those runs comparable with runs made before briefs existed."""
    if not directives:
        return ""
    items = "\n".join(f"- {d.strip()}" for d in directives if d.strip())
    if not items:
        return ""
    return ASSIGNMENT.format(items=items)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def param_help(param_space: dict[str, tuple], schedule_axis: str) -> str:
    parts = []
    for name in sorted(param_space):
        if name == schedule_axis:
            continue
        vals = list(param_space[name])
        parts.append(f"{name} (one of {vals})")
    return "; ".join(parts)


def plant_params_help(param_space: dict[str, tuple], schedule_axis: str) -> str:
    return ", ".join(f'"{n}": <value>' for n in sorted(param_space)
                     if n != schedule_axis)


def narrative(*, name: str, width: int, height: int,
              param_space: dict[str, tuple], schedule_axis: str,
              others: str, scaffold: str) -> str:
    """The prose half: what the world is, and nothing about how to investigate
    it. This is what the banned-term scan checks."""
    if scaffold not in SCAFFOLD_LEVELS:
        raise ValueError(f"unknown scaffold level: {scaffold}")
    if scaffold == "bare":
        return BARE.format(name=name)
    body = RULES_ONLY.format(
        name=name, width=width, height=height, others=others,
        param_help=param_help(param_space, schedule_axis))
    if scaffold == "rules_plus_method":
        body += METHOD_EXTRA
    return body


def system_prompt(*, name: str, width: int, height: int,
                  param_space: dict[str, tuple], schedule_axis: str,
                  others: str, scaffold: str,
                  protocol_offered: bool = True,
                  directives: tuple[str, ...] = ()) -> str:
    """``directives`` is the observer's brief, and defaults to none.

    It sits outside ``narrative`` on purpose: the banned-term scan is a test of
    OUR prose, and sweeping an operator's own sentences into it would either
    fail their run or, worse, tempt someone to loosen the scan. Contamination
    in a directive is reported by ``scan_banned`` instead, where it is
    attributed to the person who wrote it.
    """
    body = narrative(name=name, width=width, height=height,
                     param_space=param_space, schedule_axis=schedule_axis,
                     others=others, scaffold=scaffold)
    actions = CORE_ACTIONS.format(
        plant_params=plant_params_help(param_space, schedule_axis))
    if protocol_offered:
        actions += PROTOCOL_ACTIONS
    return body + assignment_block(directives) + actions + REPLY_FORMAT


def observation_block(obs, *, budget_chars: int = 2200) -> str:
    """Deterministic rendering of what the agent can see.

    Sorted keys and fixed float formatting throughout: if this varied by dict
    ordering the call log would key on noise and replay would break.
    """
    d = obs.to_dict()
    lines = [
        f"Day {d['tick']} (day {d['day_of_cycle']} of the cycle). "
        f"You are at ({d['x']},{d['y']}).",
        f"Food {_fmt(d['food'])}, energy {_fmt(d['energy'])}, "
        f"water {d['water_stock']}, seeds {d['seeds']}, "
        f"skill {_fmt(d['skill_farming'])}, standing {_fmt(d['reputation'])}.",
    ]

    if d.get("last_action_result"):
        r = d["last_action_result"]
        if r.get("ok"):
            lines.append(f"Yesterday you did {r.get('action')}.")
        else:
            lines.append(f"Yesterday failed: {r.get('hint', '')}")

    tiles = []
    for t in d["nearby_tiles"]:
        bits = [f"({t['tile'][0]},{t['tile'][1]}) {t['terrain']}"]
        if t["planted"]:
            bits.append("planted" + (" READY" if t["crop_ready"] else " growing"))
        if t.get("fallow"):
            bits.append("resting")
        if t.get("soil_band") is not None:
            bits.append(f"soil {t['soil_band']}")
        tiles.append(" ".join(bits))
    lines.append("Around you: " + "; ".join(tiles))

    if d["my_plots"]:
        lines.append("Your plots: " + "; ".join(
            f"({p['tile'][0]},{p['tile'][1]})" + (" READY" if p["ready"] else "")
            for p in d["my_plots"]))

    if d["recent_harvests"]:
        lines.append("Recent harvests: " + "; ".join(
            f"{_fmt(h.get('yield_kg', 0))} ({h.get('crop_health', '')})"
            for h in d["recent_harvests"][-6:]))

    if d["visible_agents"]:
        lines.append("You can see: " + "; ".join(
            f"{v['name']} (#{v['agent_id']}) at ({v['tile'][0]},{v['tile'][1]})"
            for v in d["visible_agents"]))

    if d["messages"]:
        lines.append("Said to you: " + "; ".join(
            f"#{m['from']}: {m['text']}" for m in d["messages"][-3:]))

    if d["my_claims"]:
        lines.append("Your records: " + "; ".join(
            f"{c['claim_id']} [{c['state']}] {c['describes']}"
            for c in d["my_claims"][-4:]))

    if d["beliefs"]:
        lines.append("You were told: " + "; ".join(
            f"{b['claim_id']} from #{b['source']} ({b['status']})"
            for b in d["beliefs"][-4:]))

    text = "\n".join(lines)
    return text[:budget_chars]


def summary_block(summary: dict[tuple, dict]) -> str:
    """The engine-computed aggregate of an agent's own results.

    Non-optional for a 7B model. Handing it sixty individual noisy harvests and
    expecting it to average them in its head produces a random walk; handing it
    a small table of means gives it something it can actually reason over.
    """
    if not summary:
        return ""
    rows = ["What you have grown so far (average yield by choice):"]
    for key in sorted(summary, key=str):
        s = summary[key]
        rows.append(f"  {key}: n={s['n']} mean={_fmt(s['mean'])} "
                    f"best={_fmt(s['best'])}")
    return "\n".join(rows)


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "params": {"type": "object"},
        "message": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["action"],
}


def schema_json() -> str:
    return json.dumps(ACTION_SCHEMA, sort_keys=True)
