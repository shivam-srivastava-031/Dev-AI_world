# Dev-AI_world — AI Civilization

A persistent artificial world where AI inhabitants learn, discover, verify,
teach, contest and build on knowledge — and where we can **measure** whether
that happened rather than read a convincing narrative about it.

Everything here follows from one constraint:

> An LLM can always claim "I invented a fusion reactor."
> So the **world**, never the model, decides what is true.

---

## What exists

| Layer | Where | What it does |
|---|---|---|
| Determinism substrate | `aiciv/rng.py`, `hashing.py` | key-derived randomness; consumption order cannot affect results |
| World engine | `aiciv/world/` | grid, fallow, actions, validation, execution, upkeep, tick loop |
| Hidden mechanics | `aiciv/world/domains/` | synthetic (deceptive, known optimum) and agronomy (real published science) |
| Verification | `aiciv/knowledge/` | claims, staged lifecycle, causal hygiene, replication kinds, shadow verifier |
| Agents | `aiciv/agents/` | observation boundary, policies: random, scripted, archetypes, builder, LLM |
| Society | `aiciv/civilization/` | belief vs knowledge, endogenous trust, specialization detection |
| Capability | `aiciv/capabilities/` | knowledge → procedure → capability → artifact → changed action space |
| Language | `aiciv/language/` | graded English bound; coinage preserved, substitute grammars rejected |
| Measurement | `aiciv/metrics/` | ground-truth grading, competence, adoption, resilience |
| Persistence | `aiciv/persistence/` | one SQLite file per run, exact replay |
| API / dashboard | `api/`, `frontend/` | read-only observation, inheriting the same information boundary |

**239 tests.** `pytest tests -q` for the CI tier, `pytest tests/nightly -q` for
the 20 000-replicate statistical robustness tier.

---

## Quick start

```bash
pip install -e ".[dev]"

# a run: 5 agents, 900 ticks, saved to runs/demo.sqlite
python -m aiciv.cli run --seed 42 --ticks 900 --policy builder --run-id demo

# prove the engine is deterministic by re-executing the recorded decisions
python -m aiciv.cli replay --run demo

# what the civilization actually did
python -m aiciv.cli metrics --run demo

# arms x paired seeds, with bootstrap contrasts
python -m aiciv.cli experiment --name transfer --seeds 42,43,44

# what did the model already know, BEFORE it farmed anything?
python -m aiciv.cli prior-probe --model assistant:latest

# observe
uvicorn api.main:app --port 8000
cd frontend && npm install && npm run dev      # http://localhost:3000
```

The engine must pass everything above with the API and frontend stopped. If it
ever cannot, the science has acquired a dependency on a web server.

---

## The five ideas that matter

**1. The world is the arbiter, structurally.**
Agents never submit evidence. They act; the world records the outcome and signs
the row with a secret that never enters an observation, a memory or a prompt.
A claim's support is a *query over world-authored trials*. Only trials
harvested **after** registration count, so an agent cannot submit its luckiest
history. An author cannot replicate their own claim.

**2. Verification is staged, and the stages are disjoint.**
`SUPPORTED` on discovery tiles → `CONFIRMED` on confirmation tiles with an
independent replicator → `GENERALIZED` on holdout tiles used in neither. The
partition is hash-interleaved and invisible; agents are deliberately *not*
blocked from holdout tiles, because blocking would leak the split through
rejection codes.

Measured full-pipeline false-confirmation rate: **0.00050, 95% CI
[0.00024, 0.00092]** — exactly the predicted α × confirmation_α. Requiring
independent replication cuts false confirmations **16×**.

**3. Nothing is chosen by hand that could be measured.**
`min_trials_per_group` is the smallest n whose CI lower bound clears 0.80
power, per domain. Sigma is set by what keeps a single lucky trial
uninformative. Every calibrated value and its date is in
[`docs/statistics.md`](docs/statistics.md).

**4. Agents are graded on their own information state.**
An "opportunity" to form a hypothesis means *this agent held enough of its own
evidence in two conditions*. An agent that never had a formulable comparison is
**undefined**, not zero. Hidden fields provably cannot influence the answer —
the test removes them and asserts the output is byte-identical.

**5. Knowledge has to be able to change the world.**
A confirmed claim compiles into a procedure, which with practice unlocks a
capability, which builds an artifact that persists — and raises the water
available nearby, making recipes attemptable that were previously impossible.
Including for agents who learned none of it. A technology that changes nothing
about what is possible is a badge.

---

## What this project will not claim

Read [`docs/agent_prior_knowledge.md`](docs/agent_prior_knowledge.md) before
interpreting any result.

An LLM agent **cannot be a cognitive blank slate**. A 7B model already knows
what a plant is and what a fair test is. So "the civilization invented
controlled experimentation" is false by construction here. The most that can
ever be said is that it *applied* a latent capability to a domain it had no
information about.

The prior probe measures that: the model is run cold, and everything it can
already name is recorded as the baseline that in-run discoveries are credited
against. If the cold model names the optimum, the discovery is **recall**, and
the report says so.

Two further honest limits, both recorded rather than buried:

* Offering `PROPOSE_CLAIM` with a `baseline` parameter **grants comparison
  vocabulary**. The `protocol_offered=False` arm exists to measure what merely
  naming it is worth.
* The shadow verifier scans marginal single-variable hypotheses only, so its
  count is not like-for-like with agent claims. Use
  `aiciv/metrics/adoption.py` for per-agent grading.

---

## Documentation

| Document | What it settles |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | the experimental contract: what selects behavior, the information boundary, claim types, replication kinds, the language bound |
| [`docs/agent_prior_knowledge.md`](docs/agent_prior_knowledge.md) | exactly what agents are granted, what is withheld, and what cannot be taken away |
| [`docs/statistics.md`](docs/statistics.md) | every calibrated number, how it was measured, and when |

---

## Reporting standards

- Every verdict carries an effect, a 95% interval, an adjusted p, an effect
  size and an n. A p-value alone says whether a difference exists without
  saying whether it matters.
- False discovery rate, graded against ground truth, is a **headline** number.
- A claim confirmed only by taught replications is reported as **diffusion**,
  never as independent corroboration.
- If an experiment fails, that is the finding. If something is scripted it is
  called scripted; if it is emergent it is called emergent, with the
  prior-baseline caveat attached.
