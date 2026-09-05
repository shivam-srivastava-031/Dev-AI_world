# Agent Prior Knowledge

**Status:** normative. Changing anything here changes what the experiment means,
so changes require a version bump and invalidate comparison with prior runs.

---

## 0. The honest limit, stated first

**An LLM-based agent cannot be a cognitive blank slate.**

A 7B instruction-tuned model already knows what a plant is, what counting is,
what "more" means, and what a controlled comparison is. We did not put that
there and we cannot take it out. Any sentence of the form

> "the civilization invented controlled experimentation"

is, in this project, **false by construction**. The most we can ever claim is:

> "the civilization *applied* controlled experimentation, which the model
> already latently possessed, to a domain it had no information about."

This document exists to make that boundary explicit and *measurable*, not to
pretend it away. Every result we publish must be read against it.

---

## 1. The four classes

Every prior an agent holds falls into exactly one class.

| Class | Definition | Under our control? |
|---|---|---|
| `GRANTED` | We put it in the system prompt or the initial state. | Yes, fully |
| `WITHHELD` | Deliberately absent, and actively scrubbed if it appears. | Yes, fully |
| `DERIVED` | Learned inside the world during the run. | Emergent |
| `LATENT` | Present in the model weights. | **No** |

`LATENT` is the whole problem. Section 5 is how we measure it.

---

## 2. GRANTED — exactly what we hand them

Anything not on this list is not granted. Enforced by
`test_scaffold_rules_only_contains_no_method_terms`.

### 2.1 Initial world model
- The grid is 20x20 and positions are integer `(x, y)`.
- Adjacency is Chebyshev distance 1 (eight neighbours).
- The full action verb list, with each verb's parameter names and legal ranges.
- The semantics of *rejection*: an illegal action is refused, costs the tick,
  and returns a reason code with a hint.
- That crops take time to mature and must be harvested.
- That food is consumed each tick and that a floor prevents starvation death.

### 2.2 Initial knowledge
- The names of the parameters they may vary — `spacing`, `plant_day`, `water`,
  `companion` — and the legal values of each.
- That `INSPECT_TILE` reveals a tile's soil band.

**Explicitly NOT granted:** that any of those parameters affects yield at all.
An agent is told it *may* set `companion`, never that it *should*, nor that
doing so does anything whatsoever.

### 2.3 Initial skills
- `skill_farming = 0.0` for every agent.
- No procedural knowledge. There is no starting Procedure, Technique or recipe.

### 2.4 Initial social knowledge
- That the other agents exist, their ids, and that they are addressable.
- The existence of the `SAY`, `ASK` and `TEACH` verbs.

**Explicitly NOT granted:** any norm. Nothing tells an agent that teaching is
good, that honesty is expected, that sharing is reciprocated, that other agents
are trustworthy, or that cooperation is worthwhile. Trust starts neutral and
becomes endogenous through outcomes only.

### 2.5 Initial goals
None. `GOAL_SET` exists and is empty at tick 0. No agent is given an objective,
a target yield, a research programme, or a suggestion to investigate anything.

---

## 3. WITHHELD — deliberately absent and actively scrubbed

- The yield function and every constant in it.
- That a hidden optimum exists at all.
- That `companion` and `spacing` interact.
- That `soil_band` is linear in yield, or that it matters.
- The tile stratum (discovery / confirmation / holdout).
- The noise magnitude sigma.
- Any true effect size, and the ground-truth optimum.
- **That experimentation is a useful thing to do.** No mention of hypotheses,
  controls, baselines, replication, confounds, sample size or significance
  appears in the `rules_only` scaffold.

Enforcement: `aiciv.information.assert_no_leak` runs on every Observation before
it reaches a policy, and a lexical scan asserts the banned methodological
vocabulary never appears in the default system prompt.

---

## 4. LATENT — what we cannot remove

Non-exhaustive, and that is precisely the point:

- English, at full fluency.
- Arithmetic, comparison, averaging, ordering.
- Real-world agronomy folk knowledge: plants need water, crowding hurts,
  legumes fix nitrogen, marigolds repel pests. **Some of this is true in our
  world and some is false**, which is itself a confound: a model may transfer a
  real-world prior that happens to be wrong here.
- The concept of a fair test, a control group, and repeated measurement.
- The idea that a claim can be supported or refuted by evidence.

> The sharpest case. `CLOVER` is a real nitrogen-fixing companion crop and
> `MARIGOLD` is a real pest-repellent companion. A model carrying those priors
> may reach the right answer here **for reasons having nothing to do with
> evidence gathered in our world.** In this domain CLOVER is in fact optimal at
> wide spacing and MARIGOLD is a trap — so a folk prior gets one right and one
> wrong. The prior probe is what detects this, and it is the reason the probe
> is mandatory rather than nice-to-have.

---

## 5. Making LATENT measurable: the prior probe

```
aiciv prior-probe --model <tag> --domain <d> --scaffold rules_only
```

Run **cold**: identical scaffolding, zero experience, no memory, no trials.

| # | Task | Scored by |
|---|---|---|
| 1 | Name the best parameter combination you can. | distance from ground-truth optimum |
| 2 | Predict yields for 20 held-out combinations. | MAE, and calibration of stated intervals |
| 3 | Unprompted: how would you find out? | rubric scan for controls, replication, confound awareness |

The output is a **prior baseline**, stored once per (model, domain, prompt
version) and referenced by `prior_probe_id` in every run manifest.

### 5.1 How the baseline is used

**In-run discoveries are credited only against the baseline.**

- If the cold model already names `CLOVER` at wide spacing, the target discovery
  is *recall*, not discovery, and the report says so in those words.
- If the cold model already describes one-factor-at-a-time control, then
  controlled experimentation is `LATENT`. `protocol_adoption_rate` then measures
  **application**, never emergence, and must be labelled that way.
- Task 2's MAE is the floor for `CAN_PREDICT`. An agent demonstrates predictive
  understanding only by beating its own cold baseline.

### 5.2 What the probe cannot do

It measures what a model will *say* when asked directly. A model may fail to
articulate a method it would nonetheless follow, or articulate one it does not
follow. The probe therefore gives a **lower bound on latent knowledge** and
thereby an **upper bound on the credit we may assign to emergence**. We report
it as a bound, never as a point estimate.

---

## 6. Scaffold levels

| Level | Contains | Purpose |
|---|---|---|
| `bare` | verbs and legal ranges only | floor |
| `rules_only` | **default** — the world model of section 2, no method vocabulary | the experiment |
| `rules_plus_method` | adds an explicit description of controlled experimentation | contrast arm |

`rules_plus_method` exists solely to quantify what *handing* them the method is
worth. The gap between it and `rules_only` is one of the more informative
numbers this project can produce.

Banned in `rules_only` (lexical scan): *hypothesis, experiment, control group,
baseline, replicate, confound, variable, significance, sample size, p-value,
scientific, systematic, methodical, isolate, hold constant*.

---

## 7. Reporting requirement

Every result involving discovery or method **must** state the prior baseline
alongside it. A discovery count without its baseline is not a result.
