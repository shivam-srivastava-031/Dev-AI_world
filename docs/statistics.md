# Statistics: calibrated values and how they were obtained

**Status:** normative. Every number here is measured, not chosen. Re-run the
calibration and update this file whenever the domain, sigma, or the analysis
changes — a `VerificationSpec` whose `calibrated_by` does not match a dated
entry below is not trustworthy.

---

## 1. The rule

`min_trials_per_group` is **never hand-picked**. It is the smallest n whose
**95% CI lower bound clears 0.80 power** for the domain's target effect, under
the domain's *default* analysis, using realistically messy agent data.

Requiring the CI lower bound rather than the point estimate matters: at n=15 the
point estimate was 0.799 and would have looked like a pass under a sloppier
reading.

---

## 2. Synthetic domain — calibrated 2026-09-05

Landscape constants: `SIGMA = 0.50`, skill multiplier `0.60 + 0.40*skill`,
trials drawn at skill in [0.0, 0.3] (early-run conditions, the hard case).

### 2.1 Target effect

`CLOVER` vs `NONE` at spacing in {4, 5}. True synergy delta `+1.00`, realised as
`+0.60` to `+0.72` after the skill multiplier — i.e. **1.2 to 1.4 sigma**.

### 2.2 Power (alpha = 0.01, one-sided, with a Hedges' g >= 0.5 floor)

| n/group | controlled Welch | controlled ANCOVA | uncontrolled Welch | uncontrolled ANCOVA |
|---:|---:|---:|---:|---:|
| 4 | 0.108 | 0.073 | 0.042 | 0.013 |
| 8 | 0.341 | 0.389 | 0.087 | 0.106 |
| 12 | 0.547 | 0.667 | 0.146 | 0.202 |
| 15 | 0.671 | 0.800 | 0.188 | 0.283 |
| 20 | 0.829 | 0.913 | 0.256 | 0.393 |
| 25 | 0.910 | 0.964 | 0.329 | 0.474 |

*controlled* = the agent fixed `plant_day` and used the `crop_health` hint to
set water. *uncontrolled* = the agent varied everything.

### 2.3 Chosen n (8000 replicates per cell, ANCOVA, the default analysis)

| n | power | 95% CI | clears 0.80 |
|---:|---:|---|---|
| 15 | 0.799 | [0.790, 0.807] | no |
| **16** | **0.835** | **[0.826, 0.843]** | **yes** |
| 18 | 0.883 | [0.876, 0.890] | yes |
| 20 | 0.920 | [0.914, 0.926] | yes |

**`min_trials_per_group = 16`.**

### 2.4 Null calibration (single stage)

True delta zero (`CLOVER` vs `CLOVER`). The test must hold its nominal rate.

At **alpha = 0.01** with the `g >= 0.5` floor, 20 000 replicates:

| n | Welch | ANCOVA | nominal |
|---:|---:|---:|---:|
| 10 | 0.0090 | 0.0076 | 0.01 |
| 12 | 0.0096 | 0.0092 | 0.01 |
| 15 | 0.0101 | 0.0075 | 0.01 |

At **alpha = 0.05** with no effect-size floor — the cleanest calibration check,
since the floor is a second condition that confounds the reading:

| test | replicates | FPR | 95% CI | contains nominal |
|---|---:|---:|---|---|
| Welch | 2 000 | 0.0395 | [0.0314, 0.0490] | **no** |
| Welch | 20 000 | 0.0485 | [0.0455, 0.0515] | yes |
| ANCOVA | 2 000 | 0.0565 | [0.0468, 0.0675] | yes |
| ANCOVA | 20 000 | 0.0516 | [0.0486, 0.0548] | yes |

**Both tests are calibrated on nominal.** The 2 000-replicate Welch row is
Monte Carlo noise: its CI upper bound (0.0490) misses 0.05 by 0.001.

#### Why the fast tier does not assert a two-sided interval

A first version of this test asserted "the 95% CI contains nominal" at 2 000
replicates. That assertion is **flaky by construction** — a 95% interval
excludes the true value 5% of the time, which is what a 95% interval *means*.
Choosing a seed that passes would be p-hacking.

The fast tier therefore asserts only the safety-critical direction — the test
must not reject *more* often than alpha, since over-rejection inflates every
downstream discovery count — plus a floor catching a test that never rejects
at all. The precise two-sided calibration lives in the nightly tier at 20 000
replicates, where the interval is tight enough for the question to be
answerable.

> An earlier draft of this document attributed the low Welch figure to a
> platykurtic mixture distribution. That was wrong: the marginal yield
> distribution is essentially normal (skew +0.02, excess kurtosis −0.07,
> D'Agostino p = 0.13). It was simply noise.

### 2.5 Why sigma = 0.50

Two constraints pull in opposite directions and 0.50 satisfies both.

*A single lucky trial must not be proof.* P(one treatment trial beats one
control trial):

| comparison | novice | early | expert |
|---|---:|---:|---:|
| optimum vs trap (delta 0.50 raw) | 0.664 | 0.695 | 0.760 |
| target CLOVER vs NONE (delta 1.00 raw) | 0.802 | 0.846 | 0.921 |

All below the 0.85 ceiling except an expert agent on the target comparison. That
exception is acceptable and arguably correct: skill 1.0 takes roughly 50 trials
to reach, so by the time an agent is that good it has earned the discrimination.
The property holds where it matters — early, when discoveries are actually made.

---

## 2.6 Full-pipeline false-confirmation rate — measured 2026-09-05

20 000 replicates, no true effect anywhere, `n = 16`, ANCOVA, `alpha = 0.01`,
`confirmation_alpha = 0.05`.

| stage | false rate | 95% CI (Clopper-Pearson) |
|---|---:|---|
| reaches `SUPPORTED` | 0.00820 (164/20 000) | — |
| reaches `CONFIRMED` | **0.00050** (10/20 000) | **[0.00024, 0.00092]** |

Predicted `alpha * confirmation_alpha = 0.01 * 0.05 = 0.00050`. The measurement
lands **exactly** on the prediction.

Two things this establishes:

1. **Requiring independent replication does real work.** It cuts the false rate
   by a factor of 16, from 0.0082 to 0.0005. Replication is not ceremony.
2. **The pre-registered ceiling of 2e-3 holds with room to spare** — the CI
   upper bound is 0.00092, less than half the ceiling.

This is the number that licenses the phrase "verified discovery" anywhere in a
report. Without it, `p = 0.003` could just as easily be a bug in our own
statistics as a fact about the world.

---

## 3. Trial budget — why 4 plots per agent

A claim costs about `2 * 16 = 32` trials to reach SUPPORTED and a similar number
again for CONFIRMED. Each trial costs two actions (PLANT then HARVEST), so the
action budget binds before crop maturation does.

| plots/agent | trials/agent (400 ticks) | run total (5 agents) | claims affordable |
|---:|---:|---:|---:|
| 1 | 66 | 330 | 5 |
| 2 | 133 | 665 | 11 |
| **4** | **200** | **1000** | **16** |
| 6 | 200 | 1000 | 16 |

One plot per agent would make the run affordable for only ~5 claims, which is
too tight to observe anything social. **`max_concurrent_plots = 4`**; 6 adds
nothing because the two-actions-per-trial cost saturates first.

---

## 4. The three acceptance tests

Never conflated into one number.

| | Purpose | Design | Assertion |
|---|---|---|---|
| **Calibration** (CI, every commit) | does the single-stage test hold nominal alpha? | SUPPORTED stage alone at a deliberately loose alpha=0.05, 2000 replicates | Clopper-Pearson CI contains 0.05 |
| **Robustness** (nightly) | full-pipeline false-confirmation rate | SUPPORTED -> CONFIRMED with replication, 20 000 replicates | estimate + exact CI reported; hard ceiling 2e-3 |
| **Acceptance** (the report) | the scientific claim | pre-registered, stated as the robustness interval | never a bare pass/fail |

The loose alpha in the calibration test is deliberate. A 5e-4 rate cannot be
characterised at 2000 replicates, so the fast test measures something it *can*
resolve; the nightly test measures the real pipeline rate.

**The null harness does not run the simulation.** It uses the verifier plus the
vectorised generator in `tests/support/evidence.py`, so 20 000 replicates take
seconds rather than days. This is the only reason characterising the pipeline
rate is feasible at all.

---

## 4.1 Phase 4 result — the control arms discriminate (2026-09-05)

1000 ticks, 5 agents, seed 42, synthetic domain. Both arms learn water from the
public `crop_health` signal and neither imports the domain.

| arm | found a global optimum | confirmed claims | best confirmed effect |
|---|---:|---|---:|
| `scripted_factorial` | **5/5 agents** | 4 (BEANS at spacing 2) | **+1.455** |
| `scripted_greedy` | 2/5 agents | 3 (MARIGOLD at spacing 3 — the trap) | +0.356 |

Greedy is **not wrong**, it is stuck: the trap is a real +0.25 synergy, so it
banks a claim that is true and small. The factorial arm banks a claim four
times larger. That gap is the yardstick any LLM policy is measured against.

A representative confirmed claim, reported as the protocol requires:

```
[comparison] companion=BEANS and spacing=2 yields more than
             companion=NONE and spacing=2 by at least 0.30
effect = +1.412   95% CI [+1.140, +1.683]
p = 1.73e-11      Hedges g = 2.17      n = 16/16
replication = independent_rediscovery by agent 1
```

Two calibration notes learned the hard way:

* **4 replicates per cell was not enough.** At sigma 0.50 a 4-trial cell mean
  has SE 0.25 against between-cell differences of ~0.5, and both policies
  ranked 25 cells wrongly — factorial picked `MARIGOLD,4` and greedy picked
  `NONE,5`. Six replicates fixed it.
* **Water must be calibrated BEFORE the factorial, not during it.** Learning
  water while sweeping cells means early cells are measured at the wrong water
  and look bad for a reason unrelated to their companion.

## 4.2 Making GENERALIZED reachable (2026-09-05)

A state that is theoretically defined but practically unreachable is not a
metric. In the first Phase 4 run only 40 of 1656 trials (2.4%) landed on holdout
tiles and **no claim ever reached GENERALIZED**.

### The invariant that had to survive the fix

```
agent cannot know stratum
agent cannot target stratum
verifier can accumulate enough holdout evidence
```

So the fix had to be at the WORLD level, not by telling anyone anything.

### Three changes, none of which expose the partition

1. **Fallow after harvest** (`FALLOW_TICKS = 10`). A harvested tile rests before
   it can be replanted. Applied identically to every tile regardless of
   stratum, so it rotates agents around the map while carrying zero information
   about which tiles count for which stage. Tiles farmed per agent went from a
   tight cluster to 15-60; holdout trials went 40 -> 136.
2. **Rebalanced partition**, 70/20/10 -> **60/25/15**.
3. **A separate, calibrated holdout sample size.**

### The actual bug, which was neither of those

`CONFIRMED` was not in `OPEN_STATES`, so `open_records()` dropped confirmed
claims and the verifier **never evaluated the generalized stage at all** — even
with 28 spec / 30 baseline holdout trials available against a requirement of 16.
Split into two sets: `OPEN_STATES` (counts against an agent's registration
budget) and `ADVANCEABLE_STATES` (the verifier keeps re-examining, and includes
`CONFIRMED`). Conflating "still costs a slot" with "still worth checking" is
what made the state decorative.

### `min_trials_holdout` — calibrated at alpha = 0.05, 6000 replicates

The holdout stage re-tests an effect already established twice, at a looser
alpha, so it needs fewer trials. Still measured, never chosen:

| n | power | 95% CI | clears 0.80 |
|---:|---:|---|---|
| 8 | 0.677 | [0.666, 0.689] | no |
| 10 | 0.779 | [0.768, 0.789] | no |
| **12** | **0.857** | **[0.847, 0.865]** | **yes** |
| 16 | 0.921 | [0.914, 0.928] | yes |

**`min_trials_holdout = 12`.** Null FPR at that n: 0.0513, 95% CI
[0.0483, 0.0545] — contains nominal 0.05.

### Result

| ticks | seeds reaching GENERALIZED |
|---:|---|
| 800 | 0/5 (claims register ~tick 620-745; no time to accumulate) |
| 1000 | 3/5 |
| **1200** | **5/5** — first transition at tick 848-1103 |

Time-to-GENERALIZED is a property of the *policy*, not a defect: brute-force
factorial search spends ~190 trials per agent before it registers anything. A
policy that formed hypotheses earlier would get there sooner, which is exactly
the kind of difference the LLM arms are meant to expose.

### Verified: agents cannot target strata

Chi-square of **distinct tiles farmed** against the stratum mix of arable tiles
(trial counts are inflated by revisits and are not independent draws):

| seed | tiles | discovery | confirmation | holdout | chi2 p |
|---:|---:|---:|---:|---:|---:|
| 42 | 121 | 71.9% | 12.4% | 15.7% | 0.056 |
| 43 | 80 | 58.8% | 26.2% | 15.0% | 0.897 |
| 44 | 120 | 60.8% | 24.2% | 15.0% | 0.802 |
| 45 | 132 | 62.1% | 21.2% | 16.7% | 0.936 |
| 46 | 77 | 57.1% | 23.4% | 19.5% | 0.711 |

Available (arable): 58.6% / 25.3% / 16.1%. All p > 0.05 — no evidence of
targeting. Seed 42's earlier-looking deviation is sampling noise at p = 0.056,
which is why this is measured across seeds rather than asserted from one.

---

## 4.3 Protocol adoption — a fair denominator

`registered_claims / X` is only fair if `X` counts occasions when the agent
could reasonably have formed a hypothesis. Dividing by ticks, or trials, or by
what the world knows to be true, punishes an agent for never having encountered
comparable evidence — which is not a failure to adopt the protocol.

**Opportunity** (`aiciv/metrics/adoption.py`): this agent, from its OWN observed
trials, held at least `MIN_PER_ARM = 6` results in each of two levels of one
parameter. That is when a comparison became formulable *to it*.

Two rules the module obeys, both enforced by tests:

1. **Only agent-visible fields.** Rows are projected through `VISIBLE_FIELDS`
   before use, so `stratum`, `true_mu` and `signature` cannot influence the
   answer. `test_result_is_identical_with_and_without_hidden_fields` proves it
   by removing them and asserting the output is unchanged.
2. **No opportunity means UNDEFINED, not zero.** An agent that never had a
   formulable comparison is excluded from the aggregate rather than scored 0.0.

Two rates, answering different questions:

| metric | question |
|---|---|
| `formal_protocol_adoption` | did it register anything at all, per salient chance |
| `matched_adoption` | did it register a claim about *that* contrast |

`salient` uses the agent's **own observed** standardised effect (>= 0.5), never
ground truth: a contrast it could form but which looked like nothing to it is a
weaker expectation than one that looked substantial.

### Still open

**The shadow verifier's ratio is not like-for-like.** It scans marginal
single-variable hypotheses and cannot represent the companion x spacing
interaction, so agent claims are strictly richer than anything it proposes. Its
`protocol_adoption_rate` is superseded for per-agent grading by
`aiciv/metrics/adoption.py`; the shadow figure remains useful only as a coarse
"was there discoverable signal at all" check. See the limitation note in
`knowledge/shadow.py`.

---

## 5. Known soft spot

Uncontrolled trials reach only 0.393 power even at n=20. That is the intended
design pressure toward controlled comparison, but it carries a real risk: a 7B
model that never controls its trials will verify **nothing**, and the LLM arms
could return zero discoveries.

This is a genuine possible outcome, not a bug to be tuned away. It is
expected-failure #3 in the plan, and the shadow verifier exists so we can report
it with evidence rather than quietly loosening alpha until results appear.
