"""Statistical primitives for claim verification.

Every verdict reports effect, confidence interval, adjusted p, effect size and
n -- never a bare p-value. A number without an interval is not a result.

Nothing here knows about agents, worlds or domains; it operates on arrays. That
keeps the null-calibration harness (tests/test_calibration.py) able to run tens
of thousands of replicates in seconds without touching the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    effect: float                    # treat mean - control mean (or beta_treat)
    ci_low: float
    ci_high: float
    p_value: float
    hedges_g: float
    n_treat: int
    n_ctrl: int
    df: float
    test_name: str
    p_adjusted: float | None = None
    detail: dict = field(default_factory=dict)

    def describe(self) -> str:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return (
            f"effect = {self.effect:+.3f}   95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]\n"
            f"p{'_adjusted' if self.p_adjusted is not None else ''} = {p:.4g}   "
            f"Hedges g = {self.hedges_g:.3f}   n = {self.n_treat + self.n_ctrl} "
            f"({self.n_treat}/{self.n_ctrl})   [{self.test_name}]"
        )


def hedges_g(treat: np.ndarray, ctrl: np.ndarray) -> float:
    """Standardised mean difference with the small-sample correction J."""
    n1, n2 = len(treat), len(ctrl)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = np.var(treat, ddof=1), np.var(ctrl, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled <= 0:
        return 0.0
    d = (np.mean(treat) - np.mean(ctrl)) / np.sqrt(pooled)
    denom = 4 * (n1 + n2) - 9
    j = 1.0 - 3.0 / denom if denom > 0 else 1.0
    return float(d * j)


def welch(treat: np.ndarray, ctrl: np.ndarray, *, conf: float = 0.95) -> TestResult:
    """One-sided Welch t-test (H1: treat > ctrl) with a two-sided CI on the diff.

    One-sided because a claim declares its predicted direction at registration;
    a two-sided test would reward an agent for being wrong in an interesting way.
    The CI stays two-sided because that is what a reader needs to judge size.
    """
    treat = np.asarray(treat, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    n1, n2 = len(treat), len(ctrl)
    if n1 < 2 or n2 < 2:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, n1, n2, 0.0, "welch",
                          detail={"error": "insufficient n"})

    res = stats.ttest_ind(treat, ctrl, equal_var=False, alternative="greater")
    v1, v2 = np.var(treat, ddof=1), np.var(ctrl, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    effect = float(np.mean(treat) - np.mean(ctrl))

    if se == 0:
        lo = hi = effect
        df = float(n1 + n2 - 2)
    else:
        df = float((v1 / n1 + v2 / n2) ** 2 /
                   ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
        tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
        lo, hi = effect - tcrit * se, effect + tcrit * se

    return TestResult(effect, float(lo), float(hi), float(res.pvalue),
                      hedges_g(treat, ctrl), n1, n2, df, "welch")


def ancova(
    y: np.ndarray,
    treat: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    conf: float = 0.95,
) -> TestResult:
    """OLS y ~ 1 + treat + covariates, one-sided t on the treat coefficient.

    Fitted with numpy.linalg.lstsq so we carry no statsmodels dependency.
    ``covariates`` must contain PRE_TREATMENT variables only for causal claims;
    aiciv.knowledge.causal enforces that, not this function.
    """
    y = np.asarray(y, dtype=float)
    treat = np.asarray(treat, dtype=float).reshape(-1, 1)
    n = len(y)
    cols = [np.ones((n, 1)), treat]
    if covariates is not None and np.size(covariates):
        C = np.asarray(covariates, dtype=float).reshape(n, -1)
        # Drop zero-variance columns; they are collinear with the intercept and
        # would make the design rank-deficient.
        keep = [i for i in range(C.shape[1]) if np.ptp(C[:, i]) > 0]
        if keep:
            cols.append(C[:, keep])
    X = np.hstack(cols)
    p = X.shape[1]
    df = n - p
    n1, n2 = int(treat.sum()), int(n - treat.sum())
    if df <= 0 or n1 < 2 or n2 < 2:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, n1, n2, 0.0, "ancova",
                          detail={"error": "insufficient df"})

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / df
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, n1, n2, 0.0, "ancova",
                          detail={"error": "singular design"})

    se = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    effect = float(beta[1])
    if se == 0:
        return TestResult(effect, effect, effect, 0.0 if effect > 0 else 1.0,
                          0.0, n1, n2, float(df), "ancova")

    t = effect / se
    pval = float(stats.t.sf(t, df))                       # one-sided, H1: beta > 0
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
    g = hedges_g(y[treat.ravel() == 1], y[treat.ravel() == 0])
    return TestResult(effect, effect - tcrit * se, effect + tcrit * se, pval,
                      g, n1, n2, float(df), "ancova",
                      detail={"n_covariates": p - 2})


def benjamini_hochberg(pvals: list[float], alpha: float) -> list[bool]:
    """BH step-up. Returns a rejection mask aligned with the input order."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.asarray(pvals, dtype=float)[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    if k:
        out[order[:k]] = True
    return out.tolist()


def clopper_pearson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial CI. Used to report false-positive rates honestly."""
    if n == 0:
        return (0.0, 1.0)
    a = 1 - conf
    lo = 0.0 if successes == 0 else stats.beta.ppf(a / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else stats.beta.ppf(1 - a / 2, successes + 1, n - successes)
    return (float(lo), float(hi))
