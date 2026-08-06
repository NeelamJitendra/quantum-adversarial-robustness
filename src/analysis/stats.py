"""
Statistical methodology for the thesis's RQ comparisons (see
src/config.py / the thesis architecture plan): 8 seeds, t-based 95% CI
on the mean, paired Wilcoxon signed-rank as the primary significance
test (paired t-test as a secondary robustness check), paired Cohen's d
as effect size, and Benjamini-Hochberg FDR correction across each
family of comparisons (not Bonferroni, which is overly conservative at
n=8 with likely-correlated comparisons).

statsmodels is not installed (avoided adding a new dependency this
late in the timeline -- see the mitigation module design in the
architecture plan for the same reasoning); Benjamini-Hochberg is
simple enough to hand-roll correctly.
"""

from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_ci(x: Sequence[float], confidence: float = 0.95) -> dict:
    """
    t-based confidence interval on the mean (appropriate at small n,
    e.g. the project's 8 seeds -- a normal-approximation CI understates
    uncertainty this small).
    """

    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sem = stats.sem(x, ddof=1) if n > 1 else 0.0

    if n > 1:
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_crit * sem
    else:
        margin = 0.0

    return {
        "mean": float(mean),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "n": n,
    }


def paired_cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean difference / std of
    differences. Consistent with the convention already used in
    04D_phase0_diagnostics's quantum-output effect-size analysis."""

    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)

    if diff.std(ddof=1) == 0:
        return 0.0

    return float(diff.mean() / diff.std(ddof=1))


def paired_comparison(x: Sequence[float], y: Sequence[float], label: str = "") -> dict:
    """
    Full paired comparison between two same-length, seed-matched
    samples (e.g. VQC accuracy vs classical accuracy, same 8 seeds).

    Returns
    -------
    dict
        label, n, mean_x, mean_y, mean_diff, ci_diff (t-based 95% CI on
        the paired difference), wilcoxon_p (primary test),
        ttest_p (secondary robustness check), cohens_d.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y

    diff_ci = mean_ci(diff)

    # Wilcoxon signed-rank requires at least one non-zero difference.
    if np.all(diff == 0):
        wilcoxon_p = 1.0
    else:
        _, wilcoxon_p = stats.wilcoxon(x, y)

    _, ttest_p = stats.ttest_rel(x, y)

    return {
        "label": label,
        "n": len(x),
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "mean_diff": diff_ci["mean"],
        "ci_diff_lower": diff_ci["ci_lower"],
        "ci_diff_upper": diff_ci["ci_upper"],
        "wilcoxon_p": float(wilcoxon_p),
        "ttest_p": float(ttest_p),
        "cohens_d": paired_cohens_d(x, y),
    }


def fdr_correct(p_values: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg false discovery rate correction.

    Parameters
    ----------
    p_values : array-like of raw p-values.

    Returns
    -------
    np.ndarray
        FDR-corrected p-values, same order as input.
    """

    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    corrected = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity: each corrected p-value can't exceed the
    # next-larger one's corrected value (standard BH step-up procedure).
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)

    result = np.empty(n)
    result[order] = corrected

    return result


def per_seed_spearman_trend(
    df, seed_col: str, x_col: str, y_col: str
) -> Tuple[np.ndarray, dict]:
    """
    Computes a Spearman correlation between x_col and y_col
    *separately for each seed*, then a one-sample test of whether the
    mean per-seed correlation differs from 0 -- avoids the
    pseudo-replication of pooling all seeds into one correlation
    (which would treat each seed's several rows as independent, when
    they share a common trained model).

    Used for RQ(b)/(c)-style trend questions (e.g. "does accuracy
    decline as noise level increases", "are accuracy and robustness
    correlated across noise conditions").

    Returns
    -------
    per_seed_correlations : np.ndarray
    summary : dict
        mean_correlation, ci (t-based), ttest_p (one-sample, H0: mean
        correlation = 0), wilcoxon_p (one-sample signed-rank vs 0).
    """

    correlations = []

    for seed, group in df.groupby(seed_col):
        if group[x_col].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(group[x_col], group[y_col])
        correlations.append(rho)

    correlations = np.array(correlations)
    ci = mean_ci(correlations)

    _, ttest_p = stats.ttest_1samp(correlations, popmean=0)

    if np.all(correlations == correlations[0]):
        wilcoxon_p = 1.0 if correlations[0] == 0 else 0.0
    else:
        _, wilcoxon_p = stats.wilcoxon(correlations)

    summary = {
        "mean_correlation": ci["mean"],
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "ttest_p": float(ttest_p),
        "wilcoxon_p": float(wilcoxon_p),
        "n_seeds": len(correlations),
    }

    return correlations, summary
