"""
Tests for src/analysis/stats.py -- the paired-comparison and trend-test
machinery used in 10_statistical_analysis.ipynb.
"""

import numpy as np
import pandas as pd

from src.analysis.stats import (
    fdr_correct,
    mean_ci,
    paired_cohens_d,
    paired_comparison,
    per_seed_spearman_trend,
)


def test_mean_ci_known_values():
    x = [1, 2, 3, 4, 5]
    result = mean_ci(x)

    assert result["mean"] == 3.0
    assert result["ci_lower"] < 3.0 < result["ci_upper"]
    assert result["n"] == 5


def test_mean_ci_single_value_has_zero_width():
    result = mean_ci([5.0])
    assert result["ci_lower"] == result["ci_upper"] == 5.0


def test_paired_cohens_d_identical_arrays_is_zero():
    x = [1, 2, 3, 4]
    assert paired_cohens_d(x, x) == 0.0


def test_paired_cohens_d_sign_matches_direction():
    # Non-constant differences (9, 8, 10, 9) -- a genuinely varying
    # paired difference, not the constant-difference edge case (which
    # has zero variance and is covered separately below).
    x = [10, 11, 12, 14]
    y = [1, 3, 2, 5]
    assert paired_cohens_d(x, y) > 0
    assert paired_cohens_d(y, x) < 0


def test_paired_cohens_d_constant_difference_is_zero_not_infinite():
    # A degenerate edge case: every pair has the identical difference,
    # so the difference has zero variance -- Cohen's d (mean_diff /
    # std_diff) is undefined here; returning 0.0 rather than raising
    # or dividing by zero is the documented, deliberate behavior.
    x = [10, 11, 12, 13]
    y = [1, 2, 3, 4]
    assert paired_cohens_d(x, y) == 0.0


def test_paired_comparison_identical_samples():
    x = [0.6, 0.62, 0.61, 0.63, 0.60, 0.64, 0.61, 0.62]
    result = paired_comparison(x, x, label="identical")

    assert result["mean_diff"] == 0.0
    assert result["wilcoxon_p"] == 1.0
    assert result["cohens_d"] == 0.0


def test_paired_comparison_clear_difference_is_significant():
    # Classical baseline (~99.8%) vs VQC (~62%) -- should be a very
    # clear, significant difference (this mirrors the real RQ5
    # comparison in the project).
    classical = [0.998, 0.998, 0.999, 0.998, 0.999, 0.996, 0.998, 0.998]
    vqc = [0.629, 0.615, 0.618, 0.627, 0.618, 0.628, 0.621, 0.641]

    result = paired_comparison(classical, vqc, label="classical_vs_vqc")

    assert result["mean_diff"] > 0.3
    assert result["wilcoxon_p"] < 0.05
    assert result["ttest_p"] < 0.05
    assert result["cohens_d"] > 1.0  # large effect


def test_fdr_correct_all_significant_stay_ordered():
    # Known small example: FDR-corrected values must be
    # non-decreasing when sorted by raw p-value, and >= raw p-values.
    raw = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    corrected = fdr_correct(raw)

    assert len(corrected) == len(raw)
    assert np.all(corrected >= raw - 1e-12)

    order = np.argsort(raw)
    assert np.all(np.diff(corrected[order]) >= -1e-12)


def test_fdr_correct_matches_hand_computed_example():
    # Classic textbook example: 5 p-values, BH at alpha=0.05.
    raw = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    corrected = fdr_correct(raw)

    # Rank i (1-indexed) correction: p_i * n / i, then cumulative min
    # from the largest rank down.
    expected_step = raw * 5 / np.array([1, 2, 3, 4, 5])
    expected = np.minimum.accumulate(expected_step[::-1])[::-1]

    np.testing.assert_allclose(corrected, expected)


def test_per_seed_spearman_trend_detects_positive_trend():
    # 3 seeds, each with a perfect increasing trend across 5 levels.
    rows = []
    for seed in [42, 43, 44]:
        for level in range(5):
            rows.append({"seed": seed, "level_num": level, "value": level * 2 + seed * 0.001})

    df = pd.DataFrame(rows)
    correlations, summary = per_seed_spearman_trend(df, "seed", "level_num", "value")

    assert len(correlations) == 3
    assert np.all(correlations > 0.99)
    assert summary["mean_correlation"] > 0.99
    assert summary["ttest_p"] < 0.05


def test_per_seed_spearman_trend_no_relationship():
    rng = np.random.default_rng(0)
    rows = []
    for seed in [42, 43, 44, 45, 46]:
        for level in range(6):
            rows.append({"seed": seed, "level_num": level, "value": rng.normal()})

    df = pd.DataFrame(rows)
    correlations, summary = per_seed_spearman_trend(df, "seed", "level_num", "value")

    # With pure noise, the mean correlation should not be significantly
    # different from 0 (not a strict guarantee, but true for this seed).
    assert summary["ttest_p"] > 0.05
