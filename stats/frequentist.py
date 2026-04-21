"""Frequentist statistical helpers for experiment design and analysis."""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from config import ALPHA, Z_ALPHA, Z_BETA


def _validate_binary_inputs(
    successes_a: int,
    failures_a: int,
    successes_b: int,
    failures_b: int,
) -> None:
    """Raise a clear error when binary outcome counts are invalid."""
    for label, successes, failures in (
        ("A", successes_a, failures_a),
        ("B", successes_b, failures_b),
    ):
        if successes < 0 or failures < 0:
            raise ValueError(f"Group {label}: counts cannot be negative.")
        if successes + failures == 0:
            raise ValueError(f"Group {label}: total sample size is zero.")


def check_srm(n_a: int, n_b: int, threshold: float = 0.05) -> Tuple[bool, float]:
    """Check whether the observed split deviates too far from 50/50."""
    if n_a < 0 or n_b < 0:
        raise ValueError("Sample sizes cannot be negative.")
    if n_a + n_b == 0:
        raise ValueError("At least one observation is required to check SRM.")

    ratio = n_a / (n_a + n_b)
    has_mismatch = abs(ratio - 0.5) > threshold
    return has_mismatch, ratio


def calculate_lift(mean_a: float, mean_b: float) -> float:
    """Calculate relative lift from control to variant."""
    if mean_a == 0:
        raise ValueError("Control mean must be non-zero to calculate lift.")
    return (mean_b - mean_a) / mean_a


def bonferroni_adjusted_alpha(alpha: float = ALPHA, n_comparisons: int = 1) -> float:
    """Return a Bonferroni-adjusted alpha for multiple primary comparisons."""
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1.")
    return alpha / n_comparisons


def build_frequentist_guardrails(
    n_comparisons: int = 1,
    peeked_early: bool = False,
    alpha: float = ALPHA,
) -> dict[str, int | float | bool]:
    """Summarize the main guardrails that affect p-value interpretation.

    The Bonferroni adjustment is a conservative default when several primary
    metrics are reviewed. Early peeking does not have a clean correction here,
    so the function surfaces a warning flag rather than pretending to adjust it
    without a full sequential design.
    """
    adjusted_alpha = bonferroni_adjusted_alpha(alpha=alpha, n_comparisons=n_comparisons)
    return {
        "n_comparisons": n_comparisons,
        "adjusted_alpha": adjusted_alpha,
        "alpha_adjusted": n_comparisons > 1,
        "peeked_early": peeked_early,
    }


def chi_squared_test(
    successes_a: int,
    failures_a: int,
    successes_b: int,
    failures_b: int,
) -> dict[str, float | str | bool]:
    """Run a chi-squared test for a binary outcome."""
    _validate_binary_inputs(successes_a, failures_a, successes_b, failures_b)

    contingency_table = [
        [successes_a, failures_a],
        [successes_b, failures_b],
    ]
    statistic, p_value, _, expected = chi2_contingency(contingency_table)

    total_n = successes_a + failures_a + successes_b + failures_b
    cramers_v = np.sqrt(statistic / total_n)
    min_expected = float(np.min(expected))

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(cramers_v),
        "effect_size_label": "Cramer's V",
        "test_name": "Chi-Squared Test",
        "min_expected_count": min_expected,
        "chi_square_valid": min_expected >= 5,
    }


def welch_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
) -> dict[str, float | str]:
    """Run Welch's t-test for a continuous outcome."""
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Each group must have at least 2 observations for Welch's t-test.")
    if group_a.isna().any() or group_b.isna().any():
        raise ValueError("Continuous outcome groups cannot contain missing values.")

    statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)

    n_a = len(group_a)
    n_b = len(group_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * group_a.std() ** 2 + (n_b - 1) * group_b.std() ** 2)
        / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        raise ValueError("Effect size is undefined when both groups have zero variance.")

    cohens_d = (group_b.mean() - group_a.mean()) / pooled_std

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(cohens_d),
        "effect_size_label": "Cohen's d",
        "test_name": "Welch's T-Test",
    }


def confidence_interval_binary(
    cr_a: float,
    cr_b: float,
    n_a: int,
    n_b: int,
) -> Tuple[float, float]:
    """Calculate a confidence interval on relative lift for binary outcomes."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    if cr_a <= 0:
        raise ValueError("Control conversion rate must be greater than zero.")
    if not (0 <= cr_a <= 1 and 0 <= cr_b <= 1):
        raise ValueError("Conversion rates must be between 0 and 1.")

    se_a = np.sqrt(cr_a * (1 - cr_a) / n_a)
    se_b = np.sqrt(cr_b * (1 - cr_b) / n_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)

    margin = Z_ALPHA * se_diff
    ci_lower = ((cr_b - margin) - cr_a) / cr_a
    ci_upper = ((cr_b + margin) - cr_a) / cr_a
    return ci_lower, ci_upper


def confidence_interval_continuous(
    group_a: pd.Series,
    group_b: pd.Series,
) -> Tuple[float, float]:
    """Calculate a confidence interval on relative lift for continuous outcomes."""
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Each group must have at least 2 observations to build a confidence interval.")
    if group_a.isna().any() or group_b.isna().any():
        raise ValueError("Continuous outcome groups cannot contain missing values.")

    mean_a = group_a.mean()
    mean_b = group_b.mean()
    if mean_a == 0:
        raise ValueError("Control mean must be non-zero to calculate lift.")

    se_a = group_a.std() / np.sqrt(len(group_a))
    se_b = group_b.std() / np.sqrt(len(group_b))
    se_diff = np.sqrt(se_a**2 + se_b**2)

    margin = Z_ALPHA * se_diff
    ci_lower = ((mean_b - margin) - mean_a) / mean_a
    ci_upper = ((mean_b + margin) - mean_a) / mean_a
    return ci_lower, ci_upper


def is_significant(p_value: float, alpha: float = ALPHA) -> bool:
    """Return whether a p-value passes the chosen alpha threshold."""
    if not 0 <= p_value <= 1:
        raise ValueError("p_value must be between 0 and 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    return p_value < alpha


def calculate_sample_size(
    baseline: float,
    mde: float,
    daily_traffic: int,
    split_ratio: float = 0.5,
) -> dict[str, int]:
    """Estimate total sample size and duration for an A/B test.

    This formula is paired with :func:`calculate_reverse_mde`, which is its
    algebraic inverse for the default 50/50 split case.
    """
    if not (0.001 <= baseline <= 0.999):
        raise ValueError("Baseline conversion must be between 0.1% and 99.9%.")
    if mde <= 0:
        raise ValueError("MDE must be greater than zero.")
    if daily_traffic <= 0:
        raise ValueError("Daily traffic must be greater than zero.")
    if not (0 < split_ratio < 1):
        raise ValueError("Split ratio must be between 0 and 1.")

    p2 = baseline * (1 + mde)
    if p2 >= 1:
        raise ValueError("Target lift pushes the projected conversion rate above 100%.")

    delta = p2 - baseline
    split_factor = (1 / split_ratio) + (1 / (1 - split_ratio))
    pooled_var = (baseline * (1 - baseline) + p2 * (1 - p2)) / 2
    z_sum = Z_ALPHA + Z_BETA
    n_total = (z_sum**2) * pooled_var * split_factor / (delta**2)
    days = np.ceil(n_total / daily_traffic)

    split_penalty = 0
    if split_ratio != 0.5:
        split_penalty = int((((split_factor / 4) - 1) * 100))

    return {
        "n_total": int(np.ceil(n_total)),
        "days": int(days),
        "split_penalty": split_penalty,
    }


def calculate_reverse_mde(
    baseline: float,
    daily_visitors: int,
    weeks: int,
    split_ratio: float = 0.5,
) -> dict[str, Any]:
    """Estimate the smallest relative lift detectable within a time window.

    The calculation uses the same variance setup as ``calculate_sample_size``
    so the two functions stay algebraically consistent.
    """
    total_n = daily_visitors * (weeks * 7)

    if total_n < 100:
        return {"error": "Not enough traffic. You need at least 100 total visitors."}
    if not (0.001 <= baseline <= 0.999):
        return {"error": "Baseline conversion must be between 0.1% and 99.9%."}
    if not (0 < split_ratio < 1):
        return {"error": "Split ratio must be between 0 and 1."}

    z_sum = Z_ALPHA + Z_BETA
    pooled_var = baseline * (1 - baseline)
    split_factor = (1 / split_ratio) + (1 / (1 - split_ratio))

    delta_absolute = z_sum * np.sqrt(split_factor * pooled_var / total_n)
    feasible_mde = delta_absolute / baseline

    if feasible_mde > 10.0:
        return {
            "error": (
                f"Detectable lift is {feasible_mde:.1%} - too high. "
                "Increase traffic or extend the test duration."
            )
        }

    return {"mde": float(feasible_mde)}
