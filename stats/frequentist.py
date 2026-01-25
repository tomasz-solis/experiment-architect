"""Frequentist statistical analysis functions."""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from config import Z_ALPHA, ALPHA


def check_srm(n_a: int, n_b: int, threshold: float = 0.05) -> Tuple[bool, float]:
    """Check for Sample Ratio Mismatch.

    Args:
        n_a: Sample size group A
        n_b: Sample size group B
        threshold: Acceptable deviation from 50/50

    Returns:
        tuple: (has_mismatch, ratio)
    """
    ratio = n_a / (n_a + n_b)
    has_mismatch = abs(ratio - 0.5) > threshold
    return has_mismatch, ratio


def calculate_lift(mean_a: float, mean_b: float) -> float:
    """Calculate relative lift."""
    return (mean_b - mean_a) / mean_a


def chi_squared_test(successes_a: int, failures_a: int, successes_b: int, failures_b: int) -> Dict[str, float]:
    """Run Chi-Squared test for binary outcomes.

    Returns:
        dict: Test results with statistic, p-value, effect_size
    """
    contingency_table = [
        [successes_a, failures_a],
        [successes_b, failures_b]
    ]
    stat, p_val, _, _ = chi2_contingency(contingency_table)

    # Effect size (Cramér's V)
    n = successes_a + failures_a + successes_b + failures_b
    cramers_v = np.sqrt(stat / n)

    return {
        "statistic": stat,
        "p_value": p_val,
        "effect_size": cramers_v,
        "effect_size_label": "Cramér's V",
        "test_name": "Chi-Squared Test"
    }


def welch_t_test(group_a: pd.Series, group_b: pd.Series) -> Dict[str, float]:
    """Run Welch's T-Test for continuous outcomes.

    Returns:
        dict: Test results with statistic, p-value, effect_size
    """
    stat, p_val = ttest_ind(group_a, group_b, equal_var=False)

    # Effect size (Cohen's d)
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_std = np.sqrt(((n_a - 1) * group_a.std()**2 +
                         (n_b - 1) * group_b.std()**2) /
                        (n_a + n_b - 2))
    cohens_d = (group_b.mean() - group_a.mean()) / pooled_std

    return {
        "statistic": stat,
        "p_value": p_val,
        "effect_size": cohens_d,
        "effect_size_label": "Cohen's d",
        "test_name": "Welch's T-Test"
    }


def confidence_interval_binary(cr_a: float, cr_b: float, n_a: int, n_b: int) -> Tuple[float, float]:
    """Calculate confidence interval on lift for binary outcomes.

    Returns:
        tuple: (ci_lower, ci_upper)
    """
    se_a = np.sqrt(cr_a * (1 - cr_a) / n_a)
    se_b = np.sqrt(cr_b * (1 - cr_b) / n_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)

    margin = Z_ALPHA * se_diff
    ci_lower = ((cr_b - margin) - cr_a) / cr_a
    ci_upper = ((cr_b + margin) - cr_a) / cr_a

    return ci_lower, ci_upper


def confidence_interval_continuous(group_a: pd.Series, group_b: pd.Series) -> Tuple[float, float]:
    """Calculate confidence interval on lift for continuous outcomes.

    Returns:
        tuple: (ci_lower, ci_upper)
    """
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    n_a = len(group_a)
    n_b = len(group_b)

    se_a = group_a.std() / np.sqrt(n_a)
    se_b = group_b.std() / np.sqrt(n_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)

    margin = Z_ALPHA * se_diff
    ci_lower = ((mean_b - margin) - mean_a) / mean_a
    ci_upper = ((mean_b + margin) - mean_a) / mean_a

    return ci_lower, ci_upper


def is_significant(p_value: float) -> bool:
    """Check if result is statistically significant."""
    return p_value < ALPHA


def calculate_sample_size(baseline: float, mde: float, daily_traffic: int, split_ratio: float = 0.5) -> Dict[str, int]:
    """Calculate required sample size and duration for A/B test.

    Returns:
        dict: Sample size calculation results
    """
    from config import Z_ALPHA, Z_BETA

    p2 = baseline * (1 + mde)
    delta = p2 - baseline
    split_factor = (1 / split_ratio) + (1 / (1 - split_ratio))
    pooled_var = (baseline * (1 - baseline) + p2 * (1 - p2)) / 2
    z_score = (Z_ALPHA + Z_BETA)**2
    n_total = z_score * pooled_var * split_factor / (delta**2)
    days = np.ceil(n_total / daily_traffic)

    # Calculate inefficiency of unequal split
    loss_pct = 0
    if split_ratio != 0.5:
        loss_pct = int((((split_factor / 4) - 1) * 100))

    return {
        "n_total": int(n_total),
        "days": int(days),
        "split_penalty": loss_pct
    }


def calculate_reverse_mde(baseline: float, daily_visitors: int, weeks: int) -> Dict[str, any]:
    """Calculate minimum detectable effect given constraints.

    Returns:
        dict: MDE calculation results or error
    """
    from config import Z_ALPHA, Z_BETA

    total_n = daily_visitors * (weeks * 7)
    z_score = (Z_ALPHA + Z_BETA)**2
    pooled_var = 2 * baseline * (1 - baseline)

    # Sanity checks
    if total_n < 100:
        return {"error": "Not enough traffic. You need at least 100 total visitors."}
    if baseline < 0.001 or baseline > 0.999:
        return {"error": "Baseline conversion must be between 0.1% and 99.9%"}

    feasible_mde = np.sqrt((z_score * pooled_var) / total_n) / baseline

    if feasible_mde > 10.0:
        return {"error": f"Your detectable lift is {feasible_mde:.1%} - way too high. Increase traffic or wait longer."}

    return {"mde": feasible_mde}
