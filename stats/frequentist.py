"""Frequentist statistical helpers for experiment design and analysis."""

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare, norm, ttest_ind

from config import (
    ALPHA,
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_RESAMPLES,
    DEFAULT_POWER,
    SRM_P_VALUE_THRESHOLD,
    Z_ALPHA,
    Z_BETA,
)
from stats.power import allocation_cost

# Pooled SD assumes equal variances (classic Cohen's d). The averaged-SD form,
# sqrt((var_a + var_b) / 2), is the variance structure Welch's test itself uses,
# so it is the consistent effect size when the group variances differ.
EffectSizeMethod = Literal["pooled", "averaged"]


class ChiSquaredResult(TypedDict):
    """Result of a chi-squared test for a binary outcome."""

    statistic: float
    p_value: float
    effect_size: float
    effect_size_label: str
    test_name: str
    min_expected_count: float
    chi_square_valid: bool


class WelchTTestResult(TypedDict):
    """Result of Welch's t-test for a continuous outcome."""

    statistic: float
    p_value: float
    effect_size: float
    effect_size_label: str
    test_name: str


# A frequentist significance read is either the binary (chi-squared) or the
# continuous (Welch) result. Both share the keys the UI renders unconditionally.
FrequentistTestResult = ChiSquaredResult | WelchTTestResult


class SampleSizeResult(TypedDict):
    """Sample-size and duration estimate for a two-group A/B test."""

    n_total: int
    days: int
    split_penalty: int


class FrequentistGuardrails(TypedDict):
    """Guardrail summary that affects how a p-value should be interpreted."""

    n_comparisons: int
    adjusted_alpha: float
    alpha_adjusted: bool
    peeked_early: bool


class ReverseMDEResult(TypedDict, total=False):
    """Reverse-MDE estimate. Carries ``mde`` on success or ``error`` on failure."""

    mde: float
    error: str


class SRMResult(TypedDict):
    """Result of a sample-ratio-mismatch check against the intended split."""

    observed_share: float
    expected_share: float
    p_value: float
    has_mismatch: bool


class AttritionResult(TypedDict):
    """Result of a chi-squared test for differential attrition between arms."""

    dropped_share_a: float
    dropped_share_b: float
    p_value: float
    has_differential_attrition: bool


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


def check_srm(n_a: int, n_b: int, expected_share_b: float = 0.5) -> SRMResult:
    """Check whether the observed split deviates from the split the test intended.

    A fixed percentage-point gap is the wrong instrument here: it is blind to
    real mismatches at small samples and fires on ordinary noise at large ones.
    A chi-squared goodness-of-fit test against the *intended* split scales the
    right way with sample size, and it correctly leaves a deliberate uneven
    split (e.g. 70/30) alone as long as the observed split matches it.
    """
    if n_a < 0 or n_b < 0:
        raise ValueError("Sample sizes cannot be negative.")
    total = n_a + n_b
    if total == 0:
        raise ValueError("At least one observation is required to check SRM.")
    if not 0 < expected_share_b < 1:
        raise ValueError("expected_share_b must be between 0 and 1.")

    observed_share = n_b / total
    expected_counts = [total * (1 - expected_share_b), total * expected_share_b]
    _, p_value = chisquare([n_a, n_b], f_exp=expected_counts)

    return {
        "observed_share": observed_share,
        "expected_share": expected_share_b,
        "p_value": float(p_value),
        "has_mismatch": p_value < SRM_P_VALUE_THRESHOLD,
    }


def check_differential_attrition(
    kept_a: int,
    dropped_a: int,
    kept_b: int,
    dropped_b: int,
) -> AttritionResult:
    """Check whether cleaning removed rows unevenly between the two arms.

    SRM catches assignment gone wrong; this catches cleaning gone wrong. A 2x2
    chi-squared test on kept/dropped by arm. When neither arm lost any rows,
    the kept/dropped table has a zero column and the chi-squared expected
    frequencies are undefined, so that case is reported directly as "no
    attrition" instead of being handed to scipy.
    """
    for label, kept, dropped in (("A", kept_a, dropped_a), ("B", kept_b, dropped_b)):
        if kept < 0 or dropped < 0:
            raise ValueError(f"Group {label}: counts cannot be negative.")
        if kept + dropped == 0:
            raise ValueError(f"Group {label}: total sample size is zero.")

    dropped_share_a = dropped_a / (kept_a + dropped_a)
    dropped_share_b = dropped_b / (kept_b + dropped_b)

    if dropped_a == 0 and dropped_b == 0:
        return {
            "dropped_share_a": dropped_share_a,
            "dropped_share_b": dropped_share_b,
            "p_value": 1.0,
            "has_differential_attrition": False,
        }

    table = [[kept_a, dropped_a], [kept_b, dropped_b]]
    _, p_value, _, _ = chi2_contingency(table)

    return {
        "dropped_share_a": dropped_share_a,
        "dropped_share_b": dropped_share_b,
        "p_value": float(p_value),
        "has_differential_attrition": p_value < SRM_P_VALUE_THRESHOLD,
    }


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
) -> FrequentistGuardrails:
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
) -> ChiSquaredResult:
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
    effect_size_method: EffectSizeMethod = "pooled",
) -> WelchTTestResult:
    """Run Welch's t-test for a continuous outcome.

    The test never assumes equal variances. The effect-size denominator is
    configurable:

    - ``"pooled"`` (default): classic Cohen's d with the pooled standard
      deviation. The most comparable to published benchmarks, but it assumes
      roughly equal variances.
    - ``"averaged"``: Cohen's d using ``sqrt((var_a + var_b) / 2)``. This is the
      variance structure Welch's test itself uses, so it is the consistent
      choice when the group variances differ sharply.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Each group must have at least 2 observations for Welch's t-test.")
    if group_a.isna().any() or group_b.isna().any():
        raise ValueError("Continuous outcome groups cannot contain missing values.")

    statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)

    n_a = len(group_a)
    n_b = len(group_b)
    var_a = float(group_a.std() ** 2)
    var_b = float(group_b.std() ** 2)

    if effect_size_method == "averaged":
        denominator = np.sqrt((var_a + var_b) / 2)
        effect_size_label = "Cohen's d (unequal var)"
    else:
        denominator = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        effect_size_label = "Cohen's d"

    if denominator == 0:
        raise ValueError("Effect size is undefined when both groups have zero variance.")

    cohens_d = (group_b.mean() - group_a.mean()) / denominator

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(cohens_d),
        "effect_size_label": effect_size_label,
        "test_name": "Welch's T-Test",
    }


def _two_sided_critical_value(alpha: float) -> float:
    """Two-sided normal critical value, exact at the module's pinned default.

    Returns the pinned ``Z_ALPHA`` constant when ``alpha`` is left at its
    default, so no existing caller's interval moves by even a rounding unit.
    A non-default ``alpha`` is computed from scipy instead.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    return Z_ALPHA if alpha == ALPHA else float(norm.ppf(1 - alpha / 2))


def confidence_interval_binary(
    cr_a: float,
    cr_b: float,
    n_a: int,
    n_b: int,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Calculate a confidence interval on relative lift for binary outcomes.

    Uses a first-order delta-method approximation: the interval is built on the
    absolute difference and then divided by the control rate, treating the
    denominator as fixed. That is accurate when the control rate is estimated
    far more precisely than the lift, which holds for the usual A/B sample
    sizes. For very small control groups, prefer a bootstrap interval.

    The variant endpoint (``cr_b`` plus or minus the margin) is clamped to
    ``[0, 1]`` before it is turned into a relative change, because it stands
    in for a rate and a rate cannot go negative or above 100%. Skipping that
    clamp lets a wide margin push the lower bound past -100% relative change,
    which reads as "fell by more than everything there was to fall", a bound
    the data never actually supports.

    ``alpha`` defaults to the module's fixed two-sided 95% critical value
    (``Z_ALPHA``), so every existing caller's interval is unchanged to the
    unit. Passing a non-default alpha switches to a scipy-computed z-value.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    if cr_a <= 0:
        raise ValueError("Control conversion rate must be greater than zero.")
    if not (0 <= cr_a <= 1 and 0 <= cr_b <= 1):
        raise ValueError("Conversion rates must be between 0 and 1.")

    se_a = np.sqrt(cr_a * (1 - cr_a) / n_a)
    se_b = np.sqrt(cr_b * (1 - cr_b) / n_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)

    margin = _two_sided_critical_value(alpha) * se_diff
    variant_lower = min(max(cr_b - margin, 0.0), 1.0)
    variant_upper = min(max(cr_b + margin, 0.0), 1.0)
    ci_lower = (variant_lower - cr_a) / cr_a
    ci_upper = (variant_upper - cr_a) / cr_a
    return ci_lower, ci_upper


def confidence_interval_continuous(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Calculate a confidence interval on relative lift for continuous outcomes.

    ``alpha`` defaults to the module's fixed two-sided 95% critical value
    (``Z_ALPHA``), so every existing caller's interval is unchanged to the
    unit. Passing a non-default alpha switches to a scipy-computed z-value.
    """
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

    margin = _two_sided_critical_value(alpha) * se_diff
    ci_lower = ((mean_b - margin) - mean_a) / mean_a
    ci_upper = ((mean_b + margin) - mean_a) / mean_a
    return ci_lower, ci_upper


def bootstrap_ci_relative_lift_continuous(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI on relative lift for continuous outcomes.

    Resamples each group with replacement and reads the empirical percentiles of
    the relative lift. Unlike :func:`confidence_interval_continuous`, it assumes
    neither normality nor a fixed denominator, so it is the more honest interval
    for small or heavily skewed samples (revenue, session length). The seed makes
    the interval reproducible across reruns.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Each group must have at least 2 observations to bootstrap a CI.")
    if group_a.isna().any() or group_b.isna().any():
        raise ValueError("Continuous outcome groups cannot contain missing values.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")
    if group_a.mean() == 0:
        raise ValueError("Control mean must be non-zero to calculate relative lift.")

    rng = np.random.default_rng(seed)
    a = group_a.to_numpy(dtype=float)
    b = group_b.to_numpy(dtype=float)

    means_a = a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(axis=1)
    means_b = b[rng.integers(0, len(b), size=(n_resamples, len(b)))].mean(axis=1)

    # Drop resamples where the control mean is zero (relative lift undefined).
    valid = means_a != 0
    if not valid.any():
        raise ValueError("Every bootstrap resample produced a zero control mean.")
    lifts = (means_b[valid] - means_a[valid]) / means_a[valid]

    lower = float(np.percentile(lifts, 100 * alpha / 2))
    upper = float(np.percentile(lifts, 100 * (1 - alpha / 2)))
    return lower, upper


def calculate_sample_size(
    baseline: float,
    mde: float,
    daily_traffic: int,
    split_ratio: float = 0.5,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    rho: float = 0.0,
    cluster_design_effect: float = 1.0,
) -> SampleSizeResult:
    """Estimate total sample size and duration for an A/B test.

    This is the one function that owns binary sample sizing: Signal 01, the
    sanity checks, and the locked-plan sizing in ``render_plan_lock`` all call
    this instead of each carrying its own variance formula, so a design and
    its later verification are always sized the same way.

    This formula is paired with :func:`calculate_reverse_mde`, which is its
    algebraic inverse for the default 50/50 split case.

    ``alpha`` and ``power`` default to the module's fixed critical values
    (``Z_ALPHA``/``Z_BETA``), so every existing caller's numbers are unchanged
    to the unit. Passing a non-default alpha or power switches to a
    scipy-computed z-value instead. ``rho`` applies the CUPED variance
    reduction described in :func:`stats.power.cuped_variance_retained` and
    defaults to 0.0, a no-op. ``cluster_design_effect``, from
    :func:`stats.power.design_effect`, applies the same group-randomisation
    correction Signal 02's sizing block uses, so a plan locked from that
    block is sized under the same clustering assumption it was designed
    under, and defaults to 1.0, a no-op.
    """
    if not (0.001 <= baseline <= 0.999):
        raise ValueError("Baseline conversion must be between 0.1% and 99.9%.")
    if mde <= 0:
        raise ValueError("MDE must be greater than zero.")
    if daily_traffic <= 0:
        raise ValueError("Daily traffic must be greater than zero.")
    if not (0 < split_ratio < 1):
        raise ValueError("Split ratio must be between 0 and 1.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1.")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")
    if cluster_design_effect < 1.0:
        raise ValueError("Cluster design effect must be at least 1.0.")

    p2 = baseline * (1 + mde)
    if p2 >= 1:
        raise ValueError("Target lift pushes the projected conversion rate above 100%.")

    delta = p2 - baseline
    split_factor = (1 / split_ratio) + (1 / (1 - split_ratio))
    pooled_var = (baseline * (1 - baseline) + p2 * (1 - p2)) / 2
    variance_retained = 1 - rho**2
    if alpha == ALPHA and power == DEFAULT_POWER:
        z_sum = Z_ALPHA + Z_BETA
    else:
        z_sum = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    n_total = (
        (z_sum**2) * pooled_var * variance_retained * split_factor / (delta**2) * cluster_design_effect
    )
    days = np.ceil(n_total / daily_traffic)

    split_penalty = 0
    if split_ratio != 0.5:
        # Same multiplier as stats.power.allocation_cost, expressed as a percentage
        # of extra sample the uneven split costs relative to 50/50.
        split_penalty = int((allocation_cost(split_ratio) - 1) * 100)

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
) -> ReverseMDEResult:
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
