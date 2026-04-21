"""Deterministic sanity checks for A/B test feasibility.

Each rule returns (name, status, reason) where status is one of
'ok', 'caution', or 'fail'. The UI renders ok as success, caution
as warning, and fail as error.

These replace an LLM-based sanity check. Rules are reproducible,
testable, and don't ask the model to make statistical judgments.
"""

from typing import List, Literal, Tuple

import numpy as np

from config import Z_ALPHA, Z_BETA

Status = Literal["ok", "caution", "fail"]
CheckResult = Tuple[str, Status, str]


def check_traffic_vs_mde(
    baseline: float,
    mde: float,
    daily_traffic: int,
    weeks: int,
) -> CheckResult:
    """Check whether the traffic budget is sufficient to detect the target MDE.

    Computes required sample size from the same formula as calculate_sample_size
    and compares it to the available budget.

    Args:
        baseline: Current conversion rate (e.g. 0.10 for 10%).
        mde: Target relative lift (e.g. 0.10 for 10% relative improvement).
        daily_traffic: Total daily visitors across both groups.
        weeks: Planned experiment duration.

    Returns:
        CheckResult tuple (name, status, reason).
    """
    total_n = daily_traffic * weeks * 7
    p2 = baseline * (1 + mde)
    delta = p2 - baseline
    pooled_var = (baseline * (1 - baseline) + p2 * (1 - p2)) / 2
    z_score = (Z_ALPHA + Z_BETA) ** 2
    required_n = z_score * pooled_var * 4 / (delta ** 2)
    ratio = total_n / required_n

    if ratio >= 1.5:
        return (
            "Traffic vs MDE",
            "ok",
            f"Budget is {ratio:.1f}× the required sample ({int(required_n):,} needed, {int(total_n):,} available).",
        )
    if ratio >= 1.0:
        return (
            "Traffic vs MDE",
            "caution",
            f"Budget covers the minimum ({ratio:.1f}×) but leaves no buffer. "
            "Any early peeking or ramp-up delays could push you below significance.",
        )
    return (
        "Traffic vs MDE",
        "fail",
        f"Budget covers only {ratio:.0%} of the required sample. "
        f"Need {int(required_n):,} visitors but only {int(total_n):,} available. "
        "Increase traffic, extend duration, or raise the target MDE.",
    )


def check_mde_plausibility(mde: float) -> CheckResult:
    """Flag MDEs outside typical observed ranges for product experiments.

    This is a heuristic, not a hard rule. The thresholds are calibrated
    against typical conversion-rate experiments in B2C/B2B products.

    Args:
        mde: Target relative lift (e.g. 0.10 for 10% relative improvement).

    Returns:
        CheckResult tuple (name, status, reason).
    """
    if mde > 0.5:
        return (
            "MDE plausibility",
            "fail",
            f"Target lift of {mde:.0%} is larger than nearly all observed product effects. "
            "Verify this is a relative lift, not an absolute change.",
        )
    if mde > 0.2:
        return (
            "MDE plausibility",
            "caution",
            f"Target lift of {mde:.0%} is aggressive. Validate against historical "
            "experiment data before committing to this target.",
        )
    if mde < 0.01:
        return (
            "MDE plausibility",
            "caution",
            f"Target lift of {mde:.1%} is very small. "
            "The required sample size will be large — confirm the business value justifies the wait.",
        )
    return ("MDE plausibility", "ok", f"Target lift of {mde:.1%} is within typical range.")


def check_baseline_stability(baseline: float) -> CheckResult:
    """Flag baselines where the normal approximation breaks down.

    The two-proportion z-test assumes np >= 5 and n(1-p) >= 5 per cell.
    Very low or very high baselines need much larger samples than the
    standard formula predicts, because the normal approximation is poor.

    Args:
        baseline: Current conversion rate (e.g. 0.10 for 10%).

    Returns:
        CheckResult tuple (name, status, reason).
    """
    if baseline < 0.01 or baseline > 0.99:
        return (
            "Baseline stability",
            "fail",
            f"Baseline of {baseline:.1%} is too extreme for the normal approximation. "
            "Use Fisher's exact test and treat sample size estimates as lower bounds.",
        )
    if baseline < 0.03 or baseline > 0.97:
        return (
            "Baseline stability",
            "caution",
            f"Baseline of {baseline:.1%} is near the boundary of the normal approximation. "
            "The sample size estimate may understate what you actually need.",
        )
    return ("Baseline stability", "ok", "")


def run_all_checks(
    baseline: float,
    mde: float,
    daily_traffic: int,
    weeks: int,
) -> List[CheckResult]:
    """Run all sanity checks and return results.

    Args:
        baseline: Current conversion rate.
        mde: Target relative lift.
        daily_traffic: Total daily visitors.
        weeks: Planned experiment duration in weeks.

    Returns:
        List of CheckResult tuples, one per rule.
    """
    return [
        check_traffic_vs_mde(baseline, mde, daily_traffic, weeks),
        check_mde_plausibility(mde),
        check_baseline_stability(baseline),
    ]
