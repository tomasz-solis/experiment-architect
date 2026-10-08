"""Deterministic sanity checks for A/B test feasibility.

Each rule returns (name, status, reason) where status is one of
'ok', 'caution', or 'fail'. The UI renders ok as success, caution
as warning, and fail as error.

These replace an LLM-based sanity check. Rules are reproducible,
testable, and don't ask the model to make statistical judgments.
"""

from typing import Literal

from config import ALPHA, DEFAULT_POWER
from stats.frequentist import calculate_sample_size

Status = Literal["ok", "caution", "fail"]
CheckResult = tuple[str, Status, str]

_SEVERITY_RANK: dict[Status, int] = {"ok": 0, "caution": 1, "fail": 2}


def severity_rank(status: Status) -> int:
    """Map a check status to an ordered severity rank (higher is worse)."""
    return _SEVERITY_RANK[status]


def check_traffic_vs_mde(
    baseline: float,
    mde: float,
    daily_traffic: int,
    weeks: int,
    split_ratio: float = 0.5,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    rho: float = 0.0,
    cluster_design_effect: float = 1.0,
) -> CheckResult:
    """Check whether the traffic budget is sufficient to detect the target MDE.

    Delegates to calculate_sample_size for the required-n calculation, and is
    passed the same split, alpha, power, CUPED rho, and cluster design effect
    the caller actually configured. Sharing both the function and its inputs
    is what keeps this check and the design tab from diverging; passing a
    different split or a default design effect while the design tab uses the
    real ones would silently reintroduce the divergence.
    """
    size = calculate_sample_size(
        baseline=baseline,
        mde=mde,
        daily_traffic=daily_traffic,
        split_ratio=split_ratio,
        alpha=alpha,
        power=power,
        rho=rho,
        cluster_design_effect=cluster_design_effect,
    )
    required_n = size["n_total"]
    total_n = daily_traffic * weeks * 7
    ratio = total_n / required_n

    if ratio >= 1.5:
        return (
            "Traffic vs MDE",
            "ok",
            f"Budget is {ratio:.1f}× the required sample ({required_n:,} needed, {total_n:,} available).",
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
        f"Need {required_n:,} visitors but only {total_n:,} available. "
        "Increase traffic, extend duration, or raise the target MDE.",
    )


def check_mde_plausibility(mde: float) -> CheckResult:
    """Flag MDEs outside typical observed ranges for product experiments.

    This is a heuristic, not a hard rule. The thresholds are calibrated
    against typical conversion-rate experiments in B2C/B2B products.
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
            "The required sample size will be large - confirm the business value justifies the wait.",
        )
    return ("MDE plausibility", "ok", f"Target lift of {mde:.1%} is within typical range.")


def check_baseline_stability(baseline: float) -> CheckResult:
    """Flag baselines where the normal approximation breaks down.

    The two-proportion z-test assumes np >= 5 and n(1-p) >= 5 per cell.
    Very low or very high baselines need much larger samples than the
    standard formula predicts, because the normal approximation is poor.
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
    split_ratio: float = 0.5,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    rho: float = 0.0,
    cluster_design_effect: float = 1.0,
) -> list[CheckResult]:
    """Run all sanity checks and return their results in order."""
    return [
        check_traffic_vs_mde(
            baseline,
            mde,
            daily_traffic,
            weeks,
            split_ratio=split_ratio,
            alpha=alpha,
            power=power,
            rho=rho,
            cluster_design_effect=cluster_design_effect,
        ),
        check_mde_plausibility(mde),
        check_baseline_stability(baseline),
    ]
