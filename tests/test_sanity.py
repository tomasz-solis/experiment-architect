"""Unit tests for deterministic experiment sanity checks."""

from __future__ import annotations

import re

import pytest

from stats.frequentist import calculate_sample_size
from stats.sanity import (
    check_baseline_stability,
    check_mde_plausibility,
    check_traffic_vs_mde,
    run_all_checks,
    severity_rank,
)


@pytest.mark.parametrize(
    ("daily_traffic", "weeks", "expected_status"),
    [
        (2000, 4, "ok"),
        (1400, 4, "caution"),
        (500, 2, "fail"),
    ],
)
def test_traffic_vs_mde_statuses(
    daily_traffic: int,
    weeks: int,
    expected_status: str,
) -> None:
    """Traffic feasibility should reflect the ratio of available to required users."""
    name, status, reason = check_traffic_vs_mde(
        baseline=0.10,
        mde=0.10,
        daily_traffic=daily_traffic,
        weeks=weeks,
    )

    assert name == "Traffic vs MDE"
    assert status == expected_status
    assert reason


@pytest.mark.parametrize(
    ("split_ratio", "cluster_design_effect"),
    [
        (0.5, 1.0),
        (0.2, 1.0),
        (0.5, 1.45),
    ],
)
def test_traffic_vs_mde_requirement_matches_calculate_sample_size(
    split_ratio: float,
    cluster_design_effect: float,
) -> None:
    """The required-n the check reports must be exactly calculate_sample_size's n_total,
    for the same split and design effect, not a hardcoded 50/50 no-clustering read."""
    baseline = 0.10
    mde = 0.20
    daily_traffic = 100_000
    weeks = 8

    expected_n_total = calculate_sample_size(
        baseline=baseline,
        mde=mde,
        daily_traffic=daily_traffic,
        split_ratio=split_ratio,
        cluster_design_effect=cluster_design_effect,
    )["n_total"]

    _, status, reason = check_traffic_vs_mde(
        baseline=baseline,
        mde=mde,
        daily_traffic=daily_traffic,
        weeks=weeks,
        split_ratio=split_ratio,
        cluster_design_effect=cluster_design_effect,
    )

    assert status == "ok"
    match = re.search(r"([\d,]+) needed", reason)
    assert match is not None
    assert int(match.group(1).replace(",", "")) == expected_n_total


def test_traffic_vs_mde_defaults_reproduce_previous_behaviour() -> None:
    """Callers that don't pass the new parameters must see the same status and
    required-n as before this fix (a hardcoded 50/50, no-clustering read),
    so existing integrations are unaffected."""
    baseline = 0.10
    mde = 0.10
    daily_traffic = 1400
    weeks = 4

    expected_n_total = calculate_sample_size(
        baseline=baseline,
        mde=mde,
        daily_traffic=daily_traffic,
        split_ratio=0.5,
    )["n_total"]

    _, status, reason = check_traffic_vs_mde(
        baseline=baseline,
        mde=mde,
        daily_traffic=daily_traffic,
        weeks=weeks,
    )

    assert status == "caution"
    total_n = daily_traffic * weeks * 7
    assert f"{total_n / expected_n_total:.1f}" in reason


def test_traffic_vs_mde_flags_underpowered_uneven_split_as_fail() -> None:
    """Reproduces the reported bug: a 12% baseline, 5% MDE, 5000 daily visitors,
    4 weeks, and an 80/20 split needs 146,813 users against a 140,000 budget.
    Scored at the real 80/20 split this is underpowered and must read 'fail',
    not the 'caution' a hardcoded 50/50 comparison used to report."""
    name, status, reason = check_traffic_vs_mde(
        baseline=0.12,
        mde=0.05,
        daily_traffic=5000,
        weeks=4,
        split_ratio=0.2,
    )

    assert name == "Traffic vs MDE"
    assert status == "fail"
    assert "146,813" in reason


@pytest.mark.parametrize(
    ("mde", "expected_status"),
    [
        (0.60, "fail"),
        (0.25, "caution"),
        (0.005, "caution"),
        (0.10, "ok"),
    ],
)
def test_mde_plausibility_statuses(mde: float, expected_status: str) -> None:
    """MDE plausibility should flag tiny, aggressive, and impossible targets."""
    name, status, reason = check_mde_plausibility(mde)

    assert name == "MDE plausibility"
    assert status == expected_status
    assert reason


@pytest.mark.parametrize(
    ("baseline", "expected_status"),
    [
        (0.005, "fail"),
        (0.02, "caution"),
        (0.98, "caution"),
        (0.10, "ok"),
    ],
)
def test_baseline_stability_statuses(baseline: float, expected_status: str) -> None:
    """Baseline checks should flag boundary conversion rates."""
    name, status, _ = check_baseline_stability(baseline)

    assert name == "Baseline stability"
    assert status == expected_status


def test_run_all_checks_preserves_rule_order() -> None:
    """The UI expects all sanity checks in a stable, readable order."""
    checks = run_all_checks(baseline=0.10, mde=0.10, daily_traffic=2000, weeks=4)

    assert [name for name, _, _ in checks] == [
        "Traffic vs MDE",
        "MDE plausibility",
        "Baseline stability",
    ]


def test_severity_rank_orders_statuses() -> None:
    """Severity must increase from ok to caution to fail so the UI can pick the worst."""
    assert severity_rank("ok") < severity_rank("caution") < severity_rank("fail")


def test_severity_rank_selects_worst_check() -> None:
    """max() keyed on severity_rank should surface the most severe finding."""
    checks = run_all_checks(baseline=0.005, mde=0.60, daily_traffic=200, weeks=1)
    _, worst_status, _ = max(checks, key=lambda item: severity_rank(item[1]))

    assert worst_status == "fail"
