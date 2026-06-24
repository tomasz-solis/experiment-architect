"""Unit tests for deterministic experiment sanity checks."""

from __future__ import annotations

import pytest

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
