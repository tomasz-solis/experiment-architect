"""Unit tests for frequentist statistical helpers."""

import numpy as np
import pandas as pd
import pytest

from stats.frequentist import (
    bonferroni_adjusted_alpha,
    build_frequentist_guardrails,
    calculate_lift,
    calculate_reverse_mde,
    calculate_sample_size,
    check_srm,
    chi_squared_test,
    confidence_interval_binary,
    confidence_interval_continuous,
    is_significant,
    welch_t_test,
)


class TestSRM:
    """Tests for sample ratio mismatch detection."""

    def test_no_srm_equal_split(self) -> None:
        has_mismatch, ratio = check_srm(1000, 1000)
        assert not has_mismatch
        assert ratio == 0.5

    def test_srm_detected(self) -> None:
        has_mismatch, ratio = check_srm(1200, 800)
        assert has_mismatch
        assert ratio == 0.6

    def test_srm_threshold(self) -> None:
        has_mismatch, _ = check_srm(1080, 920)
        assert not has_mismatch

        has_mismatch, _ = check_srm(1120, 880)
        assert has_mismatch


class TestLift:
    """Tests for lift calculation."""

    def test_positive_lift(self) -> None:
        assert calculate_lift(10.0, 11.0) == pytest.approx(0.1)

    def test_negative_lift(self) -> None:
        assert calculate_lift(10.0, 9.0) == pytest.approx(-0.1)

    def test_zero_lift(self) -> None:
        assert calculate_lift(10.0, 10.0) == pytest.approx(0.0)


class TestChiSquared:
    """Tests for the chi-squared test wrapper."""

    def test_significant_difference(self) -> None:
        result = chi_squared_test(100, 900, 150, 850)
        assert result["p_value"] < 0.05
        assert result["effect_size"] > 0
        assert result["test_name"] == "Chi-Squared Test"
        assert result["chi_square_valid"]

    def test_no_difference(self) -> None:
        result = chi_squared_test(100, 900, 100, 900)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert result["effect_size"] == pytest.approx(0.0, abs=0.001)

    def test_low_expected_cell_count_is_flagged(self) -> None:
        result = chi_squared_test(1, 19, 8, 12)
        assert not result["chi_square_valid"]
        assert result["min_expected_count"] < 5


class TestWelchTTest:
    """Tests for the Welch's t-test wrapper."""

    def test_significant_difference(self) -> None:
        group_a = pd.Series([1, 2, 3, 4, 5] * 100)
        group_b = pd.Series([3, 4, 5, 6, 7] * 100)
        result = welch_t_test(group_a, group_b)
        assert result["p_value"] < 0.001
        assert abs(result["effect_size"]) > 1
        assert result["test_name"] == "Welch's T-Test"

    def test_no_difference(self) -> None:
        group_a = pd.Series([1, 2, 3, 4, 5] * 100)
        group_b = pd.Series([1, 2, 3, 4, 5] * 100)
        result = welch_t_test(group_a, group_b)
        assert result["p_value"] > 0.9
        assert abs(result["effect_size"]) < 0.01


class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_binary_ci_positive_lift(self) -> None:
        ci_lower, ci_upper = confidence_interval_binary(0.10, 0.11, 1000, 1000)
        assert ci_lower < 0.1
        assert ci_upper > 0.1
        assert ci_lower < ci_upper

    def test_continuous_ci(self) -> None:
        rng = np.random.default_rng(42)
        group_a = pd.Series(rng.normal(10.0, 2.0, 1000))
        group_b = pd.Series(rng.normal(11.0, 2.0, 1000))
        ci_lower, ci_upper = confidence_interval_continuous(group_a, group_b)
        assert ci_lower < 0.15
        assert ci_upper > 0.05
        assert ci_lower < ci_upper


class TestSignificance:
    """Tests for p-value significance helpers."""

    def test_significant(self) -> None:
        assert is_significant(0.01)
        assert is_significant(0.04)
        assert is_significant(0.01, alpha=0.01) is False

    def test_not_significant(self) -> None:
        assert not is_significant(0.06)
        assert not is_significant(0.5)


class TestGuardrails:
    """Tests for frequentist inference guardrails."""

    def test_bonferroni_adjustment(self) -> None:
        assert bonferroni_adjusted_alpha(alpha=0.05, n_comparisons=5) == pytest.approx(0.01)

    def test_guardrail_summary_marks_multiple_metrics_and_peeking(self) -> None:
        result = build_frequentist_guardrails(n_comparisons=3, peeked_early=True)
        assert result["alpha_adjusted"]
        assert result["peeked_early"]
        assert result["adjusted_alpha"] == pytest.approx(0.05 / 3)


class TestSampleSize:
    """Tests for sample size calculations."""

    def test_reasonable_sample_size(self) -> None:
        result = calculate_sample_size(baseline=0.10, mde=0.10, daily_traffic=5000)
        assert result["n_total"] > 0
        assert result["days"] > 0
        assert result["split_penalty"] == 0

    def test_unequal_split_penalty(self) -> None:
        result = calculate_sample_size(
            baseline=0.10,
            mde=0.10,
            daily_traffic=5000,
            split_ratio=0.8,
        )
        assert result["split_penalty"] > 0

    def test_larger_mde_needs_less_traffic(self) -> None:
        small_mde = calculate_sample_size(0.10, 0.05, 5000)
        large_mde = calculate_sample_size(0.10, 0.20, 5000)
        assert small_mde["days"] > large_mde["days"]


class TestReverseMDE:
    """Tests for reverse MDE calculations."""

    def test_round_trips_with_sample_size(self) -> None:
        baseline = 0.10
        mde = 0.10
        daily_traffic = 5000

        sample_size = calculate_sample_size(baseline, mde, daily_traffic)
        weeks = int(np.ceil(sample_size["days"] / 7))
        reverse = calculate_reverse_mde(baseline, daily_traffic, weeks)

        assert "mde" in reverse
        assert abs(reverse["mde"] - mde) < 0.02

    def test_respects_non_equal_split_ratio(self) -> None:
        baseline = 0.10
        mde = 0.10
        daily_traffic = 5000
        split_ratio = 0.7

        sample_size = calculate_sample_size(
            baseline,
            mde,
            daily_traffic,
            split_ratio=split_ratio,
        )
        weeks = int(np.ceil(sample_size["days"] / 7))
        reverse = calculate_reverse_mde(
            baseline,
            daily_traffic,
            weeks,
            split_ratio=split_ratio,
        )

        assert "mde" in reverse
        assert reverse["mde"] > 0

    def test_insufficient_traffic_returns_error(self) -> None:
        result = calculate_reverse_mde(0.10, 5, 1)
        assert "error" in result
