"""Unit tests for frequentist statistical functions."""

import pytest
import numpy as np
import pandas as pd
from stats.frequentist import (
    check_srm,
    calculate_lift,
    chi_squared_test,
    welch_t_test,
    confidence_interval_binary,
    confidence_interval_continuous,
    is_significant,
    calculate_sample_size
)


class TestSRM:
    """Tests for Sample Ratio Mismatch detection."""

    def test_no_srm_equal_split(self):
        has_mismatch, ratio = check_srm(1000, 1000)
        assert not has_mismatch
        assert ratio == 0.5

    def test_srm_detected(self):
        has_mismatch, ratio = check_srm(1200, 800)
        assert has_mismatch
        assert ratio == 0.6

    def test_srm_threshold(self):
        # 54% is within default 5% threshold
        has_mismatch, _ = check_srm(1080, 920)
        assert not has_mismatch

        # 56% exceeds threshold
        has_mismatch, _ = check_srm(1120, 880)
        assert has_mismatch


class TestLift:
    """Tests for lift calculation."""

    def test_positive_lift(self):
        lift = calculate_lift(10.0, 11.0)
        assert lift == pytest.approx(0.1)

    def test_negative_lift(self):
        lift = calculate_lift(10.0, 9.0)
        assert lift == pytest.approx(-0.1)

    def test_zero_lift(self):
        lift = calculate_lift(10.0, 10.0)
        assert lift == pytest.approx(0.0)


class TestChiSquared:
    """Tests for Chi-Squared test."""

    def test_significant_difference(self):
        # Larger difference for clear significance
        result = chi_squared_test(100, 900, 150, 850)
        assert result['p_value'] < 0.05
        assert result['effect_size'] > 0
        assert result['test_name'] == "Chi-Squared Test"

    def test_no_difference(self):
        # Identical groups
        result = chi_squared_test(100, 900, 100, 900)
        assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        assert result['effect_size'] == pytest.approx(0.0, abs=0.001)


class TestWelchTTest:
    """Tests for Welch's T-Test."""

    def test_significant_difference(self):
        group_a = pd.Series([1, 2, 3, 4, 5] * 100)
        group_b = pd.Series([3, 4, 5, 6, 7] * 100)
        result = welch_t_test(group_a, group_b)
        assert result['p_value'] < 0.001
        assert abs(result['effect_size']) > 1
        assert result['test_name'] == "Welch's T-Test"

    def test_no_difference(self):
        group_a = pd.Series([1, 2, 3, 4, 5] * 100)
        group_b = pd.Series([1, 2, 3, 4, 5] * 100)
        result = welch_t_test(group_a, group_b)
        assert result['p_value'] > 0.9
        assert abs(result['effect_size']) < 0.01


class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_binary_ci_positive_lift(self):
        ci_lower, ci_upper = confidence_interval_binary(0.10, 0.11, 1000, 1000)
        assert ci_lower < 0.1
        assert ci_upper > 0.1
        assert ci_lower < ci_upper

    def test_continuous_ci(self):
        # Add realistic variance
        np.random.seed(42)
        group_a = pd.Series(np.random.normal(10.0, 2.0, 1000))
        group_b = pd.Series(np.random.normal(11.0, 2.0, 1000))
        ci_lower, ci_upper = confidence_interval_continuous(group_a, group_b)
        # CI should contain the true lift (~10%)
        assert ci_lower < 0.15
        assert ci_upper > 0.05
        assert ci_lower < ci_upper


class TestSignificance:
    """Tests for significance checking."""

    def test_significant(self):
        assert is_significant(0.01)
        assert is_significant(0.04)

    def test_not_significant(self):
        assert not is_significant(0.06)
        assert not is_significant(0.5)


class TestSampleSize:
    """Tests for sample size calculation."""

    def test_reasonable_sample_size(self):
        result = calculate_sample_size(
            baseline=0.10,
            mde=0.10,
            daily_traffic=5000
        )
        assert result['n_total'] > 0
        assert result['days'] > 0
        assert result['split_penalty'] == 0  # 50/50 split

    def test_unequal_split_penalty(self):
        result = calculate_sample_size(
            baseline=0.10,
            mde=0.10,
            daily_traffic=5000,
            split_ratio=0.8
        )
        assert result['split_penalty'] > 0

    def test_larger_mde_needs_less_traffic(self):
        small_mde = calculate_sample_size(0.10, 0.05, 5000)
        large_mde = calculate_sample_size(0.10, 0.20, 5000)
        assert small_mde['days'] > large_mde['days']
