"""Unit tests for frequentist statistical helpers."""

import numpy as np
import pandas as pd
import pytest

from stats.frequentist import (
    bonferroni_adjusted_alpha,
    bootstrap_ci_relative_lift_continuous,
    build_frequentist_guardrails,
    calculate_lift,
    calculate_reverse_mde,
    calculate_sample_size,
    check_differential_attrition,
    check_srm,
    chi_squared_test,
    confidence_interval_binary,
    confidence_interval_continuous,
    welch_t_test,
)


class TestSRM:
    """Tests for sample-ratio-mismatch detection against the intended split.

    ``check_srm`` tests the observed split against the split the experiment
    was actually meant to run, via a chi-squared goodness-of-fit test rather
    than a fixed percentage-point gap.
    """

    def test_no_srm_equal_split(self) -> None:
        result = check_srm(1000, 1000)
        assert not result["has_mismatch"]
        assert result["observed_share"] == pytest.approx(0.5)
        assert result["expected_share"] == 0.5

    def test_deliberate_uneven_split_is_not_a_mismatch(self) -> None:
        """A 70/30 split tested against its own intended share passes."""
        result = check_srm(3000, 7000, expected_share_b=0.7)
        assert not result["has_mismatch"]
        assert result["observed_share"] == pytest.approx(0.7)

    def test_same_split_fails_against_the_wrong_expectation(self) -> None:
        """The same 70/30 split fails once checked against a 50/50 expectation."""
        result = check_srm(3000, 7000, expected_share_b=0.5)
        assert result["has_mismatch"]

    def test_large_imbalance_is_detected(self) -> None:
        result = check_srm(1200, 800)
        assert result["has_mismatch"]

    def test_proper_test_flags_what_the_old_fixed_threshold_missed(self) -> None:
        """1080 vs 920 passed the old +/-5-point threshold; a proper test rejects it."""
        result = check_srm(1080, 920)
        assert result["has_mismatch"]
        assert result["p_value"] == pytest.approx(0.00035, abs=0.0001)

    def test_small_samples_do_not_fire_on_noise(self) -> None:
        result = check_srm(12, 8)
        assert not result["has_mismatch"]

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError):
            check_srm(-1, 100)

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError):
            check_srm(0, 0)

    def test_expected_share_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            check_srm(100, 100, expected_share_b=1.0)


class TestDifferentialAttrition:
    """Tests for the kept/dropped chi-squared check between arms."""

    def test_no_attrition_is_not_flagged(self) -> None:
        result = check_differential_attrition(kept_a=1000, dropped_a=0, kept_b=1000, dropped_b=0)
        assert not result["has_differential_attrition"]
        assert result["p_value"] == 1.0

    def test_even_attrition_is_not_flagged(self) -> None:
        result = check_differential_attrition(kept_a=950, dropped_a=50, kept_b=950, dropped_b=50)
        assert not result["has_differential_attrition"]

    def test_uneven_attrition_is_flagged(self) -> None:
        result = check_differential_attrition(kept_a=990, dropped_a=10, kept_b=900, dropped_b=100)
        assert result["has_differential_attrition"]
        assert result["dropped_share_a"] == pytest.approx(0.01)
        assert result["dropped_share_b"] == pytest.approx(0.10)

    def test_one_arm_with_zero_dropped_still_computes(self) -> None:
        """Only one arm losing rows is a valid, non-degenerate table."""
        result = check_differential_attrition(kept_a=1000, dropped_a=0, kept_b=900, dropped_b=100)
        assert result["has_differential_attrition"]

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError):
            check_differential_attrition(kept_a=-1, dropped_a=0, kept_b=100, dropped_b=0)

    def test_empty_arm_raises(self) -> None:
        with pytest.raises(ValueError):
            check_differential_attrition(kept_a=0, dropped_a=0, kept_b=100, dropped_b=0)


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

    def test_averaged_effect_size_is_labelled_and_differs_under_unequal_var_and_n(self) -> None:
        rng = np.random.default_rng(0)
        # Pooled SD weights by (n-1), the averaged form weights equally, so they
        # diverge only when BOTH the variances and the sample sizes differ.
        group_a = pd.Series(rng.normal(10.0, 1.0, 200))
        group_b = pd.Series(rng.normal(11.0, 5.0, 600))

        pooled = welch_t_test(group_a, group_b)
        averaged = welch_t_test(group_a, group_b, effect_size_method="averaged")

        assert pooled["effect_size_label"] == "Cohen's d"
        assert averaged["effect_size_label"] == "Cohen's d (unequal var)"
        # The p-value is identical (same Welch test); only the effect size changes.
        assert averaged["p_value"] == pytest.approx(pooled["p_value"])
        assert averaged["effect_size"] != pytest.approx(pooled["effect_size"])

    def test_effect_size_methods_agree_under_equal_variance(self) -> None:
        rng = np.random.default_rng(1)
        # Equal variances => pooled and averaged denominators coincide exactly,
        # even with unequal group sizes.
        group_a = pd.Series(rng.normal(10.0, 2.0, 300))
        group_b = pd.Series(rng.normal(11.0, 2.0, 900))

        pooled = welch_t_test(group_a, group_b)["effect_size"]
        averaged = welch_t_test(group_a, group_b, effect_size_method="averaged")["effect_size"]
        assert averaged == pytest.approx(pooled, rel=0.05)


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

    def test_binary_ci_default_alpha_matches_the_pinned_z_alpha(self) -> None:
        """Item 2: a non-default alpha must actually change the interval width,
        but the default path must not move by even a unit from the pinned
        Z_ALPHA critical value."""
        default = confidence_interval_binary(0.10, 0.11, 1000, 1000)
        explicit_default = confidence_interval_binary(0.10, 0.11, 1000, 1000, alpha=0.05)
        assert default == explicit_default

    def test_binary_ci_is_wider_at_a_tighter_alpha(self) -> None:
        wide_default = confidence_interval_binary(0.10, 0.11, 1000, 1000)
        tighter = confidence_interval_binary(0.10, 0.11, 1000, 1000, alpha=0.01)
        default_width = wide_default[1] - wide_default[0]
        tighter_width = tighter[1] - tighter[0]
        assert tighter_width > default_width

    def test_binary_ci_lower_bound_never_reads_below_a_100_percent_drop(self) -> None:
        """A relative lift cannot fall below -100%: the variant rate that is
        being compared to control cannot itself go negative. A wide margin on
        a near-zero variant rate used to push the lower bound past -100%
        (e.g. -130%), which reads as an impossible claim."""
        ci_lower, ci_upper = confidence_interval_binary(0.008, 0.0, 5000, 5000)
        assert ci_lower == -1.0
        assert ci_lower <= ci_upper

    def test_binary_ci_upper_bound_never_exceeds_a_full_rate(self) -> None:
        ci_lower, ci_upper = confidence_interval_binary(0.02, 1.0, 20, 20)
        assert ci_upper == pytest.approx((1.0 - 0.02) / 0.02)

    def test_continuous_ci_default_alpha_matches_the_pinned_z_alpha(self) -> None:
        rng = np.random.default_rng(42)
        group_a = pd.Series(rng.normal(10.0, 2.0, 1000))
        group_b = pd.Series(rng.normal(11.0, 2.0, 1000))
        default = confidence_interval_continuous(group_a, group_b)
        explicit_default = confidence_interval_continuous(group_a, group_b, alpha=0.05)
        assert default == explicit_default


class TestBootstrapCI:
    """Tests for the percentile bootstrap CI on relative lift."""

    def test_brackets_the_true_lift(self) -> None:
        rng = np.random.default_rng(42)
        group_a = pd.Series(rng.normal(10.0, 2.0, 800))
        group_b = pd.Series(rng.normal(11.0, 2.0, 800))  # ~10% true lift
        lower, upper = bootstrap_ci_relative_lift_continuous(group_a, group_b)
        assert lower < 0.10 < upper
        assert lower < upper

    def test_is_reproducible_under_fixed_seed(self) -> None:
        rng = np.random.default_rng(7)
        group_a = pd.Series(rng.normal(5.0, 1.0, 40))
        group_b = pd.Series(rng.normal(5.5, 1.0, 40))
        first = bootstrap_ci_relative_lift_continuous(group_a, group_b)
        second = bootstrap_ci_relative_lift_continuous(group_a, group_b)
        assert first == second

    def test_rejects_zero_control_mean(self) -> None:
        group_a = pd.Series([-1.0, 1.0, -2.0, 2.0])  # mean exactly 0
        group_b = pd.Series([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="non-zero"):
            bootstrap_ci_relative_lift_continuous(group_a, group_b)

    def test_rejects_missing_values(self) -> None:
        group_a = pd.Series([1.0, 2.0, np.nan])
        group_b = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="missing values"):
            bootstrap_ci_relative_lift_continuous(group_a, group_b)


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

    def test_default_alpha_and_power_match_the_pinned_constants(self) -> None:
        """The Z_ALPHA/Z_BETA path must not move existing callers by even one unit.

        Regression test for Item 4: adding optional alpha/power/rho to this
        function must not change its default-argument answer, since Signal 01,
        the sanity checks, and the calibration suite all pin these numbers.
        """
        result = calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900)
        explicit_defaults = calculate_sample_size(
            baseline=0.12, mde=0.05, daily_traffic=900, alpha=0.05, power=0.80, rho=0.0
        )
        assert result == explicit_defaults
        assert result["n_total"] == 93960

    def test_higher_power_costs_more_sample(self) -> None:
        default_power = calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900)
        higher_power = calculate_sample_size(
            baseline=0.12, mde=0.05, daily_traffic=900, power=0.90
        )
        assert higher_power["n_total"] > default_power["n_total"]

    def test_rho_reduces_sample_by_the_cuped_variance_factor(self) -> None:
        """rho applies the same ``(1 - rho**2)`` factor as stats.power.cuped_variance_retained."""
        no_rho = calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900)
        rho = 0.6
        with_rho = calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900, rho=rho)
        assert with_rho["n_total"] == pytest.approx(no_rho["n_total"] * (1 - rho**2), rel=0.01)

    def test_rejects_out_of_range_alpha_power_and_rho(self) -> None:
        with pytest.raises(ValueError, match="Alpha"):
            calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900, alpha=1.5)
        with pytest.raises(ValueError, match="Power"):
            calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900, power=0.0)
        with pytest.raises(ValueError, match="rho"):
            calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900, rho=1.5)

    def test_cluster_design_effect_multiplies_the_required_sample(self) -> None:
        """Item 1: the plan a user locks must be sized under the same clustering
        assumption the sizing block showed them, not as though every user were
        independent."""
        base = calculate_sample_size(baseline=0.12, mde=0.05, daily_traffic=900)
        clustered = calculate_sample_size(
            baseline=0.12, mde=0.05, daily_traffic=900, cluster_design_effect=1.45
        )
        assert clustered["n_total"] == pytest.approx(base["n_total"] * 1.45, rel=0.01)

    def test_rejects_a_design_effect_below_one(self) -> None:
        with pytest.raises(ValueError):
            calculate_sample_size(
                baseline=0.12, mde=0.05, daily_traffic=900, cluster_design_effect=0.9
            )


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
