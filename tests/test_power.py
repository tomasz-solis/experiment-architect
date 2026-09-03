"""Unit tests for the power, variance, duration, and compliance helpers."""

import numpy as np
import pandas as pd
import pytest

from stats.power import (
    allocation_cost,
    compliance_effects,
    cuped_variance_retained,
    estimate_cuped_rho,
    guardrail_detectable_harm,
    intensity_options,
    mde_from_sample_continuous,
    plan_duration,
    post_treatment_risk,
    sample_size_continuous,
    simulate_power,
    skew_diagnostics,
)


class TestAllocationCost:
    """The efficiency penalty an uneven split pays."""

    def test_even_split_is_the_baseline(self) -> None:
        assert allocation_cost(0.5) == pytest.approx(1.0)

    def test_known_multipliers(self) -> None:
        assert allocation_cost(0.2) == pytest.approx(1.5625)
        assert allocation_cost(0.1) == pytest.approx(2.7778, abs=1e-4)

    def test_penalty_is_symmetric(self) -> None:
        assert allocation_cost(0.3) == pytest.approx(allocation_cost(0.7))

    def test_rejects_degenerate_split(self) -> None:
        with pytest.raises(ValueError):
            allocation_cost(0.0)


class TestContinuousSampleSize:
    """n scales with variance and with the inverse square of the effect."""

    def test_matches_textbook_value(self) -> None:
        # sd=1, mde=0.2, alpha=0.05, power=0.80 is the classic ~393-per-arm case.
        result = sample_size_continuous(sd=1.0, mde_absolute=0.2)
        assert 780 <= result["n_total"] <= 800

    def test_doubling_sd_quadruples_sample(self) -> None:
        base = sample_size_continuous(sd=1.0, mde_absolute=0.2)["n_total"]
        doubled = sample_size_continuous(sd=2.0, mde_absolute=0.2)["n_total"]
        assert doubled == pytest.approx(4 * base, rel=0.001)

    def test_doubling_effect_quarters_sample(self) -> None:
        base = sample_size_continuous(sd=1.0, mde_absolute=0.2)["n_total"]
        stronger = sample_size_continuous(sd=1.0, mde_absolute=0.4)["n_total"]
        assert stronger == pytest.approx(base / 4, rel=0.01)

    def test_uneven_split_needs_more_total_sample(self) -> None:
        even = sample_size_continuous(sd=1.0, mde_absolute=0.2)["n_total"]
        uneven = sample_size_continuous(sd=1.0, mde_absolute=0.2, split_ratio=0.2)
        assert uneven["n_total"] > even
        assert uneven["allocation_cost"] == pytest.approx(1.5625)

    def test_cuped_reduces_sample_by_one_minus_rho_squared(self) -> None:
        base = sample_size_continuous(sd=1.0, mde_absolute=0.2)["n_total"]
        adjusted = sample_size_continuous(sd=1.0, mde_absolute=0.2, rho=0.6)
        assert adjusted["variance_retained"] == pytest.approx(0.64)
        assert adjusted["n_total"] == pytest.approx(base * 0.64, rel=0.01)

    def test_higher_power_costs_sample(self) -> None:
        at_80 = sample_size_continuous(sd=1.0, mde_absolute=0.2, power=0.80)["n_total"]
        at_90 = sample_size_continuous(sd=1.0, mde_absolute=0.2, power=0.90)["n_total"]
        assert at_90 > at_80

    def test_relative_mde_reported_when_baseline_supplied(self) -> None:
        result = sample_size_continuous(sd=120.0, mde_absolute=4.0, baseline_mean=100.0)
        assert result["mde_relative"] == pytest.approx(0.04)

    def test_rejects_non_positive_inputs(self) -> None:
        with pytest.raises(ValueError):
            sample_size_continuous(sd=0.0, mde_absolute=1.0)
        with pytest.raises(ValueError):
            sample_size_continuous(sd=1.0, mde_absolute=0.0)


class TestReverseMDE:
    """The inverse direction, which is the honest one when traffic is fixed."""

    def test_round_trips_with_sample_size(self) -> None:
        sized = sample_size_continuous(sd=120.0, mde_absolute=4.0)
        recovered = mde_from_sample_continuous(sd=120.0, n_total=sized["n_total"])
        assert recovered == pytest.approx(4.0, rel=0.01)

    def test_more_sample_detects_smaller_effects(self) -> None:
        small = mde_from_sample_continuous(sd=1.0, n_total=1_000)
        large = mde_from_sample_continuous(sd=1.0, n_total=10_000)
        assert large < small

    def test_variance_reduction_lowers_detectable_effect(self) -> None:
        plain = mde_from_sample_continuous(sd=1.0, n_total=10_000)
        adjusted = mde_from_sample_continuous(sd=1.0, n_total=10_000, rho=0.6)
        assert adjusted < plain


class TestCuped:
    """Variance reduction from a pre-period covariate."""

    def test_zero_rho_removes_nothing(self) -> None:
        assert cuped_variance_retained(0.0) == 1.0

    def test_known_reduction(self) -> None:
        assert cuped_variance_retained(0.8) == pytest.approx(0.36)

    def test_negative_rho_helps_equally(self) -> None:
        assert cuped_variance_retained(-0.6) == pytest.approx(cuped_variance_retained(0.6))

    def test_estimates_rho_from_data(self) -> None:
        rng = np.random.default_rng(7)
        pre = pd.Series(rng.normal(100, 20, 500))
        post = pd.Series(pre.to_numpy() * 0.8 + rng.normal(0, 12, 500))
        assert estimate_cuped_rho(pre, post) == pytest.approx(0.8, abs=0.1)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            estimate_cuped_rho(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]))

    def test_rejects_constant_series(self) -> None:
        with pytest.raises(ValueError):
            estimate_cuped_rho(pd.Series([1.0] * 10), pd.Series(range(10)))


class TestDuration:
    """Sample requirement into calendar time."""

    def test_maturation_extends_beyond_enrolment(self) -> None:
        plan = plan_duration(n_total=1_000, daily_new_eligible=100, maturation_days=30)
        assert plan["enrolment_days"] == 10
        assert plan["total_days"] == 40

    def test_ramp_days_are_added_before_enrolment(self) -> None:
        plan = plan_duration(n_total=1_000, daily_new_eligible=100, ramp_days=5)
        assert plan["total_days"] == 15

    def test_names_maturation_as_the_binding_constraint(self) -> None:
        plan = plan_duration(n_total=1_000, daily_new_eligible=500, maturation_days=30)
        assert "waiting for the metric" in plan["binding_constraint"].lower()

    def test_names_enrolment_when_it_dominates(self) -> None:
        plan = plan_duration(n_total=100_000, daily_new_eligible=500, maturation_days=1)
        assert "sign-up rate" in plan["binding_constraint"].lower()

    def test_rejects_zero_arrival_rate(self) -> None:
        with pytest.raises(ValueError):
            plan_duration(n_total=1_000, daily_new_eligible=0)


class TestSkewDiagnostics:
    """Whether the normal approximation is safe on this metric."""

    def test_symmetric_data_needs_little_sample(self) -> None:
        rng = np.random.default_rng(3)
        diagnostics = skew_diagnostics(pd.Series(rng.normal(100, 10, 2_000)), n_per_arm=500)
        assert abs(diagnostics["skewness"]) < 0.5
        assert diagnostics["normal_approximation_safe"]

    def test_heavy_tail_raises_the_clt_floor(self) -> None:
        rng = np.random.default_rng(3)
        diagnostics = skew_diagnostics(pd.Series(rng.lognormal(3, 1.6, 4_000)), n_per_arm=200)
        assert diagnostics["skewness"] > 2
        assert diagnostics["n_for_clt"] > 200
        assert not diagnostics["normal_approximation_safe"]

    def test_reports_zero_share(self) -> None:
        diagnostics = skew_diagnostics(pd.Series([0.0, 0.0, 0.0, 10.0]))
        assert diagnostics["zero_share"] == pytest.approx(0.75)

    def test_rejects_tiny_samples(self) -> None:
        with pytest.raises(ValueError):
            skew_diagnostics(pd.Series([1.0, 2.0]))


class TestSimulatedPower:
    """Empirical power on a distribution the closed form struggles with."""

    @staticmethod
    def _skewed(n: int = 3_000) -> pd.Series:
        return pd.Series(np.random.default_rng(11).lognormal(3, 1.3, n))

    def test_power_rises_with_sample_size(self) -> None:
        values = self._skewed()
        low = simulate_power(values, relative_lift=0.10, n_per_arm=800, iterations=120)
        high = simulate_power(values, relative_lift=0.10, n_per_arm=8_000, iterations=120)
        assert high["power"] > low["power"]

    def test_power_rises_with_effect_size(self) -> None:
        values = self._skewed()
        small = simulate_power(values, relative_lift=0.02, n_per_arm=4_000, iterations=120)
        large = simulate_power(values, relative_lift=0.20, n_per_arm=4_000, iterations=120)
        assert large["power"] > small["power"]

    def test_false_positive_rate_tracks_alpha(self) -> None:
        result = simulate_power(self._skewed(), relative_lift=0.0, n_per_arm=4_000, iterations=250)
        assert result["false_positive_rate"] < 0.15
        assert result["calibrated"]

    def test_winsorising_records_the_cap(self) -> None:
        result = simulate_power(
            self._skewed(),
            relative_lift=0.05,
            n_per_arm=1_000,
            iterations=60,
            winsorise_quantile=0.99,
        )
        assert result["winsorise_cap"] is not None

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        values = self._skewed()
        first = simulate_power(values, relative_lift=0.05, n_per_arm=1_000, iterations=60, seed=5)
        second = simulate_power(values, relative_lift=0.05, n_per_arm=1_000, iterations=60, seed=5)
        assert first["power"] == second["power"]

    def test_rejects_tiny_history(self) -> None:
        with pytest.raises(ValueError):
            simulate_power(pd.Series([1.0, 2.0]), relative_lift=0.1, n_per_arm=100)


class TestCompliance:
    """ITT stays primary; CACE is the scaled secondary."""

    def test_cace_scales_itt_by_the_take_up_gap(self) -> None:
        result = compliance_effects(itt=0.01, itt_standard_error=0.002, take_up_treatment=0.5)
        assert result["cace"] == pytest.approx(0.02)

    def test_low_take_up_widens_the_complier_interval(self) -> None:
        high = compliance_effects(itt=0.01, itt_standard_error=0.002, take_up_treatment=0.8)
        low = compliance_effects(itt=0.01, itt_standard_error=0.002, take_up_treatment=0.05)
        assert low["cace_ci"] is not None and high["cace_ci"] is not None
        assert (low["cace_ci"][1] - low["cace_ci"][0]) > (high["cace_ci"][1] - high["cace_ci"][0])
        assert "fragile" in low["estimand_note"]

    def test_no_take_up_gap_identifies_no_complier_effect(self) -> None:
        result = compliance_effects(
            itt=0.01, itt_standard_error=0.002, take_up_treatment=0.3, take_up_control=0.3
        )
        assert result["cace"] is None
        assert result["cace_ci"] is None

    def test_control_take_up_is_netted_out(self) -> None:
        result = compliance_effects(
            itt=0.01, itt_standard_error=0.002, take_up_treatment=0.5, take_up_control=0.25
        )
        assert result["cace"] == pytest.approx(0.04)

    def test_rejects_impossible_take_up(self) -> None:
        with pytest.raises(ValueError):
            compliance_effects(itt=0.01, itt_standard_error=0.002, take_up_treatment=1.4)


class TestGuardrailSensitivity:
    """What harm the test could actually have caught."""

    def test_rare_guardrails_need_far_more_sample(self) -> None:
        rare = guardrail_detectable_harm(baseline_rate=0.002, n_total=40_000)
        common = guardrail_detectable_harm(baseline_rate=0.20, n_total=40_000)
        assert rare > common

    def test_more_sample_detects_smaller_harm(self) -> None:
        small = guardrail_detectable_harm(baseline_rate=0.01, n_total=20_000)
        large = guardrail_detectable_harm(baseline_rate=0.01, n_total=200_000)
        assert large < small

    def test_rejects_rate_outside_zero_one(self) -> None:
        with pytest.raises(ValueError):
            guardrail_detectable_harm(baseline_rate=1.5, n_total=1_000)


class TestIntensityOptions:
    """Stronger doses buy time, with diminishing returns."""

    def test_stronger_dose_needs_less_sample(self) -> None:
        options = intensity_options(
            doses=[("weak", 0.02), ("strong", 0.06)],
            baseline=100.0,
            sd=120.0,
            daily_new_eligible=900.0,
        )
        assert options[0]["label"] == "weak"
        assert options[-1]["n_total"] < options[0]["n_total"]
        assert options[-1]["days_saved_vs_weakest"] > 0

    def test_marginal_return_falls_as_the_dose_grows(self) -> None:
        options = intensity_options(
            doses=[("low", 0.02), ("mid", 0.04), ("high", 0.06)],
            baseline=100.0,
            sd=120.0,
            daily_new_eligible=900.0,
        )
        mid_return = options[1]["marginal_days_per_effect_point"]
        high_return = options[2]["marginal_days_per_effect_point"]
        assert mid_return is not None and high_return is not None
        assert high_return < mid_return

    def test_rejects_empty_or_invalid_doses(self) -> None:
        with pytest.raises(ValueError):
            intensity_options(doses=[], baseline=100.0, sd=1.0, daily_new_eligible=10.0)
        with pytest.raises(ValueError):
            intensity_options(
                doses=[("bad", 0.0)], baseline=100.0, sd=1.0, daily_new_eligible=10.0
            )


class TestPostTreatmentRisk:
    """Outcome definitions that break the randomised comparison."""

    def test_value_per_active_user_fails(self) -> None:
        status, explanation = post_treatment_risk("value_per_active", False)
        assert status == "fail"
        assert "no longer fair" in explanation.lower()

    def test_any_post_state_filter_fails(self) -> None:
        status, _ = post_treatment_risk("value_per_randomised", True)
        assert status == "fail"

    def test_value_per_randomised_user_is_clean(self) -> None:
        status, _ = post_treatment_risk("value_per_randomised", False)
        assert status == "ok"

    def test_conversion_is_safe_but_narrow(self) -> None:
        status, explanation = post_treatment_risk("conversion", False)
        assert status == "caution"
        assert "never how much" in explanation
