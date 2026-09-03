"""Unit tests for the pre-registration contract and its post-test verification."""

import pytest

from stats.prereg import (
    PreRegistration,
    VerificationRow,
    achieved_mde,
    build_preregistration,
    summarise_readout,
    verify_against_plan,
)


def make_plan(**overrides: object) -> PreRegistration:
    """Build a plan with sensible defaults, overridden per test."""
    kwargs: dict[str, object] = {
        "primary_metric": "Checkout conversion",
        "metric_layer": "conversion",
        "baseline": 0.12,
        "mde_relative": 0.05,
        "n_total": 40_000,
        "ramp_days": 0,
        "enrolment_days": 44,
        "maturation_days": 0,
        "daily_new_eligible": 900.0,
    }
    kwargs.update(overrides)
    return build_preregistration(**kwargs)  # type: ignore[arg-type]


def status_for(rows: list[VerificationRow], item: str) -> str:
    """Pull one commitment's status out of a verification table."""
    return next(row["status"] for row in rows if row["item"] == item)


class TestBuildPreregistration:
    """Freezing the decisions that are easiest to revise later."""

    def test_records_the_transform_and_primary_metric_count(self) -> None:
        plan = make_plan(transform="winsorise p99", n_primary_metrics=2)
        assert plan["transform"] == "winsorise p99"
        assert plan["n_primary_metrics"] == 2

    def test_total_days_sums_ramp_enrolment_and_maturation(self) -> None:
        plan = make_plan(ramp_days=5, enrolment_days=44, maturation_days=30)
        assert plan["total_days"] == 79

    def test_computes_guardrail_sensitivity_when_a_baseline_is_given(self) -> None:
        plan = make_plan(guardrail_baseline=0.002)
        assert plan["guardrail_detectable_harm"] is not None
        assert plan["guardrail_detectable_harm"] > 0.1

    def test_skips_guardrail_sensitivity_without_a_baseline(self) -> None:
        assert make_plan()["guardrail_detectable_harm"] is None

    def test_rejects_an_impossible_rate_baseline(self) -> None:
        with pytest.raises(ValueError):
            make_plan(baseline=1.4)

    def test_rejects_a_non_positive_mde(self) -> None:
        with pytest.raises(ValueError):
            make_plan(mde_relative=0.0)


class TestAchievedMDE:
    """What the delivered sample could actually detect."""

    def test_full_sample_recovers_roughly_the_planned_mde(self) -> None:
        plan = make_plan(n_total=40_000)
        assert achieved_mde(plan, 40_000) > 0

    def test_short_sample_raises_the_detectable_effect(self) -> None:
        plan = make_plan()
        assert achieved_mde(plan, 20_000) > achieved_mde(plan, 40_000)

    def test_rejects_zero_sample(self) -> None:
        with pytest.raises(ValueError):
            achieved_mde(make_plan(), 0)


class TestVerifyAgainstPlan:
    """Every row corresponds to a way a test silently stops answering the question."""

    def test_a_clean_run_passes_every_commitment(self) -> None:
        plan = make_plan()
        rows = verify_against_plan(
            plan=plan,
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            actual_days=plan["total_days"],
        )
        assert all(row["status"] == "ok" for row in rows)

    def test_short_sample_is_flagged_as_underpowered(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(), actual_n_total=20_000, actual_split_ratio=0.5
        )
        assert status_for(rows, "Sample delivered") == "fail"

    def test_slight_shortfall_is_a_caution_not_a_failure(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(), actual_n_total=38_000, actual_split_ratio=0.5
        )
        assert status_for(rows, "Sample delivered") == "caution"

    def test_split_drift_is_a_sample_ratio_mismatch(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(), actual_n_total=40_000, actual_split_ratio=0.56
        )
        assert status_for(rows, "Assignment split") == "fail"

    def test_non_itt_analysis_fails(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(),
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            analysed_as_itt=False,
        )
        assert status_for(rows, "Estimand") == "fail"

    def test_changed_transform_fails(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(transform="none"),
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            transform_applied="winsorise p99",
        )
        assert status_for(rows, "Outcome transform") == "fail"

    def test_incomplete_maturation_is_flagged(self) -> None:
        plan = make_plan(maturation_days=30)
        rows = verify_against_plan(
            plan=plan,
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            actual_days=plan["total_days"] - 10,
            maturation_complete=False,
        )
        assert status_for(rows, "Duration and maturation") == "caution"

    def test_extra_metrics_flag_alpha_inflation(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(n_primary_metrics=1),
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            metrics_tested=4,
        )
        row = next(row for row in rows if row["item"] == "Primary metrics tested")
        assert row["status"] == "caution"
        assert "19%" in row["note"]

    def test_unplanned_peeking_is_flagged(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(planned_looks=1),
            actual_n_total=40_000,
            actual_split_ratio=0.5,
            looks_taken=6,
        )
        assert status_for(rows, "Interim looks") == "caution"

    def test_insensitive_guardrail_is_reported_as_unverified(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(guardrail_baseline=0.002),
            actual_n_total=40_000,
            actual_split_ratio=0.5,
        )
        row = next(row for row in rows if row["item"] == "Guardrail sensitivity")
        assert row["status"] == "caution"
        assert "unverified" in row["note"]


class TestSummariseReadout:
    """Decision first, then uncertainty, then money."""

    def test_material_win_when_the_interval_clears_the_bar(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1284,
            ci_relative=(0.03, 0.11),
            mde_relative=0.05,
        )
        assert summary["material"]
        assert summary["relative_uplift"] == pytest.approx(0.07)
        assert "Worth shipping" in summary["headline"]

    def test_real_but_immaterial_effect_is_not_a_ship(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1224,
            ci_relative=(0.005, 0.035),
            mde_relative=0.05,
        )
        assert not summary["material"]
        assert summary["conclusive"]
        assert "too small to act on" in summary["headline"]

    def test_wide_interval_reads_as_inconclusive_not_as_no_effect(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1212,
            ci_relative=(-0.06, 0.08),
            mde_relative=0.05,
        )
        assert not summary["conclusive"]
        assert "Cannot tell" in summary["headline"]

    def test_tight_null_rules_out_a_material_effect(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1202,
            ci_relative=(-0.01, 0.015),
            mde_relative=0.05,
        )
        assert summary["conclusive"]
        assert "Nothing worth acting on" in summary["headline"]

    def test_business_impact_scales_the_interval(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1284,
            ci_relative=(0.03, 0.11),
            mde_relative=0.05,
            population_size=500_000,
            value_per_unit=40.0,
        )
        assert summary["business_impact"] is not None
        low, high = summary["business_impact"]
        assert low == pytest.approx(0.03 * 0.12 * 500_000 * 40.0)
        assert high > low

    def test_impact_is_omitted_without_population_and_value(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1284,
            ci_relative=(0.03, 0.11),
            mde_relative=0.05,
        )
        assert summary["business_impact"] is None

    def test_rejects_an_inverted_interval(self) -> None:
        with pytest.raises(ValueError):
            summarise_readout(
                baseline=0.12,
                observed_rate_or_mean=0.13,
                ci_relative=(0.10, 0.02),
                mde_relative=0.05,
            )
