"""Unit tests for the pre-registration contract and its post-test verification."""

import json
import math

import pytest

from stats.prereg import (
    PreRegistration,
    VerificationRow,
    achieved_mde,
    build_preregistration,
    metric_labels_match,
    parse_preregistration,
    read_guardrail,
    serialise_preregistration,
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

    def test_cluster_design_effect_widens_the_achieved_mde(self) -> None:
        """Item 1: verification must use the same clustering assumption the
        plan was locked under, or a delivered sample from a group-randomised
        test reads as sharper than it actually is."""
        plain = achieved_mde(make_plan(cluster_design_effect=1.0), 40_000)
        clustered = achieved_mde(make_plan(cluster_design_effect=1.45), 40_000)
        assert clustered > plain


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

    def test_modest_overshoot_still_passes(self) -> None:
        """A quarter more than planned is ordinary overshoot, not a red flag."""
        rows = verify_against_plan(
            plan=make_plan(), actual_n_total=48_000, actual_split_ratio=0.5
        )
        assert status_for(rows, "Sample delivered") == "ok"

    def test_large_overshoot_is_flagged_as_a_stopping_rule_risk(self) -> None:
        rows = verify_against_plan(
            plan=make_plan(), actual_n_total=120_000, actual_split_ratio=0.5
        )
        row = next(row for row in rows if row["item"] == "Sample delivered")
        assert row["status"] == "caution"
        assert "stopping rule" in row["note"]

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


class TestMetricLabelsMatch:
    """A plan written for one metric should not silently verify another."""

    def test_identical_labels_match(self) -> None:
        assert metric_labels_match("Checkout conversion", "Checkout conversion")

    def test_planned_label_contained_in_observed_label_matches(self) -> None:
        assert metric_labels_match("Checkout conversion", "checkout_conversion_rate")

    def test_observed_label_contained_in_planned_label_matches(self) -> None:
        assert metric_labels_match("checkout_conversion_rate", "Checkout conversion")

    def test_unrelated_labels_do_not_match(self) -> None:
        assert not metric_labels_match("Checkout conversion", "Revenue per user")

    def test_empty_planned_label_matches_anything(self) -> None:
        assert metric_labels_match("", "Revenue per user")

    def test_empty_observed_label_matches_anything(self) -> None:
        assert metric_labels_match("Checkout conversion", "   ")


class TestSummariseReadout:
    """Decision first, then uncertainty, then money."""

    def test_material_win_when_the_interval_clears_the_bar(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1284,
            ci_relative=(0.06, 0.11),
            mde_relative=0.05,
        )
        assert summary["material"]
        assert summary["floor_clears_bar"]
        assert summary["relative_uplift"] == pytest.approx(0.07)
        assert "Worth shipping" in summary["headline"]

    def test_material_but_floor_does_not_clear_the_bar(self) -> None:
        """A point estimate above the bar is not enough if the pessimistic end isn't."""
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1272,
            ci_relative=(0.001, 0.119),
            mde_relative=0.05,
        )
        assert summary["material"]
        assert not summary["floor_clears_bar"]
        assert "Probably worth shipping" in summary["headline"]

    def test_upper_bound_alone_does_not_make_a_result_conclusive(self) -> None:
        """Ruling out only a win, while a large loss is still in the interval, is not conclusive."""
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1152,
            ci_relative=(-0.20, 0.04),
            mde_relative=0.05,
        )
        assert not summary["conclusive"]
        assert not summary["downside_ruled_out"]
        assert "Cannot rule out a loss" in summary["headline"]
        # The caption must not contradict the headline above it: it must say the
        # downside is still an open loss, not that "whatever is there is smaller
        # than you care about" (that reading only holds once the downside is
        # ruled out too).
        assert "has not ruled out" in summary["uncertainty_line"]
        assert "20.0%" in summary["uncertainty_line"]
        assert "smaller than you care about" not in summary["uncertainty_line"]

    def test_uncertainty_line_uses_the_ruled_out_wording_once_downside_is_clear(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1202,
            ci_relative=(-0.01, 0.015),
            mde_relative=0.05,
        )
        assert summary["downside_ruled_out"]
        assert "smaller than you care about" in summary["uncertainty_line"]

    def test_uncertainty_line_says_the_range_still_includes_material_change(self) -> None:
        summary = summarise_readout(
            baseline=0.12,
            observed_rate_or_mean=0.1212,
            ci_relative=(-0.06, 0.08),
            mde_relative=0.05,
        )
        assert "still includes changes worth acting on" in summary["uncertainty_line"]

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


class TestReadGuardrail:
    """A guardrail baseline is only worth writing down if the reading is checked."""

    def test_a_significant_worsening_fails_and_names_the_ship_decision(self) -> None:
        reading = read_guardrail(
            control_events=200, control_n=10_000, variant_events=400, variant_n=10_000
        )
        assert reading["status"] == "fail"
        assert reading["ci_relative"][0] > 0
        assert "does not belong to the primary metric alone" in reading["note"]

    def test_a_worse_point_estimate_inside_a_noisy_interval_is_a_caution(self) -> None:
        reading = read_guardrail(
            control_events=10, control_n=1_000, variant_events=12, variant_n=1_000
        )
        assert reading["status"] == "caution"
        assert reading["relative_change"] > 0
        assert reading["ci_relative"][0] < 0 < reading["ci_relative"][1]

    def test_an_underpowered_clean_reading_is_a_caution_not_proof_of_safety(self) -> None:
        """A flat reading the test could never have caught a real problem with is
        unverified, not evidence of safety, and the note must say so."""
        reading = read_guardrail(
            control_events=100,
            control_n=10_000,
            variant_events=100,
            variant_n=10_000,
            detectable_harm=0.50,
        )
        assert reading["status"] == "caution"
        assert reading["relative_change"] == pytest.approx(0.0)
        assert "unverified" in reading["note"]
        assert "not proof" in reading["note"]

    def test_a_flat_reading_with_no_detectable_harm_given_is_ok(self) -> None:
        reading = read_guardrail(
            control_events=100, control_n=10_000, variant_events=100, variant_n=10_000
        )
        assert reading["status"] == "ok"

    def test_an_improved_reading_is_ok_when_powered_to_see_it(self) -> None:
        reading = read_guardrail(
            control_events=200,
            control_n=10_000,
            variant_events=100,
            variant_n=10_000,
            detectable_harm=0.10,
        )
        assert reading["status"] == "ok"
        assert reading["relative_change"] < 0

    def test_rejects_a_zero_sample(self) -> None:
        with pytest.raises(ValueError, match="sample size"):
            read_guardrail(control_events=0, control_n=0, variant_events=0, variant_n=100)

    def test_rejects_events_exceeding_users(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            read_guardrail(control_events=200, control_n=100, variant_events=0, variant_n=100)

    def test_both_arms_zero_returns_caution_with_nan_ci(self) -> None:
        """When neither arm saw any events, return status caution with NaN CI."""
        reading = read_guardrail(control_events=0, control_n=1_000, variant_events=0, variant_n=1_000)
        assert reading["status"] == "caution"
        assert reading["relative_change"] == 0.0
        assert math.isnan(reading["ci_relative"][0])
        assert math.isnan(reading["ci_relative"][1])
        assert "Neither arm saw" in reading["note"]

    def test_control_zero_variant_events_higher_is_worse_significant(self) -> None:
        """Control zero with variant events, higher_is_worse=True, significant difference -> fail."""
        reading = read_guardrail(
            control_events=0, control_n=5_000,
            variant_events=7, variant_n=5_000,
            higher_is_worse=True,
        )
        assert reading["status"] == "fail"
        assert math.isinf(reading["relative_change"])
        assert math.isnan(reading["ci_relative"][0])
        assert math.isnan(reading["ci_relative"][1])
        assert "7" in reading["note"] and "5,000" in reading["note"]
        assert "percentage change against a zero baseline does not exist" in reading["note"]

    def test_control_zero_variant_events_higher_is_worse_not_significant(self) -> None:
        """Control zero with variant events, higher_is_worse=True, not significant -> caution."""
        reading = read_guardrail(
            control_events=0, control_n=5_000,
            variant_events=1, variant_n=5_000,
            higher_is_worse=True,
        )
        assert reading["status"] == "caution"
        assert math.isinf(reading["relative_change"])
        assert "1" in reading["note"] and "5,000" in reading["note"]

    def test_control_zero_variant_events_higher_is_better_improvement(self) -> None:
        """Control zero with variant events, higher_is_worse=False, is improvement -> ok."""
        reading = read_guardrail(
            control_events=0, control_n=5_000,
            variant_events=7, variant_n=5_000,
            higher_is_worse=False,
        )
        assert reading["status"] == "ok"
        assert math.isinf(reading["relative_change"])
        assert "improvement" in reading["note"]

    def test_variant_zero_control_nonzero_unchanged(self) -> None:
        """Variant zero with control events still works through normal path."""
        reading = read_guardrail(
            control_events=100, control_n=10_000,
            variant_events=0, variant_n=10_000,
        )
        assert reading["status"] == "ok"
        assert reading["relative_change"] < 0
        ci_lower, ci_upper = reading["ci_relative"]
        assert not any(math.isnan(x) for x in (ci_lower, ci_upper))


class TestReadGuardrailHigherIsBetter:
    """Item 3: a guardrail like retention or successful deliveries has the harm
    direction the other way round, and the reading must follow it rather than
    always treating a rate increase as the worse outcome."""

    def test_a_significant_drop_fails_and_names_the_ship_decision(self) -> None:
        reading = read_guardrail(
            control_events=500,
            control_n=1_000,
            variant_events=250,
            variant_n=1_000,
            higher_is_worse=False,
        )
        assert reading["status"] == "fail"
        assert reading["ci_relative"][1] < 0
        assert "does not belong to the primary metric alone" in reading["note"]

    def test_a_small_drop_inside_a_noisy_interval_is_a_caution(self) -> None:
        reading = read_guardrail(
            control_events=500,
            control_n=1_000,
            variant_events=480,
            variant_n=1_000,
            higher_is_worse=False,
        )
        assert reading["status"] == "caution"
        assert reading["relative_change"] < 0
        assert reading["ci_relative"][0] < 0 < reading["ci_relative"][1]

    def test_an_underpowered_clean_reading_is_still_a_caution(self) -> None:
        reading = read_guardrail(
            control_events=100,
            control_n=10_000,
            variant_events=100,
            variant_n=10_000,
            detectable_harm=0.50,
            higher_is_worse=False,
        )
        assert reading["status"] == "caution"
        assert "unverified" in reading["note"]

    def test_a_flat_reading_with_no_detectable_harm_given_is_ok(self) -> None:
        reading = read_guardrail(
            control_events=100,
            control_n=10_000,
            variant_events=100,
            variant_n=10_000,
            higher_is_worse=False,
        )
        assert reading["status"] == "ok"

    def test_an_improvement_no_longer_reads_as_a_failure(self) -> None:
        """The exact input that fails under the default higher-is-worse
        direction (the rate doubled) must read as ok once that same rate rise
        is the improvement, not the harm, for this guardrail."""
        reading = read_guardrail(
            control_events=200,
            control_n=10_000,
            variant_events=400,
            variant_n=10_000,
            higher_is_worse=False,
        )
        assert reading["status"] == "ok"
        assert reading["relative_change"] > 0


class TestSerialisePreregistration:
    """A locked plan must survive a page refresh via download/upload."""

    def test_round_trips_exactly(self) -> None:
        plan = make_plan(guardrail_baseline=0.002)
        restored = parse_preregistration(serialise_preregistration(plan))
        assert restored == plan

    def test_serialises_to_sorted_indented_json(self) -> None:
        plan = make_plan()
        raw = serialise_preregistration(plan)
        assert raw == json.dumps(plan, indent=2, sort_keys=True)

    def test_rejects_text_that_is_not_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_preregistration("this is not json")

    def test_rejects_json_that_is_not_an_object(self) -> None:
        with pytest.raises(ValueError, match="not a"):
            parse_preregistration(json.dumps([1, 2, 3]))

    def test_lists_every_missing_key_in_one_message(self) -> None:
        raw = json.dumps({"primary_metric": "Checkout conversion"})
        with pytest.raises(ValueError) as excinfo:
            parse_preregistration(raw)
        message = str(excinfo.value)
        assert "n_total" in message
        assert "baseline" in message

    def test_rejects_a_wrong_typed_field(self) -> None:
        plan = dict(make_plan())
        plan["n_total"] = "forty thousand"
        with pytest.raises(ValueError, match="n_total"):
            parse_preregistration(json.dumps(plan))

    def test_rejects_an_unknown_metric_layer(self) -> None:
        plan = dict(make_plan())
        plan["metric_layer"] = "made_up_layer"
        with pytest.raises(ValueError, match="made_up_layer"):
            parse_preregistration(json.dumps(plan))

    def test_a_plan_written_before_clustering_and_guardrail_direction_still_loads(self) -> None:
        """Item 1 and Item 3: a plan downloaded before these fields existed is
        not malformed, it is older than the field, so it must load with the
        documented compatibility defaults instead of failing on a missing key."""
        plan = dict(make_plan())
        del plan["cluster_design_effect"]
        del plan["guardrail_higher_is_worse"]
        restored = parse_preregistration(json.dumps(plan))
        assert restored["cluster_design_effect"] == 1.0
        assert restored["guardrail_higher_is_worse"] is True
