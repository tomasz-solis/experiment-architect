"""Unit tests for causal inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.causal import (
    check_parallel_trends,
    difference_in_differences,
    regression_discontinuity,
    select_causal_method,
)


def make_did_data(
    n_units: int = 100,
    true_effect: float = 5.0,
    pre_trend_slope: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate panel data with a known treatment effect and optional pre-trend violation."""
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    treated = (units >= n_units // 2).astype(int)

    rows: list[dict[str, float | int]] = []
    for unit, is_treated in zip(units, treated):
        for period in (0, 1, 2, 3):
            post = int(period >= 2)
            outcome = 10 + 2 * is_treated + 3 * period + true_effect * is_treated * post
            if period < 2:
                outcome += pre_trend_slope * is_treated * period
            outcome += rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": int(unit),
                    "period": int(period),
                    "treated": int(is_treated),
                    "outcome": float(outcome),
                }
            )
    return pd.DataFrame(rows)


def make_rdd_data(
    n: int = 500,
    true_effect: float = 3.0,
    cutoff: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data with a known discontinuity at the cutoff."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(0, 100, n)
    treated = (running >= cutoff).astype(int)
    outcome = 2 * running + true_effect * treated + rng.normal(0, 1, n)
    return pd.DataFrame({"score": running, "treated": treated, "outcome": outcome})


def make_manipulated_rdd_data(
    n: int = 500,
    true_effect: float = 3.0,
    cutoff: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate RDD data with excess mass just above the cutoff."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 100, n)
    oversample = rng.uniform(cutoff, cutoff + 5, n // 2)
    running = np.concatenate([base, oversample])
    treated = (running >= cutoff).astype(int)
    outcome = 2 * running + true_effect * treated + rng.normal(0, 1, len(running))
    return pd.DataFrame({"score": running, "treated": treated, "outcome": outcome})


class TestDifferenceInDifferences:
    """Tests for the DiD implementation."""

    def test_recovers_known_effect(self) -> None:
        df = make_did_data(true_effect=5.0)
        result = difference_in_differences(
            df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=2,
        )
        assert abs(result["coefficient"] - 5.0) < 0.5

    def test_no_effect_returns_near_zero(self) -> None:
        df = make_did_data(true_effect=0.0)
        result = difference_in_differences(
            df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=2,
        )
        assert abs(result["coefficient"]) < 1.0

    def test_placebo_intervention_in_pre_period_is_near_zero(self) -> None:
        df = make_did_data(true_effect=5.0)
        placebo_df = df[df["period"] < 2].copy()
        result = difference_in_differences(
            placebo_df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=1,
        )
        assert abs(result["coefficient"]) < 0.5

    def test_returns_required_keys(self) -> None:
        df = make_did_data()
        result = difference_in_differences(
            df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=2,
        )
        assert all(
            key in result
            for key in ["coefficient", "p_value", "ci_lower", "ci_upper", "model", "diagnostics"]
        )

    def test_ci_contains_true_effect(self) -> None:
        df = make_did_data(true_effect=5.0, n_units=500)
        result = difference_in_differences(
            df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=2,
        )
        assert result["ci_lower"] < 5.0 < result["ci_upper"]

    def test_does_not_mutate_input(self) -> None:
        df = make_did_data()
        original_cols = set(df.columns)
        difference_in_differences(
            df,
            unit_col="unit",
            time_col="period",
            treatment_col="treated",
            outcome_col="outcome",
            intervention_point=2,
        )
        assert set(df.columns) == original_cols


class TestParallelTrends:
    """Tests for the pre-period trend check."""

    def test_parallel_trends_passes_on_balanced_data(self) -> None:
        df = make_did_data(true_effect=5.0)
        result = check_parallel_trends(df, "period", "treated", "outcome", 2)
        assert result["passes"]
        assert 0 <= result["trend_interaction_pvalue"] <= 1

    def test_parallel_trends_fails_on_diverging_pre_periods(self) -> None:
        df = make_did_data(true_effect=5.0, pre_trend_slope=2.0)
        result = check_parallel_trends(df, "period", "treated", "outcome", 2)
        assert not result["passes"]
        assert result["trend_interaction_pvalue"] < 0.10


class TestRegressionDiscontinuity:
    """Tests for the RDD implementation."""

    def test_recovers_known_effect(self) -> None:
        df = make_rdd_data(true_effect=3.0)
        result = regression_discontinuity(df, "score", "treated", "outcome", cutoff=50.0)
        assert abs(result["coefficient"] - 3.0) < 0.5

    def test_returns_required_keys(self) -> None:
        df = make_rdd_data()
        result = regression_discontinuity(df, "score", "treated", "outcome", cutoff=50.0)
        assert all(
            key in result
            for key in ["coefficient", "p_value", "ci_lower", "ci_upper", "model", "diagnostics"]
        )

    def test_detects_density_manipulation(self) -> None:
        df = make_manipulated_rdd_data()
        result = regression_discontinuity(df, "score", "treated", "outcome", cutoff=50.0)
        assert not result["diagnostics"]["density_ok"]
        assert result["diagnostics"]["density_ratio_at_cutoff"] > 1.4

    def test_uses_rule_of_thumb_bandwidth_by_default(self) -> None:
        df = make_rdd_data()
        result = regression_discontinuity(df, "score", "treated", "outcome", cutoff=50.0)
        diagnostics = result["diagnostics"]

        assert diagnostics["bandwidth_method"] == "rule_of_thumb"
        assert diagnostics["bandwidth_used"] > 0
        assert len(diagnostics["bandwidth_sweep"]) >= 2

    def test_respects_user_supplied_bandwidth(self) -> None:
        df = make_rdd_data()
        result = regression_discontinuity(
            df,
            "score",
            "treated",
            "outcome",
            cutoff=50.0,
            bandwidth=8.0,
        )
        diagnostics = result["diagnostics"]

        assert diagnostics["bandwidth_method"] == "user_supplied"
        assert diagnostics["bandwidth_used"] == 8.0

    def test_does_not_mutate_input(self) -> None:
        df = make_rdd_data()
        original_cols = set(df.columns)
        regression_discontinuity(df, "score", "treated", "outcome", cutoff=50.0)
        assert set(df.columns) == original_cols


class TestCausalMethodSelector:
    """Tests for the causal method recommender."""

    def test_rdd_takes_priority(self) -> None:
        assert "RDD" in select_causal_method(
            has_cutoff=True,
            has_clean_control=True,
            is_opt_in=True,
        )

    def test_did_when_clean_control_no_optin(self) -> None:
        assert "DiD" in select_causal_method(
            has_cutoff=False,
            has_clean_control=True,
            is_opt_in=False,
        )

    def test_psm_when_optin(self) -> None:
        assert "PSM" in select_causal_method(
            has_cutoff=False,
            has_clean_control=True,
            is_opt_in=True,
        )

    def test_causalimpact_as_fallback(self) -> None:
        assert "CausalImpact" in select_causal_method(
            has_cutoff=False,
            has_clean_control=False,
            is_opt_in=False,
        )
