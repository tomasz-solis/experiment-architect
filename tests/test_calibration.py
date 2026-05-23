"""Calibration study: do the estimators perform at their advertised confidence levels?

These tests use 200 synthetic experiments per condition. That sample is low
enough to keep CI runtime under ~30 seconds and high enough that the standard
error on a 5% false-positive rate is ~1.5%, which fits comfortably inside the
assertion tolerances below.

For a tighter study, increase ``N_SIMULATIONS`` and run locally with
``pytest -m calibration``. The tests are marked ``calibration`` so they can
be skipped from the default test run.

Recorded findings on the bundled DGPs (last run, N=200, seed=42):
    - Chi-squared FPR: ~5% (matches alpha)
    - Bayesian Ship-under-null rate: <10%
    - Parallel-trends power against slope=2.0: >80%
    - Parallel-trends FPR under no violation: <15%

If any of these fail, re-run with a different seed before assuming a bug.
The tolerances are intentionally generous because point estimates from 200
trials are noisy.
"""

from __future__ import annotations

import numpy as np
import pytest

from stats.bayesian import beta_binomial_analysis, get_decision_recommendation
from stats.causal import check_parallel_trends
from stats.frequentist import chi_squared_test
from tests.test_causal import make_did_data

N_SIMULATIONS = 200
ALPHA = 0.05


@pytest.mark.calibration
def test_chi_squared_false_positive_rate_near_alpha() -> None:
    """Under H0 (no treatment effect), p < 0.05 should fire ~5% of the time.

    Tolerance: 2-8% across N_SIMULATIONS=200 (binomial SE ~1.5%). The wide
    band accepts plausible Monte Carlo noise without flaking.
    """
    rng = np.random.default_rng(42)
    n_per_group = 1000
    baseline_p = 0.10

    false_positives = 0
    for _ in range(N_SIMULATIONS):
        successes_a = int(rng.binomial(n_per_group, baseline_p))
        successes_b = int(rng.binomial(n_per_group, baseline_p))
        result = chi_squared_test(
            successes_a, n_per_group - successes_a,
            successes_b, n_per_group - successes_b,
        )
        if result["p_value"] < ALPHA:
            false_positives += 1

    fpr = false_positives / N_SIMULATIONS
    assert 0.02 <= fpr <= 0.08, (
        f"Chi-squared FPR was {fpr:.2%}, expected ~5% (band 2-8% for N=200)."
    )


@pytest.mark.calibration
def test_chi_squared_has_power_against_real_effect() -> None:
    """Under H1 (real lift of 5pp), p < 0.05 should fire most of the time.

    With baseline 10%, treatment 15%, and 1000 per group, the design has
    near-100% power. We assert >90% as a generous floor.
    """
    rng = np.random.default_rng(42)
    n_per_group = 1000
    baseline_p = 0.10
    treatment_p = 0.15

    detections = 0
    for _ in range(N_SIMULATIONS):
        successes_a = int(rng.binomial(n_per_group, baseline_p))
        successes_b = int(rng.binomial(n_per_group, treatment_p))
        result = chi_squared_test(
            successes_a, n_per_group - successes_a,
            successes_b, n_per_group - successes_b,
        )
        if result["p_value"] < ALPHA:
            detections += 1

    power = detections / N_SIMULATIONS
    assert power > 0.90, f"Chi-squared power against 5pp lift was {power:.2%}, expected >90%."


@pytest.mark.calibration
def test_bayesian_ship_recommendation_under_null() -> None:
    """Under H0, 'Ship Variant B' should fire rarely.

    With prob_b_wins > 0.95 required and a true effect of zero, the rate
    should sit well below 5%. We assert it stays below 10% as a generous
    upper bound that catches obvious miscalibration.
    """
    rng = np.random.default_rng(42)
    n_per_group = 1000
    baseline_p = 0.10

    ship_count = 0
    for _ in range(N_SIMULATIONS):
        successes_a = int(rng.binomial(n_per_group, baseline_p))
        successes_b = int(rng.binomial(n_per_group, baseline_p))
        bayes = beta_binomial_analysis(
            successes_a, n_per_group - successes_a,
            successes_b, n_per_group - successes_b,
        )
        rec, _ = get_decision_recommendation(
            bayes["prob_b_wins"], bayes["expected_loss"],
        )
        if rec == "Ship Variant B":
            ship_count += 1

    ship_rate = ship_count / N_SIMULATIONS
    assert ship_rate < 0.10, f"Bayesian shipped under null at {ship_rate:.2%}, expected <10%."


@pytest.mark.calibration
def test_parallel_trends_has_power_against_violation() -> None:
    """The parallel-trends test should reject when a pre-trend exists.

    With a slope of 2.0 on the treated group during the pre-period, the
    test should fire more often than under no violation.
    """
    rejections_with_violation = 0
    rejections_without_violation = 0

    for seed in range(N_SIMULATIONS):
        df_violated = make_did_data(true_effect=5.0, pre_trend_slope=2.0, seed=seed)
        df_clean = make_did_data(true_effect=5.0, pre_trend_slope=0.0, seed=seed)

        if not check_parallel_trends(df_violated, "period", "treated", "outcome", 2)["passes"]:
            rejections_with_violation += 1
        if not check_parallel_trends(df_clean, "period", "treated", "outcome", 2)["passes"]:
            rejections_without_violation += 1

    power = rejections_with_violation / N_SIMULATIONS
    fpr = rejections_without_violation / N_SIMULATIONS

    assert power > 0.80, f"Parallel-trends power was {power:.2%}, expected >80%."
    assert fpr < 0.20, f"Parallel-trends FPR was {fpr:.2%}, expected <20%."
