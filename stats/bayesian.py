"""Bayesian statistical helpers for binary experiment outcomes."""

from typing import TypedDict

import numpy as np
from scipy.stats import beta as beta_dist

from config import (
    BAYESIAN_CONSIDER_THRESHOLD,
    BAYESIAN_KEEP_CONTROL_THRESHOLD,
    BAYESIAN_RANDOM_SEED,
    BAYESIAN_SAMPLES,
    BAYESIAN_SHIP_THRESHOLD,
    DEFAULT_LOSS_TOLERANCE_ABSOLUTE,
    DEFAULT_LOSS_TOLERANCE_RELATIVE,
)
from stats.frequentist import _validate_binary_inputs


class BayesianAnalysisResult(TypedDict):
    """Posterior summary for a two-variant Beta-Binomial comparison."""

    prob_b_wins: float
    expected_loss: float
    alpha_a: float
    beta_a: float
    alpha_b: float
    beta_b: float


def beta_binomial_analysis(
    successes_a: int,
    failures_a: int,
    successes_b: int,
    failures_b: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    random_seed: int | None = BAYESIAN_RANDOM_SEED,
) -> BayesianAnalysisResult:
    """Estimate posterior win probability and expected loss for two variants.

    ``random_seed`` makes the Monte Carlo estimate reproducible so the same
    input counts always produce the same recommendation. Pass ``None`` for a
    fresh draw each call.
    """
    _validate_binary_inputs(successes_a, failures_a, successes_b, failures_b)

    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Prior parameters must be greater than zero.")

    alpha_a = alpha_prior + successes_a
    beta_a = beta_prior + failures_a
    alpha_b = alpha_prior + successes_b
    beta_b = beta_prior + failures_b

    rng = np.random.default_rng(random_seed)
    samples_a = beta_dist.rvs(alpha_a, beta_a, size=BAYESIAN_SAMPLES, random_state=rng)
    samples_b = beta_dist.rvs(alpha_b, beta_b, size=BAYESIAN_SAMPLES, random_state=rng)

    prob_b_wins = float((samples_b > samples_a).mean())
    loss_if_ship_b = float(np.maximum(samples_a - samples_b, 0).mean())

    return {
        "prob_b_wins": prob_b_wins,
        "expected_loss": loss_if_ship_b,
        "alpha_a": float(alpha_a),
        "beta_a": float(beta_a),
        "alpha_b": float(alpha_b),
        "beta_b": float(beta_b),
    }


def resolve_loss_tolerance(
    loss_tolerance: float | None,
    baseline_for_relative_tolerance: float | None,
) -> float:
    """Resolve the effective loss tolerance from explicit, relative, or absolute defaults.

    The resolution order is:
    1. If ``loss_tolerance`` is given, use it directly.
    2. Otherwise, if ``baseline_for_relative_tolerance`` is given, use
       ``DEFAULT_LOSS_TOLERANCE_RELATIVE * baseline``. This scales the
       tolerance with the conversion rate, which is the right shape for
       baselines that span orders of magnitude.
    3. Otherwise fall back to ``DEFAULT_LOSS_TOLERANCE_ABSOLUTE``.
    """
    if loss_tolerance is not None:
        return loss_tolerance
    if baseline_for_relative_tolerance is not None:
        if baseline_for_relative_tolerance <= 0:
            raise ValueError("Baseline for relative tolerance must be positive.")
        return DEFAULT_LOSS_TOLERANCE_RELATIVE * baseline_for_relative_tolerance
    return DEFAULT_LOSS_TOLERANCE_ABSOLUTE


def get_decision_recommendation(
    prob_b_wins: float,
    expected_loss: float = 0.0,
    loss_tolerance: float | None = None,
    baseline_for_relative_tolerance: float | None = None,
) -> tuple[str, str]:
    """Translate posterior metrics into a shipping recommendation.

    A high win probability is not enough on its own. When expected loss
    remains above tolerance, the safer recommendation is to wait for
    more data.

    Loss tolerance scales with the baseline when ``baseline_for_relative_tolerance``
    is provided. This handles the wide range of baselines this tool sees
    (1% checkout conversion vs 60% landing-page CTR) without forcing the
    caller to re-tune the absolute threshold each time.
    """
    if not 0 <= prob_b_wins <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if expected_loss < 0:
        raise ValueError("Expected loss must be non-negative.")

    effective_tolerance = resolve_loss_tolerance(
        loss_tolerance=loss_tolerance,
        baseline_for_relative_tolerance=baseline_for_relative_tolerance,
    )
    if effective_tolerance <= 0:
        raise ValueError("Loss tolerance must be greater than zero.")

    loss_is_acceptable = expected_loss <= effective_tolerance

    if prob_b_wins > BAYESIAN_SHIP_THRESHOLD and loss_is_acceptable:
        return "Ship Variant B", "high"
    if prob_b_wins > BAYESIAN_CONSIDER_THRESHOLD and loss_is_acceptable:
        return "Consider shipping Variant B", "moderate"
    if prob_b_wins < BAYESIAN_KEEP_CONTROL_THRESHOLD:
        return "Keep Control", "high"
    return "Need more data", "uncertain"
