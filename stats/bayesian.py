"""Bayesian statistical helpers for binary experiment outcomes."""

from typing import Tuple

import numpy as np
from scipy.stats import beta as beta_dist

from config import BAYESIAN_SAMPLES
from stats.frequentist import _validate_binary_inputs


def beta_binomial_analysis(
    successes_a: int,
    failures_a: int,
    successes_b: int,
    failures_b: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict[str, float]:
    """Estimate posterior win probability and expected loss for two variants."""
    _validate_binary_inputs(successes_a, failures_a, successes_b, failures_b)

    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Prior parameters must be greater than zero.")

    alpha_a = alpha_prior + successes_a
    beta_a = beta_prior + failures_a
    alpha_b = alpha_prior + successes_b
    beta_b = beta_prior + failures_b

    samples_a = beta_dist.rvs(alpha_a, beta_a, size=BAYESIAN_SAMPLES)
    samples_b = beta_dist.rvs(alpha_b, beta_b, size=BAYESIAN_SAMPLES)

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


def get_decision_recommendation(
    prob_b_wins: float,
    expected_loss: float = 0.0,
    loss_tolerance: float = 0.005,
) -> Tuple[str, str]:
    """Translate posterior metrics into a shipping recommendation.

    A high win probability is not enough on its own. When expected loss remains
    above tolerance, the safer recommendation is to wait for more data.
    """
    if not 0 <= prob_b_wins <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if expected_loss < 0:
        raise ValueError("Expected loss must be non-negative.")
    if loss_tolerance <= 0:
        raise ValueError("Loss tolerance must be greater than zero.")

    loss_is_acceptable = expected_loss <= loss_tolerance

    if prob_b_wins > 0.95 and loss_is_acceptable:
        return "Ship Variant B", "high"
    if prob_b_wins > 0.75 and loss_is_acceptable:
        return "Consider shipping Variant B", "moderate"
    if prob_b_wins < 0.25:
        return "Keep Control", "high"
    return "Need more data", "uncertain"
