"""Bayesian statistical analysis functions."""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import beta
from config import BAYESIAN_SAMPLES


def beta_binomial_analysis(
    successes_a: int,
    failures_a: int,
    successes_b: int,
    failures_b: int,
    alpha_prior: float = 1,
    beta_prior: float = 1
) -> Dict[str, float]:
    """Run Bayesian analysis using Beta-Binomial conjugate prior.

    Args:
        successes_a: Number of successes in group A
        failures_a: Number of failures in group A
        successes_b: Number of successes in group B
        failures_b: Number of failures in group B
        alpha_prior: Prior alpha parameter (default 1 for uniform)
        beta_prior: Prior beta parameter (default 1 for uniform)

    Returns:
        dict: Analysis results including probability and expected loss
    """
    # Posterior distributions
    alpha_a = alpha_prior + successes_a
    beta_a = beta_prior + failures_a
    alpha_b = alpha_prior + successes_b
    beta_b = beta_prior + failures_b

    # Sample from posteriors
    samples_a = beta.rvs(alpha_a, beta_a, size=BAYESIAN_SAMPLES)
    samples_b = beta.rvs(alpha_b, beta_b, size=BAYESIAN_SAMPLES)

    # P(B > A)
    prob_b_wins = (samples_b > samples_a).mean()

    # Expected loss if you ship variant B
    loss_if_ship_b = np.maximum(samples_a - samples_b, 0).mean()

    return {
        "prob_b_wins": prob_b_wins,
        "expected_loss": loss_if_ship_b,
        "alpha_a": alpha_a,
        "beta_a": beta_a,
        "alpha_b": alpha_b,
        "beta_b": beta_b
    }


def get_decision_recommendation(prob_b_wins: float) -> Tuple[str, str]:
    """Get decision recommendation based on probability.

    Returns:
        tuple: (recommendation, confidence_level)
    """
    if prob_b_wins > 0.95:
        return "Ship Variant B", "high"
    elif prob_b_wins > 0.75:
        return "Consider shipping Variant B", "moderate"
    elif prob_b_wins < 0.25:
        return "Keep Control", "high"
    else:
        return "Need more data", "uncertain"
