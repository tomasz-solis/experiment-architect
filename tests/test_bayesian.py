"""Unit tests for Bayesian statistical functions."""

import pytest
from stats.bayesian import beta_binomial_analysis, get_decision_recommendation


class TestBetaBinomialAnalysis:
    """Tests for Beta-Binomial Bayesian analysis."""

    def test_variant_clearly_wins(self):
        # B has 50% higher conversion (clear win)
        result = beta_binomial_analysis(100, 900, 150, 850)
        assert result['prob_b_wins'] > 0.9
        assert result['expected_loss'] < 0.02
        assert result['alpha_a'] > 1
        assert result['beta_a'] > 1

    def test_control_clearly_wins(self):
        # A has higher conversion
        result = beta_binomial_analysis(150, 850, 100, 900)
        assert result['prob_b_wins'] < 0.2

    def test_no_difference(self):
        # Identical groups
        result = beta_binomial_analysis(100, 900, 100, 900)
        assert 0.45 < result['prob_b_wins'] < 0.55

    def test_custom_prior(self):
        # Test with non-uniform prior
        result = beta_binomial_analysis(
            100, 900, 115, 885,
            alpha_prior=10,
            beta_prior=90
        )
        assert result['prob_b_wins'] > 0
        assert result['alpha_a'] == 110  # 10 + 100
        assert result['beta_a'] == 990   # 90 + 900


class TestDecisionRecommendation:
    """Tests for decision recommendation logic."""

    def test_high_confidence_ship(self):
        recommendation, confidence = get_decision_recommendation(0.97)
        assert "Ship" in recommendation
        assert confidence == "high"

    def test_moderate_confidence(self):
        recommendation, confidence = get_decision_recommendation(0.80)
        assert "Consider" in recommendation
        assert confidence == "moderate"

    def test_high_confidence_keep_control(self):
        recommendation, confidence = get_decision_recommendation(0.15)
        assert "Keep Control" in recommendation
        assert confidence == "high"

    def test_uncertain(self):
        recommendation, confidence = get_decision_recommendation(0.50)
        assert "Need more data" in recommendation
        assert confidence == "uncertain"

    def test_edge_cases(self):
        # Test boundary conditions
        _, conf_96 = get_decision_recommendation(0.96)
        assert conf_96 == "high"

        _, conf_95 = get_decision_recommendation(0.95)
        assert conf_95 == "moderate"

        _, conf_76 = get_decision_recommendation(0.76)
        assert conf_76 == "moderate"

        _, conf_75 = get_decision_recommendation(0.75)
        assert conf_75 == "uncertain"
