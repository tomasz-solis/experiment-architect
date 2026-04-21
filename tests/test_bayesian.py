"""Unit tests for Bayesian statistical functions."""

import pytest

from stats.bayesian import beta_binomial_analysis, get_decision_recommendation


class TestBetaBinomialAnalysis:
    """Tests for Beta-Binomial Bayesian analysis."""

    def test_variant_clearly_wins(self) -> None:
        result = beta_binomial_analysis(100, 900, 150, 850)
        assert result["prob_b_wins"] > 0.9
        assert result["expected_loss"] < 0.02
        assert result["alpha_a"] > 1
        assert result["beta_a"] > 1

    def test_control_clearly_wins(self) -> None:
        result = beta_binomial_analysis(150, 850, 100, 900)
        assert result["prob_b_wins"] < 0.2

    def test_no_difference(self) -> None:
        result = beta_binomial_analysis(100, 900, 100, 900)
        assert 0.45 < result["prob_b_wins"] < 0.55

    def test_custom_prior(self) -> None:
        result = beta_binomial_analysis(
            100,
            900,
            115,
            885,
            alpha_prior=10,
            beta_prior=90,
        )
        assert result["prob_b_wins"] > 0
        assert result["alpha_a"] == 110
        assert result["beta_a"] == 990


class TestDecisionRecommendation:
    """Tests for the loss-aware decision rule."""

    def test_high_confidence_ship(self) -> None:
        recommendation, confidence = get_decision_recommendation(0.97, expected_loss=0.001)
        assert "Ship" in recommendation
        assert confidence == "high"

    def test_moderate_confidence(self) -> None:
        recommendation, confidence = get_decision_recommendation(0.80, expected_loss=0.003)
        assert "Consider" in recommendation
        assert confidence == "moderate"

    def test_high_confidence_keep_control(self) -> None:
        recommendation, confidence = get_decision_recommendation(0.15, expected_loss=0.001)
        assert "Keep Control" in recommendation
        assert confidence == "high"

    def test_uncertain(self) -> None:
        recommendation, confidence = get_decision_recommendation(0.50, expected_loss=0.003)
        assert "Need more data" in recommendation
        assert confidence == "uncertain"

    def test_high_probability_with_large_expected_loss_waits(self) -> None:
        recommendation, confidence = get_decision_recommendation(0.97, expected_loss=0.02)
        assert recommendation == "Need more data"
        assert confidence == "uncertain"

    def test_edge_cases(self) -> None:
        _, confidence_at_96 = get_decision_recommendation(0.96, expected_loss=0.001)
        assert confidence_at_96 == "high"

        _, confidence_at_95 = get_decision_recommendation(0.95, expected_loss=0.001)
        assert confidence_at_95 == "moderate"

        _, confidence_at_76 = get_decision_recommendation(0.76, expected_loss=0.001)
        assert confidence_at_76 == "moderate"

        _, confidence_at_75 = get_decision_recommendation(0.75, expected_loss=0.001)
        assert confidence_at_75 == "uncertain"

    def test_negative_expected_loss_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_decision_recommendation(0.9, expected_loss=-0.1)
