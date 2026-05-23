"""Tests for the pure card-content builders.

These cover every branch in the four card builders so the decision logic
extracted from app.py cannot regress silently.
"""

from __future__ import annotations

import pytest

from stats.decision_cards import (
    build_bayesian_card,
    build_count_mismatch_card,
    build_input_mismatch_summary,
    build_manual_frequentist_card,
    build_weakest_signal_card,
)


class TestManualFrequentistCard:
    """Coverage for the frequentist-read card in the manual lens."""

    def test_decision_ready_when_significant_and_clean(self) -> None:
        card = build_manual_frequentist_card(
            p_value=0.02,
            adjusted_alpha=0.05,
            has_srm=False,
            peeked_early=False,
            chi_square_valid=True,
        )
        assert card["value"] == "Decision-ready"
        assert card["tone"] == "mint"
        assert "0.0200" in card["meta"]

    def test_usable_with_caution_when_significant_but_peeked(self) -> None:
        card = build_manual_frequentist_card(
            p_value=0.02,
            adjusted_alpha=0.05,
            has_srm=False,
            peeked_early=True,
            chi_square_valid=True,
        )
        assert card["value"] == "Usable with caution"
        assert card["tone"] == "amber"

    def test_usable_with_caution_when_significant_but_srm(self) -> None:
        card = build_manual_frequentist_card(
            p_value=0.02,
            adjusted_alpha=0.05,
            has_srm=True,
            peeked_early=False,
            chi_square_valid=True,
        )
        assert card["value"] == "Usable with caution"
        assert card["tone"] == "amber"

    def test_usable_with_caution_when_significant_but_chi_invalid(self) -> None:
        card = build_manual_frequentist_card(
            p_value=0.02,
            adjusted_alpha=0.05,
            has_srm=False,
            peeked_early=False,
            chi_square_valid=False,
        )
        assert card["value"] == "Usable with caution"

    def test_need_more_data_when_not_significant(self) -> None:
        card = build_manual_frequentist_card(
            p_value=0.20,
            adjusted_alpha=0.05,
            has_srm=False,
            peeked_early=False,
            chi_square_valid=True,
        )
        assert card["value"] == "Need more data"
        assert card["tone"] == "amber"

    def test_p_at_exact_boundary_is_not_significant(self) -> None:
        """Strict less-than: p == adjusted_alpha does not pass."""
        card = build_manual_frequentist_card(
            p_value=0.05,
            adjusted_alpha=0.05,
            has_srm=False,
            peeked_early=False,
            chi_square_valid=True,
        )
        assert card["value"] == "Need more data"

    def test_adjusted_alpha_below_p_value_does_not_pass(self) -> None:
        """With Bonferroni adjustment, what was significant may no longer be."""
        card = build_manual_frequentist_card(
            p_value=0.04,
            adjusted_alpha=0.025,
            has_srm=False,
            peeked_early=False,
            chi_square_valid=True,
        )
        assert card["value"] == "Need more data"


class TestWeakestSignalCard:
    """Coverage for the weakest-signal identification."""

    def test_srm_takes_priority(self) -> None:
        card = build_weakest_signal_card(
            has_srm=True,
            chi_square_valid=False,
            peeked_early=True,
            alpha_adjusted=True,
            adjusted_alpha=0.025,
        )
        assert card["value"] == "Sample ratio mismatch"
        assert card["tone"] == "red"

    def test_chi_invalid_when_no_srm(self) -> None:
        card = build_weakest_signal_card(
            has_srm=False,
            chi_square_valid=False,
            peeked_early=True,
            alpha_adjusted=True,
            adjusted_alpha=0.025,
        )
        assert card["value"] == "Low expected cell counts"
        assert card["tone"] == "amber"

    def test_peeking_when_no_structural_issues(self) -> None:
        card = build_weakest_signal_card(
            has_srm=False,
            chi_square_valid=True,
            peeked_early=True,
            alpha_adjusted=True,
            adjusted_alpha=0.025,
        )
        assert card["value"] == "Early peeking"
        assert card["tone"] == "amber"

    def test_multiple_metrics_when_only_alpha_adjusted(self) -> None:
        card = build_weakest_signal_card(
            has_srm=False,
            chi_square_valid=True,
            peeked_early=False,
            alpha_adjusted=True,
            adjusted_alpha=0.025,
        )
        assert card["value"] == "Multiple primary metrics"
        assert "0.0250" in card["meta"]
        assert card["tone"] == "amber"

    def test_clean_when_nothing_flagged(self) -> None:
        card = build_weakest_signal_card(
            has_srm=False,
            chi_square_valid=True,
            peeked_early=False,
            alpha_adjusted=False,
            adjusted_alpha=0.05,
        )
        assert card["value"] == "No structural red flag"
        assert card["tone"] == "mint"


class TestBayesianCard:
    """Coverage for the Bayesian-read card content builder."""

    def test_high_confidence_is_mint(self) -> None:
        card = build_bayesian_card(
            recommendation="Ship Variant B",
            confidence="high",
            expected_loss=0.001,
        )
        assert card["tone"] == "mint"
        assert card["value"] == "Ship Variant B"
        assert "0.100%" in card["meta"]

    def test_moderate_confidence_is_amber(self) -> None:
        card = build_bayesian_card(
            recommendation="Consider shipping Variant B",
            confidence="moderate",
            expected_loss=0.003,
        )
        assert card["tone"] == "amber"

    def test_uncertain_confidence_is_amber(self) -> None:
        card = build_bayesian_card(
            recommendation="Need more data",
            confidence="uncertain",
            expected_loss=0.005,
        )
        assert card["tone"] == "amber"


class TestStaticCards:
    """Coverage for the constant-content card builders."""

    def test_count_mismatch_card_shape(self) -> None:
        card = build_count_mismatch_card()
        assert card["value"] == "Count mismatch"
        assert card["tone"] == "red"

    def test_input_mismatch_summary_has_required_keys(self) -> None:
        summary = build_input_mismatch_summary()
        assert "kicker" in summary
        assert "title" in summary
        assert "body" in summary
        assert "pills" in summary
        assert isinstance(summary["pills"], list)
