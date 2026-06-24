"""Unit tests for the pure presentation helpers in ui/formatting.py."""

from __future__ import annotations

import pytest

from ui.formatting import (
    SIDEBAR_TIPS,
    build_card,
    duration_tone,
    first_sentence,
    sidebar_tip,
)


class TestFirstSentence:
    """first_sentence trims card copy to a single leading clause."""

    def test_empty_string_returns_empty(self) -> None:
        assert first_sentence("   ") == ""

    def test_splits_on_period_and_keeps_terminator(self) -> None:
        assert first_sentence("Traffic is fine. The rest is noise.") == "Traffic is fine."

    def test_splits_on_semicolon(self) -> None:
        assert first_sentence("Budget covers it; barely") == "Budget covers it."

    def test_splits_on_dash_without_adding_period(self) -> None:
        # The " - " delimiter is treated as a soft break, not a sentence end.
        assert first_sentence("Need 40k visitors - extend the test") == "Need 40k visitors"

    def test_no_delimiter_returns_stripped_text(self) -> None:
        assert first_sentence("  one clean clause  ") == "one clean clause"


class TestDurationTone:
    """duration_tone buckets experiment length into design-system tones."""

    @pytest.mark.parametrize(
        ("days", "expected"),
        [
            (1, "mint"),
            (28, "mint"),
            (29, "amber"),
            (56, "amber"),
            (57, "red"),
            (365, "red"),
        ],
    )
    def test_buckets(self, days: int, expected: str) -> None:
        assert duration_tone(days) == expected

    def test_boundaries_are_inclusive_on_the_healthy_side(self) -> None:
        # The four- and eight-week thresholds belong to the calmer tone.
        assert duration_tone(28) == "mint"
        assert duration_tone(56) == "amber"


class TestSidebarTip:
    """sidebar_tip maps each review lens to a one-line tip."""

    @pytest.mark.parametrize("focus", list(SIDEBAR_TIPS))
    def test_returns_tip_for_every_known_lens(self, focus: str) -> None:
        assert sidebar_tip(focus) == SIDEBAR_TIPS[focus]
        assert sidebar_tip(focus)

    def test_unknown_lens_raises(self) -> None:
        with pytest.raises(KeyError):
            sidebar_tip("not a lens")


class TestBuildCard:
    """build_card assembles the summary-card payload the UI renders."""

    def test_defaults_to_non_anchor(self) -> None:
        card = build_card("Estimated duration", "21 days", "At current traffic.", "mint")
        assert card == {
            "label": "Estimated duration",
            "value": "21 days",
            "meta": "At current traffic.",
            "tone": "mint",
            "anchor": False,
        }

    def test_anchor_flag_is_passed_through(self) -> None:
        card = build_card("Review lens", "A/B design", "Plan first.", "blue", anchor=True)
        assert card["anchor"] is True
