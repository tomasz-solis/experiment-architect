"""Pure presentation helpers for the review UI.

These functions contain no Streamlit calls and no I/O, so they live outside
``app.py`` (which executes Streamlit at import time) and can be unit tested
directly. ``app.py`` imports them to build the hero and summary content.
"""

from __future__ import annotations

from typing import TypedDict


class SummaryCard(TypedDict):
    """Payload for one card in the hero summary row."""

    label: str
    value: str
    meta: str
    tone: str
    anchor: bool


SIDEBAR_TIPS: dict[str, str] = {
    "Experiment design": "Size the claim before you size the excitement.",
    "Manual result read": "A winner with a weak stop rule is not a winner yet.",
    "Raw CSV audit": "Mapped columns are still assumptions until you inspect the frame.",
    "Causal fallback": "When randomization fails, the assumption becomes the product.",
}


def first_sentence(text: str) -> str:
    """Return the first sentence-like fragment of a string for compact card copy."""
    stripped = text.strip()
    if not stripped:
        return ""
    for delimiter in (". ", "; ", " - "):
        if delimiter in stripped:
            return stripped.split(delimiter, 1)[0].strip() + ("" if delimiter == " - " else ".")
    return stripped


def duration_tone(days: int) -> str:
    """Return a color tone for an experiment duration in days.

    Tracks the same buckets the sanity copy uses: up to four weeks reads as
    healthy (mint), up to eight weeks as a caution (amber), longer as a
    blocking concern (red).
    """
    if days <= 28:
        return "mint"
    if days <= 56:
        return "amber"
    return "red"


def sidebar_tip(review_focus: str) -> str:
    """Return a concise sidebar tip for the selected review lens."""
    return SIDEBAR_TIPS[review_focus]


def build_card(
    label: str,
    value: str,
    meta: str,
    tone: str,
    anchor: bool = False,
) -> SummaryCard:
    """Create a summary-card payload for the hero row."""
    return {
        "label": label,
        "value": value,
        "meta": meta,
        "tone": tone,
        "anchor": anchor,
    }
