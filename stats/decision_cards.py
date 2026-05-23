"""Pure card-content builders for the lens snapshots.

These functions take pre-computed analysis results and return structured
content for the UI to render. They are deliberately Streamlit-free so the
decision logic stays testable and reusable.

Card payload shape: ``{"value": str, "meta": str, "tone": str}``. The tone
is one of "mint" (positive), "amber" (caution), "red" (blocking), or
"blue" (informational), matching the UI's design tokens.
"""

from __future__ import annotations

from typing import Literal, TypedDict

Tone = Literal["mint", "amber", "red", "blue"]


class CardPayload(TypedDict):
    """Structured content for a single summary card."""

    value: str
    meta: str
    tone: Tone


def build_manual_frequentist_card(
    p_value: float,
    adjusted_alpha: float,
    has_srm: bool,
    peeked_early: bool,
    chi_square_valid: bool,
) -> CardPayload:
    """Build the frequentist-read card for the manual lens.

    The card is "Decision-ready" only when the p-value beats the adjusted
    alpha AND the structural diagnostics all clear: no sample-ratio
    mismatch, no early peeking, and the chi-squared cell counts are large
    enough for the normal approximation to hold.
    """
    is_significant = p_value < adjusted_alpha
    structure_clean = not has_srm and not peeked_early and chi_square_valid

    if is_significant and structure_clean:
        return {
            "value": "Decision-ready",
            "meta": f"p={p_value:.4f} and the structure is clean.",
            "tone": "mint",
        }
    if is_significant:
        return {
            "value": "Usable with caution",
            "meta": f"p={p_value:.4f}, but the read has structural caveats.",
            "tone": "amber",
        }
    return {
        "value": "Need more data",
        "meta": f"p={p_value:.4f} at alpha {adjusted_alpha:.4f}.",
        "tone": "amber",
    }


def build_weakest_signal_card(
    has_srm: bool,
    chi_square_valid: bool,
    peeked_early: bool,
    alpha_adjusted: bool,
    adjusted_alpha: float,
) -> CardPayload:
    """Identify the weakest structural signal in a manual-read analysis.

    Order matters: SRM is the most disqualifying finding because it
    invalidates the randomization assumption. Low expected counts come
    next because they break the chi-squared null distribution. Peeking
    and multiple-metric adjustment are softer concerns.
    """
    if has_srm:
        return {
            "value": "Sample ratio mismatch",
            "meta": "The randomization split does not look like 50/50.",
            "tone": "red",
        }
    if not chi_square_valid:
        return {
            "value": "Low expected cell counts",
            "meta": "Binary significance is fragile with sparse cells.",
            "tone": "amber",
        }
    if peeked_early:
        return {
            "value": "Early peeking",
            "meta": "The nominal p-value is optimistic without a stop rule.",
            "tone": "amber",
        }
    if alpha_adjusted:
        return {
            "value": "Multiple primary metrics",
            "meta": f"The decision threshold tightened to {adjusted_alpha:.4f}.",
            "tone": "amber",
        }
    return {
        "value": "No structural red flag",
        "meta": "The read is clean enough to interpret directly.",
        "tone": "mint",
    }


def build_bayesian_card(
    recommendation: str,
    confidence: str,
    expected_loss: float,
) -> CardPayload:
    """Build the Bayesian-read card from a decision recommendation.

    The tone reflects the confidence level: "high" confidence with a
    positive recommendation maps to mint, everything else to amber so
    the user pauses before acting.
    """
    tone: Tone = "mint" if confidence == "high" else "amber"
    return {
        "value": recommendation,
        "meta": f"Expected loss {expected_loss:.3%}.",
        "tone": tone,
    }


def build_count_mismatch_card() -> CardPayload:
    """Card content for the input-mismatch case where counts are inconsistent."""
    return {
        "value": "Count mismatch",
        "meta": "Conversions cannot exceed visitors.",
        "tone": "red",
    }


def build_input_mismatch_summary() -> dict[str, object]:
    """Hero payload for the manual lens when the raw counts are inconsistent.

    Returned shape matches the snapshot dict the UI expects, so the caller
    can return it directly without further assembly.
    """
    return {
        "kicker": "Editorial experiment review",
        "title": "Read the result, but audit the counts first.",
        "body": (
            "This lens compares significance, expected loss, and structural "
            "warnings in one place. Right now the raw counts are inconsistent, "
            "so the review stops at the input layer."
        ),
        "pills": ["Manual read", "Counts only", "Input mismatch"],
    }
