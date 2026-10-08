"""Reusable Streamlit components for showing experiment results and layout chrome."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from config import ALPHA
from stats.bayesian import BayesianAnalysisResult
from stats.frequentist import AttritionResult, FrequentistTestResult, SRMResult
from stats.prereg import GuardrailReading, PreRegistration, ReadoutSummary, VerificationRow
from ui.formatting import SummaryCard

_THEME_TOKENS_PATH = Path(__file__).parent / "theme-tokens.css"


def inject_app_styles() -> None:
    """Inject the editorial design system used across the app."""
    tokens = _THEME_TOKENS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{tokens}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            :root {
                /* Local names map onto the shared design tokens (ui/theme-tokens.css). */
                --bg: var(--ds-bg);
                --ink: var(--ds-ink);
                --muted: var(--ds-muted);
                --blue: var(--ds-blue);
                --mint: var(--ds-mint);
                --amber: var(--ds-amber);
                --red: var(--ds-red);
                /* Deep forms for value text on light cards. The bright tones above are
                   fills and marks only; these meet WCAG AA against the light --card-bg.
                   They now live in the shared token file, so all three Lab apps get them. */
                --blue-text: var(--ds-blue-strong);
                --mint-text: var(--ds-mint-deep);
                --amber-text: var(--ds-amber-deep);
                --red-text: var(--ds-red-deep);
                --sidebar-top: var(--ds-sidebar-top);
                --sidebar-bottom: var(--ds-sidebar-bottom);
                --card-border: rgba(112, 128, 156, 0.16);
                --card-bg: rgba(255, 255, 255, 0.72);
                /* Flat at rest: resting surfaces separate with the hairline border, not depth.
                   Only the hero is allowed to sit visibly above the page.
                   See DESIGN.md, The Flat-At-Rest Rule. */
                --card-shadow: none;
                --hero-shadow: 0 30px 72px rgba(12, 16, 24, 0.22);
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 8%, rgba(79, 109, 255, 0.12), transparent 34%),
                    radial-gradient(circle at 88% 10%, rgba(30, 207, 155, 0.10), transparent 28%),
                    radial-gradient(circle at 50% 100%, rgba(79, 109, 255, 0.08), transparent 34%),
                    var(--bg);
                color: var(--ink);
                font-family: var(--ds-font-sans);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stAppViewContainer"] {
                background: transparent;
            }

            [data-testid="stMainBlockContainer"] {
                max-width: 1260px;
                padding-top: 2.1rem;
                padding-bottom: 5rem;
            }

            section[data-testid="stSidebar"] {
                background:
                    radial-gradient(circle at 18% 12%, rgba(79, 109, 255, 0.24), transparent 24%),
                    radial-gradient(circle at 82% 18%, rgba(30, 207, 155, 0.18), transparent 20%),
                    linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-bottom) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.06);
            }

            section[data-testid="stSidebar"] * {
                color: #eef3fb;
            }

            section[data-testid="stSidebar"] [data-baseweb="select"] > div,
            section[data-testid="stSidebar"] [data-baseweb="input"] > div,
            section[data-testid="stSidebar"] .stNumberInput > div > div,
            section[data-testid="stSidebar"] textarea,
            section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 18px;
            }

            section[data-testid="stSidebar"] .stButton > button {
                background: rgba(255, 255, 255, 0.06);
                color: #f3f7fb;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }

            section[data-testid="stSidebar"] .stButton > button:hover {
                color: #ffffff;
                border-color: rgba(255, 255, 255, 0.28);
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(79, 109, 255, 0.18);
                background: linear-gradient(180deg, rgba(79, 109, 255, 0.10), rgba(79, 109, 255, 0.06));
                color: var(--ink);
                padding: 0.7rem 1.15rem;
                font-weight: 600;
                transition: transform 140ms ease-out, box-shadow 140ms ease-out, border-color 140ms ease-out;
            }

            /* The lift is the hover state, not the resting state. */
            .stButton > button:hover {
                border-color: rgba(79, 109, 255, 0.34);
                color: var(--blue-text);
                transform: translateY(-1px);
                box-shadow: 0 12px 24px rgba(79, 109, 255, 0.12);
            }

            [data-baseweb="select"] > div,
            [data-baseweb="input"] > div,
            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                border-radius: 18px;
            }

            /* Sentence-case labels. An uppercase tracked eyebrow on every form field
               makes the controls shout; see DESIGN.md, The Eyebrow Rule. */
            .stSelectbox label,
            .stRadio label,
            .stNumberInput label,
            .stTextInput label,
            .stTextArea label,
            .stFileUploader label {
                color: var(--ink);
                font-size: 0.82rem;
                font-weight: 600;
            }

            div[data-testid="stMetric"] {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 1rem 1rem 0.9rem;
                box-shadow: var(--card-shadow);
            }

            div[data-testid="stMetric"] label {
                font-size: 0.82rem;
                font-weight: 600;
                color: var(--muted);
            }

            div[data-testid="stMetricValue"] {
                color: var(--ink);
            }

            div[data-testid="stExpander"] {
                border: 1px solid var(--card-border);
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.64);
                box-shadow: var(--card-shadow);
                overflow: hidden;
            }

            div[data-testid="stExpander"] details summary p {
                font-weight: 700;
                color: var(--ink);
            }

            div[data-testid="stAlert"] {
                border-radius: 24px;
                border: 1px solid rgba(16, 19, 26, 0.06);
            }

            [data-testid="stFileUploader"] section {
                border-radius: 24px;
                border: 1px dashed rgba(79, 109, 255, 0.28);
                background: rgba(255, 255, 255, 0.58);
            }

            .editorial-sidebar {
                padding: 1.2rem 1rem 1rem;
                margin-bottom: 1rem;
                border-radius: 28px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
            }

            .editorial-kicker {
                margin: 0 0 0.55rem 0;
                color: rgba(241, 246, 255, 0.72);
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
            }

            .editorial-sidebar h2,
            .editorial-hero h1,
            .summary-value,
            .signal-header h2 {
                font-family: var(--ds-font-sans);
                font-variant-numeric: tabular-nums;
            }

            .editorial-sidebar h2 {
                margin: 0;
                font-size: 1.7rem;
                letter-spacing: -0.03em;
                line-height: 1.05;
            }

            .editorial-sidebar p {
                margin: 0.85rem 0 0 0;
                color: rgba(235, 242, 255, 0.76);
                line-height: 1.55;
            }

            .sidebar-chip {
                display: inline-block;
                margin-top: 0.9rem;
                padding: 0.48rem 0.78rem;
                border-radius: 999px;
                background: rgba(30, 207, 155, 0.10);
                border: 1px solid rgba(30, 207, 155, 0.18);
                color: #d5fff3;
                font-size: 0.78rem;
                font-weight: 600;
            }

            .editorial-hero {
                position: relative;
                overflow: hidden;
                padding: 2rem 2.2rem 2.05rem;
                border-radius: 28px;
                color: #f6f8fc;
                background:
                    radial-gradient(circle at 12% 18%, rgba(79, 109, 255, 0.28), transparent 22%),
                    radial-gradient(circle at 88% 20%, rgba(30, 207, 155, 0.20), transparent 20%),
                    linear-gradient(135deg, #0f1622 0%, #141c29 52%, #101926 100%);
                box-shadow: var(--hero-shadow);
            }

            .editorial-hero::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), transparent 32%);
                pointer-events: none;
            }

            .hero-title {
                margin: 0;
                max-width: 760px;
                font-size: clamp(2.2rem, 4vw, 3.3rem);
                line-height: 0.98;
                letter-spacing: -0.055em;
            }

            .hero-body {
                margin: 0.95rem 0 0;
                max-width: 760px;
                color: rgba(242, 246, 251, 0.78);
                font-size: 1rem;
                line-height: 1.68;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-top: 1.3rem;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.52rem 0.82rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.10);
                color: #f1f6ff;
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.01em;
            }

            .summary-grid,
            .empty-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }

            .empty-grid {
                /* Four signal cards read as a balanced 2x2 block. A three-across
                   grid leaves the fourth card stranded on its own row. */
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 1.25rem;
            }

            .summary-card,
            .empty-card,
            .section-note {
                border-radius: 24px;
                border: 1px solid var(--card-border);
                background: var(--card-bg);
                box-shadow: var(--card-shadow);
            }

            .summary-card {
                padding: 1.15rem 1.15rem 1rem;
            }

            /* The anchor card earns emphasis from the dark fill and the periwinkle
               border, not from depth. Flat at rest applies to featured cards too. */
            .summary-card.anchor {
                background:
                    linear-gradient(180deg, rgba(16, 19, 26, 0.96), rgba(18, 24, 36, 0.90));
                border-color: rgba(79, 109, 255, 0.24);
            }

            .summary-label,
            .empty-label,
            .signal-label,
            .note-label {
                margin: 0;
                font-size: 0.82rem;
                font-weight: 600;
                color: var(--muted);
            }

            .summary-card.anchor .summary-label {
                color: rgba(235, 242, 255, 0.62);
            }

            .summary-value {
                margin: 0.55rem 0 0;
                font-size: 1.45rem;
                line-height: 1.05;
                letter-spacing: -0.04em;
                color: var(--ink);
            }

            .summary-card.anchor .summary-value {
                color: #f4f7fc;
            }

            .summary-meta,
            .empty-body,
            .signal-body,
            .note-body {
                margin: 0.58rem 0 0;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .summary-card.anchor .summary-meta {
                color: rgba(235, 242, 255, 0.74);
            }

            .tone-blue {
                color: var(--blue-text);
            }

            .tone-mint {
                color: var(--mint-text);
            }

            .tone-amber {
                color: var(--amber-text);
            }

            .tone-red {
                color: var(--red-text);
            }

            .signal-header {
                margin: 2.4rem 0 1.1rem;
            }

            .signal-header h2 {
                margin: 0.45rem 0 0;
                font-size: clamp(1.55rem, 2.4vw, 2.2rem);
                line-height: 1.04;
                letter-spacing: -0.04em;
                color: var(--ink);
            }

            .section-note {
                padding: 1rem 1.1rem;
                margin-bottom: 1.05rem;
            }

            .note-body {
                margin-top: 0.45rem;
            }

            .editorial-rule {
                border: none;
                height: 1px;
                margin: 2.3rem 0 0.2rem;
                background: linear-gradient(90deg, transparent, rgba(100, 108, 121, 0.30), transparent);
            }

            @media (max-width: 1080px) {
                .summary-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }

                .empty-grid {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 720px) {
                .editorial-hero {
                    padding: 1.55rem;
                }

                .summary-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_intro(
    title: str,
    body: str,
    ai_enabled: bool,
    provider: str,
) -> None:
    """Render the dark sidebar brand block."""
    provider_state = f"AI mapping ready via {provider.upper()}" if ai_enabled else "AI mapping disabled"
    st.markdown(
        f'<div class="editorial-sidebar">'
        f'<p class="editorial-kicker">Experiment review</p>'
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(body)}</p>"
        f'<span class="sidebar-chip">{escape(provider_state)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_hero_card(
    kicker: str,
    title: str,
    body: str,
    pills: list[str],
) -> None:
    """Render the large dark hero card at the top of the page."""
    pill_markup = "".join(f'<span class="pill">{escape(pill)}</span>' for pill in pills)
    st.markdown(
        f'<div class="editorial-hero">'
        f'<p class="editorial-kicker">{escape(kicker)}</p>'
        f'<h1 class="hero-title">{escape(title)}</h1>'
        f'<p class="hero-body">{escape(body)}</p>'
        f'<div class="pill-row">{pill_markup}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_summary_cards(cards: list[SummaryCard]) -> None:
    """Render the short summary card row below the hero."""
    card_markup: list[str] = []
    for index, card in enumerate(cards):
        tone = str(card.get("tone", "blue"))
        is_anchor = bool(card.get("anchor", index == 0))
        card_markup.append(
            f'<div class="summary-card{" anchor" if is_anchor else ""}">'
            f'<p class="summary-label">{escape(str(card["label"]))}</p>'
            f'<p class="summary-value tone-{escape(tone)}">{escape(str(card["value"]))}</p>'
            f'<p class="summary-meta">{escape(str(card["meta"]))}</p>'
            f"</div>"
        )

    st.markdown(f'<div class="summary-grid">{"".join(card_markup)}</div>', unsafe_allow_html=True)


def render_empty_state_cards(cards: list[dict[str, str]]) -> None:
    """Render the top-of-page explainer cards used in the empty state."""
    markup = []
    for card in cards:
        markup.append(
            f'<div class="empty-card section-note">'
            f'<p class="empty-label">{escape(card["label"])}</p>'
            f'<p class="summary-value">{escape(card["title"])}</p>'
            f'<p class="empty-body">{escape(card["body"])}</p>'
            f"</div>"
        )

    st.markdown(f'<div class="empty-grid">{"".join(markup)}</div>', unsafe_allow_html=True)


def render_signal_header(signal: str, title: str, body: str) -> None:
    """Render the editorial section header used before each major section."""
    st.markdown(
        f'<div class="signal-header">'
        f'<p class="signal-label">{escape(signal)}</p>'
        f"<h2>{escape(title)}</h2>"
        f'<p class="signal-body">{escape(body)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_section_note(label: str, body: str) -> None:
    """Render a short glass-style note above a widget cluster."""
    st.markdown(
        f'<div class="section-note">'
        f'<p class="note-label">{escape(label)}</p>'
        f'<p class="note-body">{escape(body)}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_section_rule() -> None:
    """Render a visible separator between major page sections."""
    st.markdown('<hr class="editorial-rule" />', unsafe_allow_html=True)


def show_data_quality(df: pd.DataFrame) -> None:
    """Render a small data quality summary for a DataFrame."""
    st.markdown("**Data Quality**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Missing Values", int(df.isnull().sum().sum()))
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("View full dataset"):
        st.dataframe(df)


def show_srm_warning(result: SRMResult, stage_label: str = "") -> None:
    """Warn when the observed split deviates from the split the test intended.

    ``stage_label`` (e.g. "Before cleaning", "After cleaning") tells two SRM
    checks apart when both are shown on the same page; without it, a raw-data
    mismatch and a cleaning-introduced one read as the identical sentence
    twice with no way to see which is which.
    """
    if not result["has_mismatch"]:
        return
    prefix = f"{stage_label}: " if stage_label else ""
    st.warning(
        f"{prefix}Sample ratio mismatch: expected {result['expected_share']:.0%} of users in the "
        f"variant, got {result['observed_share']:.1%} (p={result['p_value']:.4f}). This means "
        "users are being dropped unevenly somewhere in assignment, eligibility, or logging. "
        "Do not read the effect until you know the cause."
    )


def show_attrition_warning(result: AttritionResult) -> None:
    """Warn when data cleaning removed rows unevenly between the two arms."""
    if not result["has_differential_attrition"]:
        return
    st.warning(
        f"Differential attrition: cleaning dropped {result['dropped_share_a']:.1%} of arm A's "
        f"rows vs {result['dropped_share_b']:.1%} of arm B's (p={result['p_value']:.4f}). The "
        "two groups may no longer be comparable, because the treatment itself could be "
        "affecting who has usable data."
    )


def show_frequentist_results(
    test_results: FrequentistTestResult,
    ci_lower: float,
    ci_upper: float,
    mean_a: float,
    mean_b: float,
    groups: list[str],
    alpha_threshold: float = ALPHA,
) -> None:
    """Render confidence intervals, effect size, p-value, and verdict."""
    st.caption(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    st.caption(
        f"Effect Size ({test_results['effect_size_label']}): "
        f"{test_results['effect_size']:.3f} | P-value: {test_results['p_value']:.4f}"
    )

    if test_results["p_value"] < alpha_threshold:
        winner = groups[1] if mean_b > mean_a else groups[0]
        st.success(f"Winner: {winner} is statistically significant.")
    else:
        st.warning("Result is not statistically significant.")

    st.caption(f"Method: {test_results['test_name']}")
    if alpha_threshold != ALPHA:
        st.caption(f"Decision threshold after correction: alpha={alpha_threshold:.4f}")

    # Only the chi-squared result carries a validity flag; Welch results omit it.
    if dict(test_results).get("chi_square_valid") is False:
        st.warning(
            "At least one expected cell count is below 5. The chi-squared approximation may be unreliable."
        )


def show_bayesian_results(
    bayes_results: BayesianAnalysisResult,
    groups: list[str],
) -> None:
    """Render the core Bayesian metrics for a two-group experiment."""
    col1, col2 = st.columns(2)
    col1.metric("P(Variant Beats Control)", f"{bayes_results['prob_b_wins']:.1%}")
    col2.metric("Expected Loss (if wrong)", f"{bayes_results['expected_loss']:.3%}")

    st.caption(
        f"Posterior: {groups[0]} ~ Beta({bayes_results['alpha_a']:.0f}, {bayes_results['beta_a']:.0f}), "
        f"{groups[1]} ~ Beta({bayes_results['alpha_b']:.0f}, {bayes_results['beta_b']:.0f})"
    )


def show_bayesian_decision(
    recommendation: str,
    confidence: str,
    group_name: str = "Variant B",
    expected_loss: float | None = None,
    loss_tolerance: float = 0.005,
) -> None:
    """Render a loss-aware Bayesian recommendation."""
    message = recommendation.replace("Variant B", group_name)

    if confidence == "high" and "Keep Control" in recommendation:
        st.error(f"High confidence: {message}")
    elif confidence == "high":
        st.success(f"High confidence: {message}")
    elif confidence == "moderate":
        st.info(f"Moderate confidence: {message}")
    else:
        st.warning(f"Uncertain: {message}")

    if expected_loss is not None:
        st.caption(
            f"Expected loss threshold: {loss_tolerance:.2%}. "
            f"Current expected loss: {expected_loss:.3%}."
        )


def show_plan_verification(rows: list[VerificationRow]) -> None:
    """Render the pre-registered plan against what the experiment actually did.

    Ordered worst-first so a broken commitment is read before the effect it
    would otherwise qualify.
    """
    ranked = sorted(rows, key=lambda row: {"fail": 0, "caution": 1, "ok": 2}[row["status"]])
    table = pd.DataFrame(
        [
            {
                "What you promised": row["item"],
                "Planned": row["planned"],
                "What happened": row["actual"],
                "Status": {"ok": "ok", "caution": "check this", "fail": "broken"}[row["status"]],
            }
            for row in ranked
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    for row in ranked:
        if row["status"] == "fail":
            st.error(f"{row['item']}: {row['note']}")
        elif row["status"] == "caution":
            st.warning(f"{row['item']}: {row['note']}")


def show_readout_summary(summary: ReadoutSummary) -> None:
    """Render a result decision-first: effect, then uncertainty, then money.

    Tone follows how strong the claim actually is, not just whether the point
    estimate looks good: ``floor_clears_bar`` is the only case where even the
    pessimistic end of the range clears the bar, so it is the only one that
    earns a green success. A "material" result whose interval floor sits just
    above zero, or a "conclusive" null, is real information but not a
    green-light, so both render as info.
    """
    if summary["floor_clears_bar"]:
        st.success(summary["headline"])
    elif summary["material"] or summary["conclusive"]:
        st.info(summary["headline"])
    else:
        st.warning(summary["headline"])

    left, right = st.columns(2)
    left.metric("Change in the metric", f"{summary['absolute_uplift']:+.4f}")
    right.metric("Change as a percentage", f"{summary['relative_uplift']:+.2%}")
    st.caption(summary["uncertainty_line"])

    if not summary["downside_ruled_out"]:
        st.caption("This test has not ruled out a loss on the downside.")

    if summary["business_impact"] is not None:
        low, high = summary["business_impact"]
        st.caption(
            f"Across everyone a full rollout would reach, that is worth somewhere between "
            f"{low:,.0f} and {high:,.0f}. The range is the honest answer. The single number "
            "people usually quote is just one point inside it."
        )


def show_guardrail_reading(reading: GuardrailReading) -> None:
    """Render what a guardrail actually did, styled like the other status renderers.

    Red for a guardrail that broke, amber for one that cannot yet be told
    apart from noise or was never sized to see harm this small, green for one
    that stayed where it should.
    """
    import math

    tone = {"fail": st.error, "caution": st.warning, "ok": st.success}[reading["status"]]
    tone(reading["note"])
    left, right = st.columns(2)
    left.metric("Control rate", f"{reading['observed_control_rate']:.2%}")
    right.metric("Variant rate", f"{reading['observed_variant_rate']:.2%}")
    ci_lower, ci_upper = reading["ci_relative"]

    relative_change = reading["relative_change"]
    if math.isinf(relative_change):
        st.caption(
            "Relative change: infinite (control arm had zero events, so no baseline for comparison)."
        )
    elif math.isnan(ci_lower) or math.isnan(ci_upper):
        st.caption(
            "Relative change: cannot be computed (baseline was zero in one or both arms)."
        )
    else:
        st.caption(
            f"Relative change: {relative_change:+.1%}, somewhere between "
            f"{ci_lower:+.1%} and {ci_upper:+.1%}."
        )


def show_preregistration(plan: PreRegistration) -> None:
    """Render the locked plan so it stays visible while the result is read."""
    st.caption(f"Plan written down {plan['created_at']} for '{plan['primary_metric']}'.")
    left, middle, right = st.columns(3)
    left.metric("Users planned", f"{plan['n_total']:,}")
    middle.metric("Change to detect", f"{plan['mde_relative']:.1%}")
    right.metric("Days planned", f"{plan['total_days']} days")
    st.caption(
        f"False alarms accepted {plan['alpha']:.3f} · chance of spotting the change "
        f"{plan['power']:.0%} · split {plan['split_ratio']:.0%} · counting "
        f"{'everyone who entered' if plan['estimand'] == 'ITT' else plan['estimand']} · trimming "
        f"{plan['transform']} · {plan['n_primary_metrics']} main metric(s) · "
        f"{plan['planned_looks']} planned check(s)"
    )
    if plan["decision_rule"]:
        st.caption(f"Agreed in advance: {plan['decision_rule']}")
