"""Reusable Streamlit components for showing experiment results and layout chrome."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from config import ALPHA


def inject_app_styles() -> None:
    """Inject the editorial design system used across the app."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #eff2f7;
                --ink: #10131a;
                --muted: #646c79;
                --blue: #4f6dff;
                --mint: #1ecf9b;
                --amber: #d18a1f;
                --red: #dd5b52;
                --sidebar-top: #0d1320;
                --sidebar-bottom: #101925;
                --card-border: rgba(112, 128, 156, 0.16);
                --card-bg: rgba(255, 255, 255, 0.72);
                --card-shadow: 0 24px 56px rgba(24, 31, 48, 0.08);
                --hero-shadow: 0 32px 72px rgba(16, 19, 26, 0.28);
                --radius-xl: 32px;
                --radius-lg: 24px;
                --radius-md: 18px;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 8%, rgba(79, 109, 255, 0.12), transparent 34%),
                    radial-gradient(circle at 88% 10%, rgba(30, 207, 155, 0.10), transparent 28%),
                    radial-gradient(circle at 50% 100%, rgba(79, 109, 255, 0.08), transparent 34%),
                    var(--bg);
                color: var(--ink);
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

            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(79, 109, 255, 0.18);
                background: linear-gradient(180deg, rgba(79, 109, 255, 0.10), rgba(79, 109, 255, 0.06));
                color: var(--ink);
                padding: 0.7rem 1.15rem;
                font-weight: 600;
                box-shadow: 0 10px 24px rgba(79, 109, 255, 0.08);
            }

            .stButton > button:hover {
                border-color: rgba(79, 109, 255, 0.34);
                color: var(--blue);
            }

            [data-baseweb="select"] > div,
            [data-baseweb="input"] > div,
            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                border-radius: 18px;
            }

            .stSelectbox label,
            .stRadio label,
            .stNumberInput label,
            .stTextInput label,
            .stTextArea label,
            .stFileUploader label {
                color: var(--ink);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }

            div[data-testid="stMetric"] {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 1rem 1rem 0.9rem;
                box-shadow: var(--card-shadow);
                backdrop-filter: blur(18px);
            }

            div[data-testid="stMetric"] label {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
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
                border-radius: 22px;
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
                font-family: "Avenir Next", "Helvetica Neue", sans-serif;
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
                border-radius: 36px;
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
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin-top: 1.25rem;
            }

            .summary-card,
            .empty-card,
            .section-note {
                border-radius: 26px;
                border: 1px solid var(--card-border);
                background: var(--card-bg);
                box-shadow: var(--card-shadow);
                backdrop-filter: blur(18px);
            }

            .summary-card {
                padding: 1.15rem 1.15rem 1rem;
            }

            .summary-card.anchor {
                background:
                    linear-gradient(180deg, rgba(16, 19, 26, 0.96), rgba(18, 24, 36, 0.90));
                border-color: rgba(79, 109, 255, 0.24);
                box-shadow: var(--hero-shadow);
            }

            .summary-label,
            .empty-label,
            .signal-label,
            .note-label {
                margin: 0;
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
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
                color: var(--blue);
            }

            .tone-mint {
                color: var(--mint);
            }

            .tone-amber {
                color: var(--amber);
            }

            .tone-red {
                color: var(--red);
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


def render_html_block(markup: str) -> None:
    """Render a raw HTML fragment through Streamlit without markdown indentation issues."""
    st.markdown(markup, unsafe_allow_html=True)


def render_sidebar_intro(
    title: str,
    body: str,
    ai_enabled: bool,
    provider: str,
) -> None:
    """Render the dark sidebar brand block."""
    provider_state = f"AI mapping ready via {provider.upper()}" if ai_enabled else "AI mapping disabled"
    render_html_block(
        f'<div class="editorial-sidebar">'
        f'<p class="editorial-kicker">Experiment review</p>'
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(body)}</p>"
        f'<span class="sidebar-chip">{escape(provider_state)}</span>'
        f"</div>"
    )


def render_hero_card(
    kicker: str,
    title: str,
    body: str,
    pills: list[str],
) -> None:
    """Render the large dark hero card at the top of the page."""
    pill_markup = "".join(f'<span class="pill">{escape(pill)}</span>' for pill in pills)
    render_html_block(
        f'<div class="editorial-hero">'
        f'<p class="editorial-kicker">{escape(kicker)}</p>'
        f'<h1 class="hero-title">{escape(title)}</h1>'
        f'<p class="hero-body">{escape(body)}</p>'
        f'<div class="pill-row">{pill_markup}</div>'
        f"</div>"
    )


def render_summary_cards(cards: list[dict[str, str | bool]]) -> None:
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

    render_html_block(f'<div class="summary-grid">{"".join(card_markup)}</div>')


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

    render_html_block(f'<div class="empty-grid">{"".join(markup)}</div>')


def render_signal_header(signal: str, title: str, body: str) -> None:
    """Render the editorial section header used before each major section."""
    render_html_block(
        f'<div class="signal-header">'
        f'<p class="signal-label">{escape(signal)}</p>'
        f"<h2>{escape(title)}</h2>"
        f'<p class="signal-body">{escape(body)}</p>'
        f"</div>"
    )


def render_section_note(label: str, body: str) -> None:
    """Render a short glass-style note above a widget cluster."""
    render_html_block(
        f'<div class="section-note">'
        f'<p class="note-label">{escape(label)}</p>'
        f'<p class="note-body">{escape(body)}</p>'
        f"</div>"
    )


def render_section_rule() -> None:
    """Render a visible separator between major page sections."""
    render_html_block('<hr class="editorial-rule" />')


def show_data_quality(df: pd.DataFrame) -> None:
    """Render a small data quality summary for a DataFrame."""
    st.markdown("**Data Quality**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Missing Values", int(df.isnull().sum().sum()))
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("View full dataset"):
        st.dataframe(df)


def show_data_preview(df: pd.DataFrame, n_rows: int = 3) -> None:
    """Render the first few rows of a DataFrame."""
    st.write("Preview:", df.head(n_rows))


def show_srm_warning(ratio: float, threshold: float = 0.05) -> None:
    """Show a warning when the experiment split looks suspicious."""
    if abs(ratio - 0.5) > threshold:
        st.warning(
            f"Sample Ratio Mismatch: {ratio:.1%} vs expected 50%. Check your randomization."
        )


def show_frequentist_results(
    test_results: dict[str, Any],
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

    if "chi_square_valid" in test_results and not test_results["chi_square_valid"]:
        st.warning(
            "At least one expected cell count is below 5. The chi-squared approximation may be unreliable."
        )


def show_bayesian_results(
    bayes_results: dict[str, float],
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
