"""Streamlit app for experiment design and analysis."""

from __future__ import annotations

from io import BytesIO
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import ALPHA, PAGE_LAYOUT, PAGE_TITLE
from llm.client import ask_agent as llm_ask_agent
from llm.client import ask_agent_json as llm_ask_agent_json
from llm.client import create_llm_client
from stats.bayesian import beta_binomial_analysis, get_decision_recommendation
from stats.causal import difference_in_differences, regression_discontinuity, select_causal_method
from stats.frequentist import (
    build_frequentist_guardrails,
    calculate_lift,
    calculate_reverse_mde,
    calculate_sample_size,
    check_srm,
    chi_squared_test,
    confidence_interval_binary,
    confidence_interval_continuous,
    welch_t_test,
)
from stats.plots import plot_power_curve
from stats.sanity import run_all_checks
from stats.validation import (
    normalize_metric_type,
    prepare_ab_test_frame,
    prepare_did_frame,
    prepare_rdd_frame,
    validate_mapping_columns,
)
from ui.components import (
    inject_app_styles,
    render_empty_state_cards,
    render_hero_card,
    render_section_note,
    render_section_rule,
    render_sidebar_intro,
    render_signal_header,
    render_summary_cards,
    show_bayesian_decision,
    show_bayesian_results,
    show_data_preview,
    show_data_quality,
    show_frequentist_results,
    show_srm_warning,
)

logger = logging.getLogger(__name__)

REVIEW_FOCI = [
    "Experiment design",
    "Manual result read",
    "Raw CSV audit",
    "Causal fallback",
]


load_dotenv()

st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

inject_app_styles()

client, ai_enabled, llm_provider = create_llm_client()


def ask_agent(system_role: str, user_prompt: str, json_mode: bool = False) -> str | None:
    """Call the configured text model with the session's provider settings."""
    return llm_ask_agent(client, llm_provider, ai_enabled, system_role, user_prompt, json_mode)


def ask_agent_json(
    system_role: str,
    user_prompt: str,
    expected_keys: Sequence[str],
) -> dict[str, Any] | None:
    """Call the configured text model and parse a small JSON payload."""
    return llm_ask_agent_json(
        client=client,
        provider=llm_provider,
        ai_enabled=ai_enabled,
        system_role=system_role,
        user_prompt=user_prompt,
        expected_keys=expected_keys,
    )


def show_dropped_rows_notice(dropped_rows: int, original_rows: int) -> None:
    """Explain when rows were removed because required analysis fields were missing."""
    if dropped_rows <= 0:
        return

    st.warning(
        f"Dropped {dropped_rows:,} of {original_rows:,} rows because one of the required "
        "analysis columns was missing. Review missingness before trusting the estimate."
    )


def render_sensitivity_analysis(
    baseline: float,
    daily_traffic: int,
    weeks: int,
    split_ratio: float,
) -> None:
    """Render a lightweight sensitivity view for traffic and baseline assumptions."""
    with st.expander("Sensitivity analysis"):
        st.caption(
            "Small changes in traffic or baseline can move the MDE more than people expect. "
            "Use this as a quick stress test before you lock the plan."
        )

        st.plotly_chart(
            plot_power_curve(baseline=baseline, daily_traffic=daily_traffic),
            width='stretch',
        )

        traffic_rows: list[dict[str, Any]] = []
        for label, factor in (
            ("50% traffic", 0.5),
            ("Current plan", 1.0),
            ("150% traffic", 1.5),
            ("200% traffic", 2.0),
        ):
            scenario_traffic = max(100, int(round(daily_traffic * factor)))
            result = calculate_reverse_mde(
                baseline=baseline,
                daily_visitors=scenario_traffic,
                weeks=weeks,
                split_ratio=split_ratio,
            )
            traffic_rows.append(
                {
                    "Scenario": label,
                    "Daily traffic": f"{scenario_traffic:,}",
                    "Detectable MDE": (
                        f"{result['mde']:.1%}" if "mde" in result else result["error"]
                    ),
                }
            )

        baseline_rows: list[dict[str, Any]] = []
        baseline_scenarios = sorted(
            {
                round(max(0.01, baseline * 0.5), 4),
                round(baseline, 4),
                round(min(0.5, baseline * 1.5), 4),
            }
        )
        for scenario_baseline in baseline_scenarios:
            result = calculate_reverse_mde(
                baseline=scenario_baseline,
                daily_visitors=daily_traffic,
                weeks=weeks,
                split_ratio=split_ratio,
            )
            baseline_rows.append(
                {
                    "Baseline conversion": f"{scenario_baseline:.1%}",
                    "Detectable MDE": (
                        f"{result['mde']:.1%}" if "mde" in result else result["error"]
                    ),
                }
            )

        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(traffic_rows), hide_index=True, width='stretch')
        right.dataframe(pd.DataFrame(baseline_rows), hide_index=True, width='stretch')


def render_frequentist_guardrail_controls(key_prefix: str) -> tuple[int, bool]:
    """Collect analyst choices that affect p-value interpretation."""
    with st.expander("Frequentist guardrails"):
        n_comparisons = int(
            st.number_input(
                "How many primary metrics are you judging?",
                min_value=1,
                value=1,
                step=1,
                key=f"{key_prefix}_n_comparisons",
            )
        )
        peeked_early = st.checkbox(
            "I looked at results before the planned stop date.",
            key=f"{key_prefix}_peeked_early",
        )
        st.caption(
            "If you test several primary metrics, the adjusted alpha matters. If you peeked "
            "early, the p-value is optimistic unless the experiment used a sequential design."
        )
    return n_comparisons, peeked_early


def show_frequentist_guardrails(guardrails: dict[str, int | float | bool]) -> None:
    """Render adjusted-alpha and peeking warnings for frequentist analyses."""
    if bool(guardrails["alpha_adjusted"]):
        st.info(
            f"Using a Bonferroni-adjusted alpha of {float(guardrails['adjusted_alpha']):.4f} "
            f"across {int(guardrails['n_comparisons'])} primary metrics."
        )
    if bool(guardrails["peeked_early"]):
        st.warning(
            "You marked this analysis as peeked early. Treat the p-value as optimistic "
            "unless the experiment used a sequential-testing plan."
        )


def severity_rank(status: str) -> int:
    """Map status text to an ordered severity rank."""
    return {"ok": 0, "caution": 1, "fail": 2}.get(status, 0)


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
    """Return a color tone for experiment duration."""
    if days <= 28:
        return "mint"
    if days <= 56:
        return "amber"
    return "red"


def read_uploaded_dataframe(widget_key: str) -> pd.DataFrame | None:
    """Read a CSV uploaded via a Streamlit file uploader key."""
    uploaded_file = st.session_state.get(widget_key)
    if uploaded_file is None:
        return None

    try:
        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception:
        return None


def sidebar_tip(review_focus: str) -> str:
    """Return a concise sidebar tip based on the selected review lens."""
    tips = {
        "Experiment design": "Size the claim before you size the excitement.",
        "Manual result read": "A winner with a weak stop rule is not a winner yet.",
        "Raw CSV audit": "Mapped columns are still assumptions until you inspect the frame.",
        "Causal fallback": "When randomization fails, the assumption becomes the product.",
    }
    return tips[review_focus]


def build_card(
    label: str,
    value: str,
    meta: str,
    tone: str,
    anchor: bool = False,
) -> dict[str, str | bool]:
    """Create a summary-card payload for the hero row."""
    return {
        "label": label,
        "value": value,
        "meta": meta,
        "tone": tone,
        "anchor": anchor,
    }


def design_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the design-review lens."""
    baseline = float(st.session_state.get("main_base", 10.0)) / 100
    mde = float(st.session_state.get("main_mde", 10.0)) / 100
    daily_traffic = int(st.session_state.get("main_traffic", 5000))
    split_ratio = float(st.session_state.get("main_split", 50)) / 100

    size = calculate_sample_size(baseline, mde, daily_traffic, split_ratio)
    weeks_required = max(1, int(np.ceil(size["days"] / 7)))
    checks = run_all_checks(baseline, mde, daily_traffic, weeks_required)
    weakest_name, weakest_status, weakest_reason = max(
        checks,
        key=lambda item: severity_rank(item[1]),
    )
    weakest_value = weakest_name if weakest_status != "ok" else "No structural red flag"
    weakest_meta = first_sentence(weakest_reason) or "Traffic, baseline, and lift are aligned."

    return {
        "kicker": "Editorial experiment review",
        "title": "Size the test before the result starts steering the roadmap.",
        "body": (
            "Traffic, lift, split, and stop window belong together. This review checks whether "
            "the claim you want to make is actually supportable before the first readout arrives."
        ),
        "pills": [
            "A/B design",
            f"Baseline {baseline:.1%}",
            f"Target lift {mde:.1%}",
            f"Variant split {int(split_ratio * 100)}%",
        ],
        "cards": [
            build_card("Review lens", "A/B design", "Plan the claim before launch.", "blue", anchor=True),
            build_card("Estimated duration", f"{size['days']} days", "At the current traffic level.", duration_tone(size["days"])),
            build_card(
                "Split penalty",
                f"{size['split_penalty']}%",
                "Time lost versus an even split.",
                "amber" if size["split_penalty"] > 0 else "mint",
            ),
            build_card("Weakest signal", weakest_value, weakest_meta, {"ok": "mint", "caution": "amber", "fail": "red"}[weakest_status]),
        ],
    }


def manual_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the manual-result lens."""
    visitors_a = int(st.session_state.get("manual_visitors_a", 1000))
    conversions_a = int(st.session_state.get("manual_conversions_a", 100))
    visitors_b = int(st.session_state.get("manual_visitors_b", 1000))
    conversions_b = int(st.session_state.get("manual_conversions_b", 115))
    n_comparisons = int(st.session_state.get("manual_n_comparisons", 1))
    peeked_early = bool(st.session_state.get("manual_peeked_early", False))

    if conversions_a > visitors_a or conversions_b > visitors_b:
        return {
            "kicker": "Editorial experiment review",
            "title": "Read the result, but audit the counts first.",
            "body": (
                "This lens compares significance, expected loss, and structural warnings in one place. "
                "Right now the raw counts are inconsistent, so the review stops at the input layer."
            ),
            "pills": ["Manual read", "Counts only", "Input mismatch"],
            "cards": [
                build_card("Review lens", "Manual read", "Use counts when you do not need raw rows.", "blue", anchor=True),
                build_card("Validation", "Count mismatch", "Conversions cannot exceed visitors.", "red"),
                build_card("Bayesian read", "On hold", "The posterior only matters after the counts are valid.", "amber"),
                build_card("Weakest signal", "Input quality", "Fix the counts before reading the result.", "red"),
            ],
        }

    cr_a = conversions_a / visitors_a
    cr_b = conversions_b / visitors_b
    lift = calculate_lift(cr_a, cr_b)
    failures_a = visitors_a - conversions_a
    failures_b = visitors_b - conversions_b
    test_results = chi_squared_test(conversions_a, failures_a, conversions_b, failures_b)
    bayes = beta_binomial_analysis(conversions_a, failures_a, conversions_b, failures_b)
    recommendation, confidence = get_decision_recommendation(
        bayes["prob_b_wins"],
        bayes["expected_loss"],
    )
    guardrails = build_frequentist_guardrails(
        n_comparisons=n_comparisons,
        peeked_early=peeked_early,
    )
    adjusted_alpha = float(guardrails["adjusted_alpha"])
    has_srm, _ = check_srm(visitors_a, visitors_b)

    if test_results["p_value"] < adjusted_alpha and not has_srm and not peeked_early and bool(test_results["chi_square_valid"]):
        frequentist_value = "Decision-ready"
        frequentist_meta = f"p={test_results['p_value']:.4f} and the structure is clean."
        frequentist_tone = "mint"
    elif test_results["p_value"] < adjusted_alpha:
        frequentist_value = "Usable with caution"
        frequentist_meta = f"p={test_results['p_value']:.4f}, but the read has structural caveats."
        frequentist_tone = "amber"
    else:
        frequentist_value = "Need more data"
        frequentist_meta = f"p={test_results['p_value']:.4f} at alpha {adjusted_alpha:.4f}."
        frequentist_tone = "amber"

    if has_srm:
        weakest_value = "Sample ratio mismatch"
        weakest_meta = "The randomization split does not look like 50/50."
        weakest_tone = "red"
    elif not bool(test_results["chi_square_valid"]):
        weakest_value = "Low expected cell counts"
        weakest_meta = "Binary significance is fragile with sparse cells."
        weakest_tone = "amber"
    elif peeked_early:
        weakest_value = "Early peeking"
        weakest_meta = "The nominal p-value is optimistic without a stop rule."
        weakest_tone = "amber"
    elif bool(guardrails["alpha_adjusted"]):
        weakest_value = "Multiple primary metrics"
        weakest_meta = f"The decision threshold tightened to {adjusted_alpha:.4f}."
        weakest_tone = "amber"
    else:
        weakest_value = "No structural red flag"
        weakest_meta = "The read is clean enough to interpret directly."
        weakest_tone = "mint"

    return {
        "kicker": "Editorial experiment review",
        "title": "Read the result before you start telling a winner story.",
        "body": (
            "A useful experiment read combines significance, expected loss, and the weak spots in the setup. "
            "This lens keeps the structural warnings next to the lift so the number does not get to speak alone."
        ),
        "pills": [
            "Manual result read",
            f"Lift {lift:.1%}",
            f"p={test_results['p_value']:.4f}",
            f"P(B>A) {bayes['prob_b_wins']:.1%}",
        ],
        "cards": [
            build_card("Review lens", "Manual read", "Counts first, narrative second.", "blue", anchor=True),
            build_card("Frequentist read", frequentist_value, frequentist_meta, frequentist_tone),
            build_card("Bayesian read", recommendation, f"Expected loss {bayes['expected_loss']:.3%}.", "mint" if confidence == "high" else "amber"),
            build_card("Weakest signal", weakest_value, weakest_meta, weakest_tone),
        ],
    }


def csv_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the raw-CSV lens."""
    df = read_uploaded_dataframe("csv_upload")
    uploaded_file = st.session_state.get("csv_upload")

    if df is None or uploaded_file is None:
        return {
            "kicker": "Editorial experiment review",
            "title": "Audit the raw frame before the mapped column becomes the story.",
            "body": (
                "The model can suggest column roles, but it cannot guarantee that the dataset is decision-safe. "
                "This lens starts with missingness, duplication, and schema shape before it runs the test."
            ),
            "pills": ["Raw CSV audit", "LLM-assisted mapping", "Schema first"],
            "cards": [
                build_card("Review lens", "CSV audit", "Use this when you want the dataframe, not just summary stats.", "blue", anchor=True),
                build_card("Dataset", "Waiting on upload", "Load a CSV to inspect structure and metric type.", "amber"),
                build_card("AI mapping", "Ready" if ai_enabled else "Disabled", "The model only proposes a schema.", "mint" if ai_enabled else "amber"),
                build_card("Weakest signal", "No frame loaded", "The audit starts once the data is visible.", "amber"),
            ],
        }

    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    weakest_value = "No structural red flag"
    weakest_meta = "The frame looks clean at the surface level."
    weakest_tone = "mint"
    if missing > 0:
        weakest_value = "Missing required fields"
        weakest_meta = f"{missing:,} missing values are present before any mapping happens."
        weakest_tone = "amber"
    elif duplicates > 0:
        weakest_value = "Duplicate rows"
        weakest_meta = f"{duplicates:,} duplicated rows could inflate the read."
        weakest_tone = "amber"

    return {
        "kicker": "Editorial experiment review",
        "title": f"Review the frame before {uploaded_file.name} starts steering the answer.",
        "body": (
            "This audit treats column mapping as a schema guess, not a truth source. Review the raw frame, "
            "the missingness, and the duplicated rows before you trust the automated test."
        ),
        "pills": [
            "Raw CSV audit",
            f"{len(df):,} rows",
            f"{len(df.columns)} columns",
            "LLM mapping" if ai_enabled else "Manual interpretation",
        ],
        "cards": [
            build_card("Review lens", "CSV audit", "Schema before significance.", "blue", anchor=True),
            build_card("Dataset", uploaded_file.name, f"{len(df):,} rows and {len(df.columns)} columns.", "blue"),
            build_card("Missing values", f"{missing:,}", "Counted before the statistical read.", "amber" if missing > 0 else "mint"),
            build_card("Weakest signal", weakest_value, weakest_meta, weakest_tone),
        ],
    }


def causal_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the causal-fallback lens."""
    has_cutoff = str(st.session_state.get("causal_has_cutoff", "No")).startswith("Yes")
    has_clean_control = str(st.session_state.get("causal_has_control", "No")).startswith("Yes")
    is_opt_in = str(st.session_state.get("causal_is_opt_in", "No (forced)")).startswith("Yes")
    method = select_causal_method(
        has_cutoff=has_cutoff,
        has_clean_control=has_clean_control,
        is_opt_in=is_opt_in,
    )

    weakest_map = {
        "Difference-in-Differences (DiD)": (
            "Parallel trends",
            "The control group must move like the treated group before the intervention.",
            "amber",
        ),
        "Regression Discontinuity (RDD)": (
            "Cutoff integrity",
            "Units cannot sort around the threshold you plan to use.",
            "amber",
        ),
        "Propensity Score Matching (PSM)": (
            "Selection bias",
            "Observed covariates rarely remove all of the self-selection problem.",
            "red",
        ),
        "CausalImpact": (
            "Counterfactual stability",
            "The time series needs a believable baseline to compare against.",
            "amber",
        ),
    }
    weakest_value, weakest_meta, weakest_tone = weakest_map[method]

    did_df = read_uploaded_dataframe("did_upload")
    rdd_df = read_uploaded_dataframe("rdd_upload")
    uploaded_rows = None
    if method == "Difference-in-Differences (DiD)" and did_df is not None:
        uploaded_rows = f"{len(did_df):,} panel rows loaded"
    elif method == "Regression Discontinuity (RDD)" and rdd_df is not None:
        uploaded_rows = f"{len(rdd_df):,} observations loaded"

    return {
        "kicker": "Editorial experiment review",
        "title": "When randomization fails, the assumption becomes the product.",
        "body": (
            "A causal fallback is only as good as the identifying assumption it borrows. This lens chooses the "
            "least bad method, then keeps its fragile point visible instead of burying it in regression output."
        ),
        "pills": [
            "Causal fallback",
            method,
            uploaded_rows if uploaded_rows else "No dataset loaded",
        ],
        "cards": [
            build_card("Review lens", "Causal fallback", "Use this when randomization is gone or weak.", "blue", anchor=True),
            build_card("Recommended method", method, "Picked from the observed study shape.", "blue"),
            build_card("AI mapping", "Ready" if ai_enabled else "Disabled", "The model only maps columns, not assumptions.", "mint" if ai_enabled else "amber"),
            build_card("Weakest signal", weakest_value, weakest_meta, weakest_tone),
        ],
    }


def build_page_snapshot(review_focus: str) -> dict[str, Any]:
    """Build top-of-page hero and summary content for the selected lens."""
    builders = {
        "Experiment design": design_snapshot,
        "Manual result read": manual_snapshot,
        "Raw CSV audit": csv_snapshot,
        "Causal fallback": causal_snapshot,
    }
    return builders[review_focus]()


def render_sidebar() -> str:
    """Render the branded control rail and return the selected review focus."""
    with st.sidebar:
        render_sidebar_intro(
            title="Experiment Architect",
            body=(
                "Review the design before the result starts steering decisions. Use the rail "
                "to pick the lens, then read the signals in order."
            ),
            ai_enabled=ai_enabled,
            provider=llm_provider,
        )
        review_focus = st.radio(
            "Review lens",
            REVIEW_FOCI,
            index=0,
            key="review_focus",
        )
        st.caption(
            "Single-page review flow. The lens here changes the hero and summary row, "
            "but the whole page stays readable from top to bottom."
        )
        st.markdown("**Tip**")
        st.caption(sidebar_tip(review_focus))
    return review_focus


def render_empty_state() -> None:
    """Render explainer cards when the app has not yet loaded any datasets."""
    render_empty_state_cards(
        [
            {
                "label": "Signal 01",
                "title": "Check the claim before launch.",
                "body": "Size the test, review the split, and look at the stop window before traffic turns into a promise.",
            },
            {
                "label": "Signal 02",
                "title": "Read risk, not just lift.",
                "body": "Use significance and expected loss together so the loudest number does not get the final word.",
            },
            {
                "label": "Signal 03",
                "title": "Audit the frame before the model maps it.",
                "body": "Raw rows still need review. A valid column name is not the same thing as a valid analysis role.",
            },
        ]
    )


review_focus = render_sidebar()
page_snapshot = build_page_snapshot(review_focus)

render_hero_card(
    kicker=page_snapshot["kicker"],
    title=page_snapshot["title"],
    body=page_snapshot["body"],
    pills=[str(pill) for pill in page_snapshot["pills"] if pill],
)
render_summary_cards(page_snapshot["cards"])

if not any(
    read_uploaded_dataframe(key) is not None
    for key in ("csv_upload", "did_upload", "rdd_upload")
):
    render_empty_state()


render_section_rule()
render_signal_header(
    "Signal 01",
    "Define the experiment before the result starts sounding inevitable.",
    "Start with the traffic, the lift you want to detect, and the stop window. This section is here to catch overconfident plans before they become dashboards.",
)
render_section_note(
    "Decision-first design",
    "If the traffic, baseline, and wait time do not line up, the launch date is not the real problem. The design is.",
)

with st.expander("Reverse MDE audit"):
    st.markdown(
        "**MDE** is the smallest change you can reliably detect with the traffic and time you have. "
        "Use this when the real question is what the experiment can see, not what you wish it would see."
    )
    wiz_left, wiz_right = st.columns(2)
    wiz_weeks = wiz_left.slider("Max wait time (weeks)", 1, 12, 4, key="wiz_weeks")
    wiz_traffic = int(wiz_right.number_input("Average daily visitors", 100, 1_000_000, 5000, key="wiz_traffic"))
    wiz_base = (
        wiz_right.number_input(
            "Baseline conversion (%)",
            0.1,
            99.0,
            10.0,
            key="wiz_base",
        )
        / 100
    )

    if st.button("Check the smallest detectable lift", key="reverse_mde_button"):
        reverse_mde = calculate_reverse_mde(
            baseline=wiz_base,
            daily_visitors=wiz_traffic,
            weeks=wiz_weeks,
        )
        if "error" in reverse_mde:
            st.error(reverse_mde["error"])
        else:
            st.info(
                f"In {wiz_weeks} weeks, the smallest lift you can reliably detect is "
                f"**{reverse_mde['mde']:.1%}**."
            )

design_left, design_right = st.columns(2)
with design_left:
    baseline = (
        st.number_input(
            "Baseline conversion (%)",
            0.1,
            99.0,
            10.0,
            step=0.5,
            key="main_base",
        )
        / 100
    )
    mde = st.number_input(
        "Target lift (relative %)",
        1.0,
        500.0,
        10.0,
        step=1.0,
        key="main_mde",
    ) / 100
with design_right:
    daily_traffic = int(
        st.number_input(
            "Daily visitors (total)",
            100,
            1_000_000,
            5000,
            step=100,
            key="main_traffic",
        )
    )
    split_ratio = st.slider(
        "Traffic allocation (variant %)",
        1,
        99,
        50,
        key="main_split",
    ) / 100

size = calculate_sample_size(baseline, mde, daily_traffic, split_ratio)
weeks_required = max(1, int(np.ceil(size["days"] / 7)))

metric_left, metric_center, metric_right = st.columns(3)
metric_left.metric("Estimated duration", f"{size['days']} days")
metric_center.metric("Total sample", f"{size['n_total']:,}")
metric_right.metric("Split penalty", f"{size['split_penalty']}%")

if st.button("Run the design review", key="sanity_button"):
    checks = run_all_checks(baseline, mde, daily_traffic, weeks_required)
    for name, status, reason in checks:
        if not reason:
            continue
        if status == "ok":
            st.success(f"{name}: {reason}")
        elif status == "caution":
            st.warning(f"{name}: {reason}")
        else:
            st.error(f"{name}: {reason}")

if split_ratio != 0.5:
    st.warning(f"This split is {size['split_penalty']}% slower than a 50/50 split.")

render_sensitivity_analysis(baseline, daily_traffic, weeks_required, split_ratio)


render_section_rule()
render_signal_header(
    "Signal 02",
    "Read the result with the assumptions still visible.",
    "This section is for the quick decision pass when all you have are counts. It keeps significance, expected loss, and structural caveats in the same field of view.",
)
render_section_note(
    "Summary-stat read",
    "A result can look clean and still be fragile. Read the winner only after you read the stop rule, the split, and the downside of being wrong.",
)

manual_method = st.radio(
    "Analysis method",
    ["Frequentist (P-values)", "Bayesian (Probability)", "Both"],
    horizontal=True,
    key="manual_method",
)

if manual_method != "Bayesian (Probability)":
    manual_n_comparisons, manual_peeked_early = render_frequentist_guardrail_controls("manual")
else:
    manual_n_comparisons, manual_peeked_early = 1, False

manual_left, manual_right = st.columns(2)
with manual_left:
    st.markdown("### Control")
    visitors_a = int(st.number_input("Visitors A", min_value=1, value=1000, key="manual_visitors_a"))
    conversions_a = int(
        st.number_input("Conversions A", min_value=0, value=100, key="manual_conversions_a")
    )
with manual_right:
    st.markdown("### Variant")
    visitors_b = int(st.number_input("Visitors B", min_value=1, value=1000, key="manual_visitors_b"))
    conversions_b = int(
        st.number_input("Conversions B", min_value=0, value=115, key="manual_conversions_b")
    )

if st.button("Read the result", key="manual_result_button"):
    if conversions_a > visitors_a or conversions_b > visitors_b:
        st.error("Conversions cannot exceed visitors.")
    else:
        cr_a = conversions_a / visitors_a
        cr_b = conversions_b / visitors_b
        lift = calculate_lift(cr_a, cr_b)
        _, srm_ratio = check_srm(visitors_a, visitors_b)
        show_srm_warning(srm_ratio)
        st.metric("Relative lift", f"{lift:.2%}")

        failures_a = visitors_a - conversions_a
        failures_b = visitors_b - conversions_b

        if manual_method in ["Frequentist (P-values)", "Both"]:
            guardrails = build_frequentist_guardrails(
                n_comparisons=manual_n_comparisons,
                peeked_early=manual_peeked_early,
            )
            st.markdown("### Frequentist read")
            show_frequentist_guardrails(guardrails)
            manual_test = chi_squared_test(
                conversions_a,
                failures_a,
                conversions_b,
                failures_b,
            )
            manual_ci_lower, manual_ci_upper = confidence_interval_binary(
                cr_a,
                cr_b,
                visitors_a,
                visitors_b,
            )
            show_frequentist_results(
                manual_test,
                manual_ci_lower,
                manual_ci_upper,
                cr_a,
                cr_b,
                ["Control", "Variant B"],
                alpha_threshold=float(guardrails["adjusted_alpha"]),
            )

        if manual_method in ["Bayesian (Probability)", "Both"]:
            if manual_method == "Both":
                st.divider()
            st.markdown("### Bayesian read")
            manual_bayes = beta_binomial_analysis(
                conversions_a,
                failures_a,
                conversions_b,
                failures_b,
            )
            show_bayesian_results(manual_bayes, ["Control", "Variant B"])
            recommendation, confidence = get_decision_recommendation(
                manual_bayes["prob_b_wins"],
                manual_bayes["expected_loss"],
            )
            show_bayesian_decision(
                recommendation,
                confidence,
                expected_loss=manual_bayes["expected_loss"],
            )


render_section_rule()
render_signal_header(
    "Signal 03",
    "Audit the raw rows before the mapped columns start telling the story.",
    "This section is for the cases where summary counts are not enough. Review the raw dataframe, then let the tool propose a mapping and run the test.",
)
render_section_note(
    "Raw dataframe audit",
    "The model can propose a schema, but it cannot promise semantic correctness. Treat the mapping as a hypothesis until the frame looks right to you.",
)

csv_method = st.radio(
    "Analysis method",
    ["Frequentist (P-values)", "Bayesian (Probability)", "Both"],
    horizontal=True,
    key="csv_method",
)

if csv_method != "Bayesian (Probability)":
    csv_n_comparisons, csv_peeked_early = render_frequentist_guardrail_controls("csv")
else:
    csv_n_comparisons, csv_peeked_early = 1, False

uploaded_file = st.file_uploader("Upload a results CSV", type="csv", key="csv_upload")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    show_data_quality(df)
    show_data_preview(df)

    if st.button("Run the dataframe audit", key="csv_analysis_button"):
        mapping = ask_agent_json(
            system_role="""
            You are a data scientist helper.
            Identify these fields from the dataset preview:
            - variant_col: exact name of the experiment group column
            - metric_col: exact name of the outcome column
            - metric_type: binary or continuous

            Return JSON only.
            """,
            user_prompt=(
                f"Headers: {list(df.columns)}\n"
                f"Preview:\n{df.head(3).to_markdown()}"
            ),
            expected_keys=["variant_col", "metric_col", "metric_type"],
        )

        if mapping:
            try:
                validated = validate_mapping_columns(
                    mapping,
                    df,
                    ["variant_col", "metric_col"],
                )
                metric_type = normalize_metric_type(mapping["metric_type"])
                analysis_df, dropped_rows = prepare_ab_test_frame(
                    df,
                    variant_col=validated["variant_col"],
                    metric_col=validated["metric_col"],
                    metric_type=metric_type,
                )
                logger.info(
                    "Accepted CSV mapping: variant=%s metric=%s type=%s",
                    validated["variant_col"],
                    validated["metric_col"],
                    metric_type,
                )

                show_dropped_rows_notice(dropped_rows, len(df))
                st.success(
                    f"Mapped: Variant=`{validated['variant_col']}`, "
                    f"Metric=`{validated['metric_col']}` ({metric_type})"
                )

                groups = analysis_df[validated["variant_col"]].drop_duplicates().tolist()
                group_a = analysis_df[analysis_df[validated["variant_col"]] == groups[0]][
                    validated["metric_col"]
                ]
                group_b = analysis_df[analysis_df[validated["variant_col"]] == groups[1]][
                    validated["metric_col"]
                ]

                n_a, n_b = len(group_a), len(group_b)
                mean_a, mean_b = group_a.mean(), group_b.mean()
                _, srm_ratio = check_srm(n_a, n_b)
                show_srm_warning(srm_ratio)

                lift = calculate_lift(float(mean_a), float(mean_b))
                st.metric(f"Lift ({groups[1]} vs {groups[0]})", f"{lift:.2%}")

                if metric_type == "binary":
                    successes_a = int(group_a.sum())
                    successes_b = int(group_b.sum())
                    failures_a = n_a - successes_a
                    failures_b = n_b - successes_b
                    test_results = chi_squared_test(
                        successes_a,
                        failures_a,
                        successes_b,
                        failures_b,
                    )
                    ci_lower, ci_upper = confidence_interval_binary(
                        float(mean_a),
                        float(mean_b),
                        n_a,
                        n_b,
                    )
                else:
                    test_results = welch_t_test(group_a, group_b)
                    ci_lower, ci_upper = confidence_interval_continuous(group_a, group_b)

                if csv_method in ["Frequentist (P-values)", "Both"]:
                    guardrails = build_frequentist_guardrails(
                        n_comparisons=csv_n_comparisons,
                        peeked_early=csv_peeked_early,
                    )
                    if csv_method == "Both":
                        st.markdown("### Frequentist read")
                    else:
                        st.markdown("### Frequentist read")
                    show_frequentist_guardrails(guardrails)
                    show_frequentist_results(
                        test_results,
                        ci_lower,
                        ci_upper,
                        float(mean_a),
                        float(mean_b),
                        [str(groups[0]), str(groups[1])],
                        alpha_threshold=float(guardrails["adjusted_alpha"]),
                    )

                if csv_method in ["Bayesian (Probability)", "Both"]:
                    if metric_type == "binary":
                        if csv_method == "Both":
                            st.divider()
                        st.markdown("### Bayesian read")
                        csv_bayes = beta_binomial_analysis(
                            successes_a,
                            failures_a,
                            successes_b,
                            failures_b,
                        )
                        show_bayesian_results(csv_bayes, [str(groups[0]), str(groups[1])])
                        recommendation, confidence = get_decision_recommendation(
                            csv_bayes["prob_b_wins"],
                            csv_bayes["expected_loss"],
                        )
                        show_bayesian_decision(
                            recommendation,
                            confidence,
                            group_name=str(groups[1]),
                            expected_loss=csv_bayes["expected_loss"],
                        )
                    else:
                        st.info("Bayesian analysis is only available for binary metrics.")

            except Exception as exc:
                logger.warning("CSV analysis failed: %s", exc)
                st.error(f"Analysis failed: {exc}")

st.divider()
render_section_note(
    "Warehouse fallback",
    "If the result still lives in SQL, generate a notebook stub here and keep the experiment review in the same pass.",
)
target_dwh = st.selectbox(
    "DB dialect",
    ["BigQuery", "Snowflake", "Redshift"],
    key="sql_dwh",
)
sql_input = st.text_area(
    "Your SQL",
    "SELECT variant_id, user_id, revenue FROM logs",
    key="sql_input",
)

if st.button("Generate analysis notebook", key="sql_generator_button"):
    sql_result = ask_agent(
        system_role=f"""
        You are an analytics engineer. Write a Python notebook snippet.
        1. Connect to {target_dwh} and run the user's SQL.
        2. Detect whether the metric is conversion or revenue.
        3. Run the appropriate statistical test.
        4. End with a plain-English print statement naming the winner and the p-value.
        """,
        user_prompt=f"SQL: {sql_input}",
    )
    if sql_result:
        st.code(sql_result, language="python")


render_section_rule()
render_signal_header(
    "Signal 04",
    "Choose the causal fallback when randomization is weak or gone.",
    "This section is deliberately skeptical. It does not ask which method sounds advanced. It asks which assumption you are actually willing to defend.",
)
render_section_note(
    "Quasi-experimental path",
    "A causal estimate without a believable identifying assumption is just a cleaner-looking guess.",
)

selector_left, selector_center, selector_right = st.columns(3)
has_cutoff_choice = selector_left.selectbox(
    "Strict cutoff?",
    ["No", "Yes (e.g. score > 600)"],
    key="causal_has_cutoff",
)
has_control_choice = selector_center.selectbox(
    "Clean control?",
    ["No", "Yes (unaffected users)"],
    key="causal_has_control",
)
is_opt_in_choice = selector_right.selectbox(
    "User self-selection?",
    ["No (forced)", "Yes (opt-in)"],
    key="causal_is_opt_in",
)

recommended_method = select_causal_method(
    has_cutoff=has_cutoff_choice.startswith("Yes"),
    has_clean_control=has_control_choice.startswith("Yes"),
    is_opt_in=is_opt_in_choice.startswith("Yes"),
)
st.success(f"Recommended method: {recommended_method}")

if recommended_method == "Difference-in-Differences (DiD)":
    st.markdown(
        "Upload panel data with a unit ID, time period, treatment flag, and outcome metric."
    )
    did_file = st.file_uploader("Upload CSV for DiD", type="csv", key="did_upload")

    if did_file is not None:
        df_did = pd.read_csv(did_file)
        show_data_quality(df_did)
        show_data_preview(df_did)

        if st.button("Run DiD analysis", key="did_analyze"):
            mapping = ask_agent_json(
                system_role="""
                You are a causal inference expert. Identify these columns:
                - unit_col: user/entity ID column
                - time_col: date or period column
                - treatment_col: binary treatment indicator (0/1)
                - outcome_col: outcome metric

                Return JSON with these 4 keys. Use exact column names from the dataset.
                """,
                user_prompt=(
                    f"Columns: {list(df_did.columns)}\n\n"
                    f"Preview:\n{df_did.head(3).to_markdown()}"
                ),
                expected_keys=["unit_col", "time_col", "treatment_col", "outcome_col"],
            )

            if mapping:
                try:
                    validated = validate_mapping_columns(
                        mapping,
                        df_did,
                        ["unit_col", "time_col", "treatment_col", "outcome_col"],
                    )
                    prepared_df, dropped_rows = prepare_did_frame(
                        df_did,
                        unit_col=validated["unit_col"],
                        time_col=validated["time_col"],
                        treatment_col=validated["treatment_col"],
                        outcome_col=validated["outcome_col"],
                    )
                    logger.info("Accepted DiD mapping: %s", validated)

                    show_dropped_rows_notice(dropped_rows, len(df_did))
                    st.success(
                        "Mapped: "
                        f"Unit={validated['unit_col']}, "
                        f"Time={validated['time_col']}, "
                        f"Treatment={validated['treatment_col']}, "
                        f"Outcome={validated['outcome_col']}"
                    )

                    unique_times = prepared_df[validated["time_col"]].drop_duplicates().tolist()
                    default_index = 1 if len(unique_times) > 1 else 0
                    intervention_point = st.selectbox(
                        "Intervention date/period",
                        options=unique_times,
                        index=default_index,
                        key="did_intervention",
                    )

                    if st.button("Calculate DiD effect", key="did_calc"):
                        did_result = difference_in_differences(
                            prepared_df,
                            unit_col=validated["unit_col"],
                            time_col=validated["time_col"],
                            treatment_col=validated["treatment_col"],
                            outcome_col=validated["outcome_col"],
                            intervention_point=intervention_point,
                        )
                        logger.info(
                            "Ran DiD on %s rows with %s units.",
                            len(prepared_df),
                            prepared_df[validated["unit_col"]].nunique(),
                        )
                        st.metric("Average treatment effect", f"{did_result['coefficient']:.4f}")
                        st.caption(
                            f"95% CI: [{did_result['ci_lower']:.4f}, {did_result['ci_upper']:.4f}]"
                        )

                        if did_result["p_value"] < ALPHA:
                            st.success(f"Significant effect (p={did_result['p_value']:.4f})")
                        else:
                            st.warning(f"Not significant (p={did_result['p_value']:.4f})")

                        diagnostics = did_result["diagnostics"]
                        if not diagnostics["parallel_trends_test_ran"]:
                            st.info(
                                "Parallel-trends pre-test did not run because there were not enough "
                                "pre-period observations."
                            )
                        elif not diagnostics["parallel_trends_ok"]:
                            st.warning(
                                "Parallel trends may be violated "
                                f"(pre-period interaction p={diagnostics['parallel_trends_pvalue']:.3f}). "
                                "Interpret the effect with caution."
                            )

                        with st.expander("Full regression output"):
                            st.text(did_result["model"].summary())

                except Exception as exc:
                    logger.warning("DiD analysis failed: %s", exc)
                    st.error(f"Analysis failed: {exc}")

elif recommended_method == "Regression Discontinuity (RDD)":
    st.markdown(
        "Upload data with a running variable, treatment flag, and outcome metric."
    )
    rdd_file = st.file_uploader("Upload CSV for RDD", type="csv", key="rdd_upload")

    if rdd_file is not None:
        df_rdd = pd.read_csv(rdd_file)
        show_data_quality(df_rdd)
        show_data_preview(df_rdd)

        if st.button("Run RDD analysis", key="rdd_analyze"):
            mapping = ask_agent_json(
                system_role="""
                You are a causal inference expert. Identify these columns:
                - running_var: the running variable (e.g. credit score, age, test score)
                - treatment_col: binary treatment indicator (0/1)
                - outcome_col: outcome metric

                Return JSON with these 3 keys. Use exact column names from the dataset.
                """,
                user_prompt=(
                    f"Columns: {list(df_rdd.columns)}\n\n"
                    f"Preview:\n{df_rdd.head(3).to_markdown()}"
                ),
                expected_keys=["running_var", "treatment_col", "outcome_col"],
            )

            if mapping:
                try:
                    validated = validate_mapping_columns(
                        mapping,
                        df_rdd,
                        ["running_var", "treatment_col", "outcome_col"],
                    )
                    prepared_df, dropped_rows = prepare_rdd_frame(
                        df_rdd,
                        running_var=validated["running_var"],
                        treatment_col=validated["treatment_col"],
                        outcome_col=validated["outcome_col"],
                    )
                    logger.info("Accepted RDD mapping: %s", validated)

                    show_dropped_rows_notice(dropped_rows, len(df_rdd))
                    st.success(
                        "Mapped: "
                        f"Running variable={validated['running_var']}, "
                        f"Treatment={validated['treatment_col']}, "
                        f"Outcome={validated['outcome_col']}"
                    )

                    cutoff = st.number_input(
                        "Treatment cutoff value",
                        value=float(prepared_df[validated["running_var"]].median()),
                        key="rdd_cutoff",
                    )

                    if st.button("Calculate RDD effect", key="rdd_calc"):
                        rdd_result = regression_discontinuity(
                            prepared_df,
                            validated["running_var"],
                            validated["treatment_col"],
                            validated["outcome_col"],
                            cutoff,
                        )
                        logger.info("Ran RDD on %s rows.", len(prepared_df))
                        st.metric("Effect at cutoff", f"{rdd_result['coefficient']:.4f}")
                        st.caption(
                            f"95% CI: [{rdd_result['ci_lower']:.4f}, {rdd_result['ci_upper']:.4f}]"
                        )

                        if rdd_result["p_value"] < ALPHA:
                            st.success(
                                f"Significant discontinuity (p={rdd_result['p_value']:.4f})"
                            )
                        else:
                            st.warning(
                                f"No significant discontinuity (p={rdd_result['p_value']:.4f})"
                            )

                        diagnostics = rdd_result["diagnostics"]
                        st.caption(
                            f"Bandwidth used: {diagnostics['bandwidth_used']:.2f} "
                            f"({str(diagnostics['bandwidth_method']).replace('_', ' ')})."
                        )
                        if not diagnostics["density_ok"]:
                            st.warning(
                                "Density looks unbalanced around the cutoff "
                                f"(ratio={diagnostics['density_ratio_at_cutoff']:.2f}). "
                                "Units may be sorting around the threshold."
                            )
                        if not diagnostics["coefficient_stable_under_bandwidth"]:
                            st.warning(
                                "The estimate shifts materially under a narrower bandwidth. "
                                "Check robustness before drawing conclusions."
                            )

                        with st.expander("RDD bandwidth diagnostics"):
                            st.dataframe(
                                pd.DataFrame(diagnostics["bandwidth_sweep"]),
                                hide_index=True,
                                width='stretch',
                            )

                        with st.expander("Full regression output"):
                            st.text(rdd_result["model"].summary())

                except Exception as exc:
                    logger.warning("RDD analysis failed: %s", exc)
                    st.error(f"Analysis failed: {exc}")

else:
    st.info(
        "This path is still a code-generation fallback. The tool will suggest a script, but it will not pretend the estimator is fully productized in-app."
    )
    target_db = st.selectbox(
        "Target database",
        ["BigQuery", "Snowflake", "Redshift", "Local CSV"],
        key="causal_target_db",
    )
    user_context = st.text_area(
        "Paste column names or SQL schema",
        height=100,
        placeholder="e.g. user_id, transaction_date, treatment_flag, total_spend",
        key="causal_context",
    )

    if st.button("Generate Python script", key="causal_codegen_button"):
        codegen_result = ask_agent(
            system_role=f"""
            You are a senior data scientist. Write a Python script for {recommended_method}.
            Target: {target_db}.

            Requirements:
            1. Use standard libraries where possible.
            2. Include connector code only if needed.
            3. Explain the statistical assumptions in comments.
            4. Keep it practical and executable.
            """,
            user_prompt=f"User context: {user_context}",
        )
        if codegen_result:
            st.code(codegen_result, language="python")
