"""Streamlit app for experiment design and analysis."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from config import ALPHA, PAGE_LAYOUT, PAGE_TITLE, SMALL_SAMPLE_THRESHOLD
from llm.client import ask_agent as llm_ask_agent
from llm.client import ask_agent_json as llm_ask_agent_json
from llm.client import create_llm_client
from stats.bayesian import beta_binomial_analysis, get_decision_recommendation
from stats.causal import difference_in_differences, regression_discontinuity, select_causal_method
from stats.frequentist import (
    EffectSizeMethod,
    FrequentistGuardrails,
    FrequentistTestResult,
    bootstrap_ci_relative_lift_continuous,
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
from stats.power import (
    METRIC_LAYERS,
    DurationPlan,
    MetricLayer,
    compliance_effects,
    estimate_cuped_rho,
    guardrail_detectable_harm,
    intensity_options,
    plan_duration,
    post_treatment_risk,
    sample_size_continuous,
    simulate_power,
    skew_diagnostics,
)
from stats.prereg import (
    PreRegistration,
    build_preregistration,
    summarise_readout,
    verify_against_plan,
)
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
    show_data_quality,
    show_frequentist_results,
    show_plan_verification,
    show_preregistration,
    show_readout_summary,
    show_srm_warning,
)
from ui.formatting import sidebar_tip
from ui.snapshots import REVIEW_FOCI, build_page_snapshot
from ui.state import (
    CAUSAL_HAS_CONTROL,
    CAUSAL_HAS_CUTOFF,
    CAUSAL_IS_OPT_IN,
    CSV_UPLOAD,
    DID_UPLOAD,
    MAIN_BASELINE,
    MAIN_MDE,
    MAIN_SPLIT,
    MAIN_TRAFFIC,
    MANUAL_CONVERSIONS_A,
    MANUAL_CONVERSIONS_B,
    MANUAL_VISITORS_A,
    MANUAL_VISITORS_B,
    POWER_ALPHA,
    POWER_BASELINE_MEAN,
    POWER_DAILY_NEW,
    POWER_GUARDRAIL_BASELINE,
    POWER_MATURATION,
    POWER_MDE_ABS,
    POWER_METRIC_LAYER,
    POWER_POWER,
    POWER_RAMP,
    POWER_RHO,
    POWER_SD,
    POWER_UPLOAD,
    PREREG_PLAN,
    RDD_UPLOAD,
    UPLOAD_KEYS,
    read_uploaded_dataframe,
)

logger = logging.getLogger(__name__)


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
            "early, the p-value is optimistic unless the experiment used a sequential design "
            "(mSPRT, group-sequential, or always-valid confidence intervals)."
        )
    return n_comparisons, peeked_early


def show_frequentist_guardrails(guardrails: FrequentistGuardrails) -> None:
    """Render adjusted-alpha and peeking warnings for frequentist analyses."""
    if guardrails["alpha_adjusted"]:
        st.info(
            f"Using a Bonferroni-adjusted alpha of {guardrails['adjusted_alpha']:.4f} "
            f"across {guardrails['n_comparisons']} primary metrics."
        )
    if guardrails["peeked_early"]:
        st.warning(
            "**Peeking invalidates this p-value.** Standard p-values assume you read "
            "the result exactly once at the planned stop date. Peeking inflates the false "
            "positive rate — with weekly peeks over an 8-week test, the effective FPR can "
            "exceed 20% even with α=0.05. To fix this prospectively, use a sequential "
            "design (mSPRT, group-sequential, or always-valid confidence intervals). "
            "If the test is already done, treat this p-value as a lower bound on uncertainty."
        )


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
                "title": "Noise costs more than traffic.",
                "body": "A jumpy metric needs far more users than a steady one. Use what people did before the test to quiet it down first.",
            },
            {
                "label": "Signal 03",
                "title": "Read risk, not just lift.",
                "body": "Use significance and expected loss together so the loudest number does not get the final word.",
            },
            {
                "label": "Signal 04",
                "title": "Audit the frame before the model maps it.",
                "body": "Raw rows still need review. A valid column name is not the same thing as a valid analysis role.",
            },
        ]
    )


# ── Signal sections ───────────────────────────────────────────────────────────


def render_design_section() -> None:
    """Render Signal 01: experiment design, sample size, and sensitivity."""
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
                key=MAIN_BASELINE,
            )
            / 100
        )
        mde = st.number_input(
            "Target lift (relative %)",
            1.0,
            500.0,
            10.0,
            step=1.0,
            key=MAIN_MDE,
        ) / 100
    with design_right:
        daily_traffic = int(
            st.number_input(
                "Daily visitors (total)",
                100,
                1_000_000,
                5000,
                step=100,
                key=MAIN_TRAFFIC,
            )
        )
        split_ratio = st.slider(
            "Traffic allocation (variant %)",
            1,
            99,
            50,
            key=MAIN_SPLIT,
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


def render_metric_layer_control() -> MetricLayer:
    """Pick the metric layer and flag outcome definitions that break randomization."""
    layer_left, layer_right = st.columns([2, 1])
    layer: MetricLayer = layer_left.selectbox(
        "What are you actually measuring?",
        options=list(METRIC_LAYERS.keys()),
        format_func=lambda key: {
            "conversion": "Signed up or converted",
            "activation": "Got far enough to use it",
            "value_per_active": "Value per active user",
            "value_per_randomised": "Value per user, counting the ones who did nothing",
        }[key],
        key=POWER_METRIC_LAYER,
    )
    filters_post_state = layer_right.checkbox(
        "I only count users who did something first",
        value=False,
        key="power_post_filter",
        help=(
            "Tick this if the number only covers people who opened an account, funded, opted in, "
            "or stayed active after the test started."
        ),
    )

    status, explanation = post_treatment_risk(layer, filters_post_state)
    if status == "fail":
        st.error(explanation)
    elif status == "caution":
        st.warning(explanation)
    else:
        st.success(explanation)
    return layer


def render_continuous_sizing() -> tuple[float, float, float, float, float, int]:
    """Size a continuous-outcome test and return the inputs the rest of the section needs."""
    st.markdown(
        "**How many users for a money metric.** Conversion rates are easy to size, because the "
        "current rate tells you almost everything. Spend and revenue are harder: what matters is "
        "how much people differ from each other. The trade is unforgiving in both directions. "
        "Twice the spread costs four times the users. Twice the change you are chasing saves "
        "three quarters of them."
    )

    size_left, size_right = st.columns(2)
    sd = float(
        size_left.number_input(
            "How much users differ (standard deviation)",
            min_value=0.01,
            value=120.0,
            step=1.0,
            key=POWER_SD,
            help=(
                "Roughly how far a typical user sits from the average. Measure it over the same "
                "time window and the same group of people you plan to test on."
            ),
        )
    )
    baseline_mean = float(
        size_left.number_input(
            "Current average (optional)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key=POWER_BASELINE_MEAN,
            help="Only used to show the change you are chasing as a percentage.",
        )
    )
    mde_absolute = float(
        size_right.number_input(
            "Smallest change worth acting on",
            min_value=0.01,
            value=4.0,
            step=0.5,
            key=POWER_MDE_ABS,
            help=(
                "The smallest change you would still do something about. Decide it from the "
                "business case first. Working backwards from the traffic you happen to have is "
                "how tests end up proving nothing."
            ),
        )
    )
    rho = float(
        size_right.slider(
            "How well past behaviour predicts the outcome (CUPED)",
            min_value=0.0,
            max_value=0.95,
            value=0.0,
            step=0.05,
            key=POWER_RHO,
            help=(
                "If the people who spent a lot last month also spend a lot this month, you can "
                "subtract that predictable part and shrink the noise. 0 means last month tells "
                "you nothing. 0.8 means it tells you a great deal. Measure it below rather than "
                "guessing at it."
            ),
        )
    )

    stance_left, stance_right = st.columns(2)
    alpha = float(
        stance_left.select_slider(
            "How often you accept a false alarm (alpha)",
            options=[0.01, 0.05, 0.10],
            value=ALPHA,
            key=POWER_ALPHA,
        )
    )
    power = float(
        stance_right.select_slider(
            "Chance of spotting a real change (power)",
            options=[0.70, 0.80, 0.90, 0.95],
            value=0.80,
            key=POWER_POWER,
        )
    )

    split_ratio = float(st.session_state.get(MAIN_SPLIT, 50)) / 100
    try:
        size = sample_size_continuous(
            sd=sd,
            mde_absolute=mde_absolute,
            baseline_mean=baseline_mean or None,
            alpha=alpha,
            power=power,
            split_ratio=split_ratio,
            rho=rho,
        )
    except ValueError as error:
        st.error(str(error))
        return sd, mde_absolute, rho, alpha, power, 0

    unadjusted = sample_size_continuous(
        sd=sd,
        mde_absolute=mde_absolute,
        alpha=alpha,
        power=power,
        split_ratio=split_ratio,
    )["n_total"]

    left, middle, right = st.columns(3)
    left.metric("Users needed", f"{size['n_total']:,}")
    middle.metric("Cost of an uneven split", f"{size['allocation_cost']:.2f}x")
    right.metric(
        "Users saved by using past behaviour",
        f"{1 - size['variance_retained']:.0%}",
        delta=f"-{unadjusted - size['n_total']:,} users" if rho > 0 else None,
    )

    if size["mde_relative"] is not None:
        st.caption(
            f"You are looking for a {size['mde_relative']:.1%} change against an average of "
            f"{baseline_mean:,.2f}."
        )
    if split_ratio != 0.5:
        st.caption(
            f"The {split_ratio:.0%} split you set in Signal 01 needs "
            f"{size['allocation_cost']:.2f}x as many users as an even one. The smaller group is "
            "always the bottleneck, so the whole test is only as sharp as the thinner side. "
            "Limited capacity or risk is a fair reason to do it. Habit is not."
        )
    return sd, mde_absolute, rho, alpha, power, size["n_total"]


def render_duration_planner(n_total: int) -> DurationPlan:
    """Translate a sample requirement into a calendar plan and return its parts."""
    st.markdown(
        "**From users needed to a date.** Dividing by daily traffic gets this wrong three ways. "
        "Someone who already joined the test does not count twice when they come back. The last "
        "person to join still needs the full measurement window before you can read their number. "
        "And a cautious slow start is not part of the real test."
    )
    duration_left, duration_middle, duration_right = st.columns(3)
    daily_new = float(
        duration_left.number_input(
            "New users joining the test per day",
            min_value=1.0,
            value=900.0,
            step=50.0,
            key=POWER_DAILY_NEW,
            help=(
                "People who qualify for the first time. Daily active users is the wrong number "
                "here: most of them are already in the test."
            ),
        )
    )
    maturation = int(
        duration_middle.number_input(
            "Days you have to wait per user (measurement window)",
            min_value=0,
            value=30,
            step=1,
            key=POWER_MATURATION,
            help=(
                "If you are measuring 30-day spend, the last person to join still needs 30 days "
                "before their number means anything."
            ),
        )
    )
    ramp = int(
        duration_right.number_input(
            "Days spent ramping up slowly first",
            min_value=0,
            value=0,
            step=1,
            key=POWER_RAMP,
            help=(
                "Showing the change to 10% first is a safety check for engineering. It is not "
                "the test, and its data usually should not be pooled with it."
            ),
        )
    )

    if n_total <= 0:
        return {
            "ramp_days": ramp,
            "enrolment_days": 0,
            "maturation_days": maturation,
            "total_days": ramp + maturation,
            "enrolment_weeks": 0.0,
            "binding_constraint": "Sizing inputs do not resolve yet.",
        }

    plan = plan_duration(
        n_total=n_total,
        daily_new_eligible=daily_new,
        maturation_days=maturation,
        ramp_days=ramp,
    )
    calendar_left, calendar_middle, calendar_right = st.columns(3)
    calendar_left.metric("Signing users up", f"{plan['enrolment_days']} days")
    calendar_middle.metric("Then waiting", f"{plan['maturation_days']} days")
    calendar_right.metric("Total time", f"{plan['total_days']} days")
    st.caption(f"What is actually holding you up: {plan['binding_constraint']}")
    if ramp > 0:
        st.caption(
            "Keep people in whichever group they landed in, including through the ramp, and only "
            "start counting once you are at the final split. If the odds of getting the new "
            "version change from week to week, the two groups end up covering different weeks, "
            "and you cannot tell the change apart from the calendar."
        )
    return plan


def render_variance_and_simulation(sd: float, mde_absolute: float, alpha: float) -> None:
    """Measure CUPED rho and empirical power from an uploaded historical column."""
    st.markdown(
        "**Measure the two numbers people usually guess.** Upload a sample of what real users "
        "did before any of this: one column for the metric, and if you have it, a second column "
        "for the same people in an earlier period."
    )
    st.file_uploader("Historical outcomes (CSV)", type="csv", key=POWER_UPLOAD)
    frame = read_uploaded_dataframe(POWER_UPLOAD)
    if frame is None:
        st.info(
            "Without a file, the numbers above rest on a textbook assumption about the shape of "
            "your metric. On spend and revenue that assumption is usually the first thing to "
            "break, because a handful of users carry most of the total."
        )
        return

    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric_columns:
        st.error("No numeric columns found in that file.")
        return

    column_left, column_right = st.columns(2)
    outcome_column = column_left.selectbox("Metric column", numeric_columns, key="power_outcome_col")
    pre_column = column_right.selectbox(
        "Same users, earlier period (optional)",
        ["(none)", *numeric_columns],
        key="power_pre_col",
    )

    outcome = frame[outcome_column]
    if pre_column != "(none)":
        try:
            measured_rho = estimate_cuped_rho(frame[pre_column], outcome)
        except ValueError as error:
            st.warning(f"Could not estimate rho: {error}")
        else:
            st.success(
                f"Past behaviour predicts this metric at {measured_rho:.2f}. Subtracting the "
                f"predictable part cuts the users you need by {1 - (1 - measured_rho**2):.0%}. "
                "Set the CUPED slider above to this number to count it in."
            )

    try:
        diagnostics = skew_diagnostics(outcome)
    except ValueError as error:
        st.error(str(error))
        return

    skew_left, skew_middle, skew_right = st.columns(3)
    skew_left.metric("Lopsidedness", f"{diagnostics['skewness']:.2f}")
    skew_middle.metric("Held by the top 1%", f"{diagnostics['share_in_top_1_pct']:.0%}")
    skew_right.metric("Users at zero", f"{diagnostics['zero_share']:.0%}")
    st.caption(
        f"The more lopsided the metric, the more users the standard maths needs before it "
        f"behaves. For this one that is roughly {diagnostics['n_for_clt']:,} per group (the "
        "25 x skewness squared rule from Boos and Hughes-Oliver). Below that, run the "
        "simulation instead of trusting the formula."
    )

    simulate_left, simulate_right = st.columns(2)
    sim_n = int(
        simulate_left.number_input(
            "Users per group to try",
            min_value=100,
            value=5000,
            step=500,
            key="power_sim_n",
        )
    )
    sim_lift = (
        simulate_right.number_input(
            "Change you want to catch (%)",
            min_value=0.1,
            value=max(0.1, round(100 * mde_absolute / max(float(outcome.mean()), 1e-9), 1)),
            step=0.5,
            key="power_sim_lift",
        )
        / 100
    )
    winsorise = st.checkbox(
        "Cap the top 1% of users",
        value=False,
        key="power_sim_winsorise",
        help=(
            "Trimming the biggest spenders makes results look tidier. It is only honest if you "
            "decide it now and write it down. Deciding it after you see the result is how people "
            "talk themselves into a win."
        ),
    )

    if st.button("Try this experiment 300 times", key="power_simulate_button"):
        with st.spinner("Running it on real users, over and over..."):
            result = simulate_power(
                values=outcome,
                relative_lift=sim_lift,
                n_per_arm=sim_n,
                alpha=alpha,
                iterations=300,
                winsorise_quantile=0.99 if winsorise else None,
            )
        power_left, power_right = st.columns(2)
        power_left.metric("Times it found the change", f"{result['power']:.0%}")
        power_right.metric("False alarms when nothing changed", f"{result['false_positive_rate']:.1%}")
        if not result["calibrated"]:
            st.error(
                f"Run on data where nothing changed at all, this test still declared a winner "
                f"{result['false_positive_rate']:.1%} of the time, against the {alpha:.0%} you "
                "asked for. On a metric this shape the maths is not behaving, so the user counts "
                "above are not trustworthy either."
            )
        elif result["power"] < 0.80:
            st.warning(
                f"With {sim_n:,} users per group, this test only spots the change "
                f"{result['power']:.0%} of the time. The textbook estimate above assumes a "
                "tidier metric than yours. Add users, quiet the noise, or look for a bigger "
                "change."
            )
        else:
            st.success(
                f"With {sim_n:,} users per group, this test spots the change "
                f"{result['power']:.0%} of the time, and only cries wolf "
                f"{result['false_positive_rate']:.1%} of the time when nothing is there."
            )
        st.caption(
            f"You typed {sd:,.2f} for how much users differ. This run used the real spread in "
            "your file instead, so when the two disagree, believe this one."
        )


def render_intensity_and_compliance(sd: float, alpha: float, power: float) -> None:
    """Price treatment doses, then separate ITT from the complier effect."""
    st.markdown(
        "**How strong to make the change.** A bolder version usually moves the metric more, and a "
        "bigger effect needs far fewer users, so going bolder buys you time cheaply. Only up to a "
        "point. Doubling an incentive rarely doubles the response, while it does double the bill. "
        "And testing a version you would never actually ship answers a question nobody asked."
    )
    baseline_mean = float(st.session_state.get(POWER_BASELINE_MEAN, 100.0))
    daily_new = float(st.session_state.get(POWER_DAILY_NEW, 900.0))
    maturation = int(st.session_state.get(POWER_MATURATION, 30))
    rho = float(st.session_state.get(POWER_RHO, 0.0))
    split_ratio = float(st.session_state.get(MAIN_SPLIT, 50)) / 100

    dose_left, dose_middle, dose_right = st.columns(3)
    weak = dose_left.number_input("Mild version moves it (%)", 0.1, 100.0, 2.0, 0.1, key="power_dose_weak") / 100
    mid = dose_middle.number_input("Medium version moves it (%)", 0.1, 100.0, 4.0, 0.1, key="power_dose_mid") / 100
    strong = dose_right.number_input("Bold version moves it (%)", 0.1, 100.0, 5.0, 0.1, key="power_dose_strong") / 100

    try:
        options = intensity_options(
            doses=[("Weak", weak), ("Medium", mid), ("Strong", strong)],
            baseline=baseline_mean,
            sd=sd,
            daily_new_eligible=daily_new,
            maturation_days=maturation,
            alpha=alpha,
            power=power,
            split_ratio=split_ratio,
            rho=rho,
        )
    except ValueError as error:
        st.error(str(error))
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Version": option["label"],
                        "Expected change": f"{option['expected_effect']:.1%}",
                        "Users needed": f"{option['n_total']:,}",
                        "Days": option["total_days"],
                        "Days saved vs the mild one": option["days_saved_vs_weakest"],
                        "Days saved per extra point of effect": (
                            f"{option['marginal_days_per_effect_point']:.1f}"
                            if option["marginal_days_per_effect_point"] is not None
                            else "-"
                        ),
                    }
                    for option in options
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Watch the last column drop. Once each extra point of effect stops buying you many "
            "days, a bolder version is mostly buying you cost."
        )

    st.divider()
    st.markdown(
        "**When people have to opt in.** Users are split into groups the moment they qualify. "
        "Whether they then take the offer is their choice, and the offer itself pushes that "
        "choice. So the people who opt in are not a random group any more, and comparing them "
        "with the people who did not tells you who was keen, not what your change did. "
        "The number that answers the shipping decision counts everyone who was offered it, "
        "whether they took it or not."
    )
    itt_left, itt_middle, itt_right = st.columns(3)
    itt = itt_left.number_input(
        "Effect across everyone offered it",
        value=0.012,
        step=0.001,
        format="%.4f",
        key="power_itt",
        help="Known as the intention-to-treat effect, or ITT.",
    )
    itt_se = itt_middle.number_input(
        "How uncertain that number is (standard error)",
        min_value=0.0,
        value=0.004,
        step=0.001,
        format="%.4f",
        key="power_itt_se",
    )
    take_up = itt_right.slider("Share who actually took it", 0.0, 1.0, 0.35, 0.05, key="power_take_up")

    compliance = compliance_effects(
        itt=float(itt),
        itt_standard_error=float(itt_se),
        take_up_treatment=float(take_up),
        alpha=alpha,
    )
    itt_low, itt_high = compliance["itt_ci"]
    st.info(
        f"Across everyone offered it: {compliance['itt']:+.4f}, somewhere between "
        f"{itt_low:+.4f} and {itt_high:+.4f}. This is the number that answers what happens if "
        "you ship it, because after launch you also get the people who ignore it."
    )
    if compliance["cace"] is not None and compliance["cace_ci"] is not None:
        cace_low, cace_high = compliance["cace_ci"]
        st.caption(
            f"Among the people who actually took it, the effect works out at "
            f"{compliance['cace']:+.4f}, somewhere between {cace_low:+.4f} and {cace_high:+.4f}. "
            f"{compliance['estimand_note']}"
        )
    else:
        st.caption(compliance["estimand_note"])


def render_plan_lock(alpha: float, power: float, rho: float, duration: DurationPlan) -> None:
    """Freeze the design decisions so the readout can verify them later."""
    baseline = float(st.session_state.get(MAIN_BASELINE, 10.0)) / 100
    mde_relative = float(st.session_state.get(MAIN_MDE, 10.0)) / 100
    split_ratio = float(st.session_state.get(MAIN_SPLIT, 50)) / 100
    layer: MetricLayer = st.session_state.get(POWER_METRIC_LAYER, "conversion")

    st.markdown(
        "**Write the plan down.** A plan is only worth writing if something later checks that you "
        "stuck to it. Once you lock this, the two readout sections below compare what you promised "
        "against what you actually got. The two lines people quietly change once the results are "
        "in are how the metric was trimmed and how many metrics counted as the main one, so both "
        "are recorded here."
    )
    lock_left, lock_right = st.columns(2)
    metric_name = lock_left.text_input("The metric this test is about", value="Checkout conversion", key="prereg_metric")
    transform = lock_right.selectbox(
        "How the metric is trimmed",
        ["none", "winsorise p99", "winsorise p95", "log"],
        key="prereg_transform",
    )
    metrics_left, looks_right = st.columns(2)
    n_primary = int(metrics_left.number_input("How many metrics count as the main one", 1, 10, 1, key="prereg_n_primary"))
    planned_looks = int(looks_right.number_input("How many times you plan to check early", 1, 20, 1, key="prereg_looks"))
    guardrail_baseline = (
        st.number_input(
            "Rate of the thing you must not break (%), such as failed payments",
            min_value=0.0,
            max_value=99.0,
            value=0.2,
            step=0.1,
            key=POWER_GUARDRAIL_BASELINE,
        )
        / 100
    )
    decision_rule = st.text_input(
        "What you agree in advance to do with the result",
        value=(
            "Ship if even the pessimistic end of the range beats the bar we set, and nothing we "
            "said we must not break has moved."
        ),
        key="prereg_rule",
    )

    rate_sd = float(np.sqrt(baseline * (1 - baseline)))
    plan_n = sample_size_continuous(
        sd=rate_sd,
        mde_absolute=baseline * mde_relative,
        alpha=alpha,
        power=power,
        split_ratio=split_ratio,
        rho=rho,
    )["n_total"]
    plan_duration_parts = plan_duration(
        n_total=plan_n,
        daily_new_eligible=float(st.session_state.get(POWER_DAILY_NEW, 900.0)),
        maturation_days=duration["maturation_days"],
        ramp_days=duration["ramp_days"],
    )

    if guardrail_baseline > 0:
        harm = guardrail_detectable_harm(
            baseline_rate=guardrail_baseline,
            n_total=plan_n,
            split_ratio=split_ratio,
            alpha=alpha,
            power=power,
        )
        tone = st.warning if harm > 0.10 else st.info
        tone(
            f"At {plan_n:,} users, something that happens {guardrail_baseline:.2%} of the time "
            f"would have to get {harm:.0%} worse before this test noticed. Anything smaller than "
            "that will look untouched whether it was or not, so read a clean guardrail as "
            "'we could not see a problem', not 'there was none'."
        )

    st.caption(
        f"Recalculated with the settings you chose above: {plan_n:,} users over "
        f"{plan_duration_parts['total_days']} days, at a {alpha:.2f} false-alarm rate, "
        f"{power:.0%} chance of spotting the change, a {split_ratio:.0%} split, and past "
        f"behaviour predicting the outcome at {rho:.2f}."
    )

    if st.button("Lock this as the pre-registered plan", key="prereg_lock_button"):
        st.session_state[PREREG_PLAN] = build_preregistration(
            primary_metric=metric_name,
            metric_layer=layer,
            baseline=baseline,
            mde_relative=mde_relative,
            n_total=plan_n,
            ramp_days=plan_duration_parts["ramp_days"],
            enrolment_days=plan_duration_parts["enrolment_days"],
            maturation_days=plan_duration_parts["maturation_days"],
            daily_new_eligible=float(st.session_state.get(POWER_DAILY_NEW, 900.0)),
            split_ratio=split_ratio,
            rho=rho,
            alpha=alpha,
            power=power,
            transform=transform,
            n_primary_metrics=n_primary,
            planned_looks=planned_looks,
            guardrail_baseline=guardrail_baseline or None,
            decision_rule=decision_rule,
        )
        st.success(
            "Plan saved. Signals 03 and 04 will now hold the result up against it before showing "
            "you the lift."
        )

    locked = st.session_state.get(PREREG_PLAN)
    if locked is not None:
        show_preregistration(locked)


def render_power_section() -> None:
    """Render Signal 02: variance, duration, compliance, and the pre-registered plan."""
    render_section_rule()
    render_signal_header(
        "Signal 02",
        "Work out what the test can see, and what it will cost you.",
        "How many users you need comes down to three things: how jumpy the metric is, how small a change you would still act on, and how long you can wait for it. Get those on the table before anyone picks a launch date.",
    )
    render_section_note(
        "How many users, and by when",
        "Most tests that answer nothing were not short of traffic. They measured the wrong thing, or measured it on the wrong group of people.",
    )

    layer = render_metric_layer_control()
    st.caption(METRIC_LAYERS[layer])

    with st.expander("How many users do I need?", expanded=True):
        sd, mde_absolute, rho, alpha, power, n_total = render_continuous_sizing()

    with st.expander("How long will that take?"):
        duration = render_duration_planner(n_total)

    with st.expander("Check it against real data"):
        render_variance_and_simulation(sd, mde_absolute, alpha)

    with st.expander("How bold to go, and who actually takes it"):
        render_intensity_and_compliance(sd, alpha, power)

    with st.expander("Write the plan down", expanded=True):
        render_plan_lock(alpha, power, rho, duration)


class PlanAttestations(TypedDict):
    """Facts about the run that only the analyst can supply."""

    actual_days: int | None
    transform_applied: str
    analysed_as_itt: bool
    looks_taken: int
    population_size: int | None
    value_per_unit: float | None


def render_plan_attestations(key_prefix: str) -> PlanAttestations:
    """Collect the facts only the analyst knows, before the readout is computed."""
    plan = st.session_state.get(PREREG_PLAN)
    if plan is None:
        st.info(
            "No plan saved yet. Write one down in Signal 02 and this section will check what you "
            "actually got against what you promised: users, split, who was counted, how the "
            "metric was trimmed, and how often you looked."
        )
        return {
            "actual_days": None,
            "transform_applied": "none",
            "analysed_as_itt": True,
            "looks_taken": 1,
            "population_size": None,
            "value_per_unit": None,
        }

    show_preregistration(plan)
    attest_left, attest_middle, attest_right = st.columns(3)
    actual_days = int(
        attest_left.number_input(
            "Days it actually ran",
            min_value=0,
            value=int(plan["total_days"]),
            step=1,
            key=f"{key_prefix}_actual_days",
        )
    )
    transform_applied = attest_middle.selectbox(
        "How the metric was actually trimmed",
        ["none", "winsorise p99", "winsorise p95", "log"],
        index=["none", "winsorise p99", "winsorise p95", "log"].index(plan["transform"]),
        key=f"{key_prefix}_actual_transform",
    )
    looks_taken = int(
        attest_right.number_input(
            "Times you actually looked at the result",
            min_value=1,
            value=int(plan["planned_looks"]),
            step=1,
            key=f"{key_prefix}_actual_looks",
        )
    )
    analysed_as_itt = st.checkbox(
        "Everyone who entered the test is in these numbers, in the group they were put in",
        value=True,
        key=f"{key_prefix}_itt",
        help=(
            "Untick if the numbers only cover people who opted in, finished, or did something "
            "else after the test started."
        ),
    )

    impact_left, impact_right = st.columns(2)
    population_size = int(
        impact_left.number_input(
            "How many users a full rollout would reach",
            min_value=0,
            value=0,
            step=1000,
            key=f"{key_prefix}_population",
            help="Leave at zero to skip the money estimate.",
        )
    )
    value_per_unit = float(
        impact_right.number_input(
            "What one conversion is worth",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key=f"{key_prefix}_unit_value",
        )
    )

    return {
        "actual_days": actual_days,
        "transform_applied": transform_applied,
        "analysed_as_itt": analysed_as_itt,
        "looks_taken": looks_taken,
        "population_size": population_size or None,
        "value_per_unit": value_per_unit or None,
    }


def render_plan_check(
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    observed_rate: float,
    ci_relative: tuple[float, float],
    metrics_tested: int,
    attestations: PlanAttestations,
) -> None:
    """Verify the delivered experiment against the locked plan, then state the result."""
    plan: PreRegistration | None = st.session_state.get(PREREG_PLAN)
    if plan is None:
        return

    actual_days = attestations["actual_days"]
    total_n = n_control + n_treatment
    rows = verify_against_plan(
        plan=plan,
        actual_n_total=total_n,
        actual_split_ratio=n_treatment / total_n if total_n else 0.0,
        actual_days=actual_days,
        metrics_tested=metrics_tested,
        looks_taken=attestations["looks_taken"],
        transform_applied=attestations["transform_applied"],
        analysed_as_itt=attestations["analysed_as_itt"],
        maturation_complete=actual_days is None or actual_days >= plan["total_days"],
    )
    st.markdown("### What you promised, and what you got")
    show_plan_verification(rows)

    if baseline_rate <= 0:
        st.caption(
            "The control group averages zero or less, so a percentage change does not mean "
            "anything here. The plan check above still holds."
        )
        return

    st.markdown("### The result, in the order that matters")
    show_readout_summary(
        summarise_readout(
            baseline=baseline_rate,
            observed_rate_or_mean=observed_rate,
            ci_relative=ci_relative,
            mde_relative=plan["mde_relative"],
            population_size=attestations["population_size"],
            value_per_unit=attestations["value_per_unit"],
        )
    )
    st.caption(
        "A p-value only answers whether luck alone could have produced this. The decision needs "
        "the range above, held up against the smallest change you said was worth acting on. That "
        "is why the number you wrote down in Signal 02 is carried all the way through to here."
    )


def render_manual_section() -> None:
    """Render Signal 03: manual counts analysis with frequentist and Bayesian reads."""
    render_section_rule()
    render_signal_header(
        "Signal 03",
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

    with st.expander("Check this readout against the pre-registered plan"):
        manual_attestations = render_plan_attestations("manual")

    manual_left, manual_right = st.columns(2)
    with manual_left:
        st.markdown("### Control")
        visitors_a = int(st.number_input("Visitors A", min_value=1, value=1000, key=MANUAL_VISITORS_A))
        conversions_a = int(
            st.number_input("Conversions A", min_value=0, value=100, key=MANUAL_CONVERSIONS_A)
        )
    with manual_right:
        st.markdown("### Variant")
        visitors_b = int(st.number_input("Visitors B", min_value=1, value=1000, key=MANUAL_VISITORS_B))
        conversions_b = int(
            st.number_input("Conversions B", min_value=0, value=115, key=MANUAL_CONVERSIONS_B)
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
                    alpha_threshold=guardrails["adjusted_alpha"],
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
                    baseline_for_relative_tolerance=cr_a,
                )
                show_bayesian_decision(
                    recommendation,
                    confidence,
                    expected_loss=manual_bayes["expected_loss"],
                )

            manual_ci = confidence_interval_binary(cr_a, cr_b, visitors_a, visitors_b)
            render_plan_check(
                n_control=visitors_a,
                n_treatment=visitors_b,
                baseline_rate=cr_a,
                observed_rate=cr_b,
                ci_relative=manual_ci,
                metrics_tested=manual_n_comparisons,
                attestations=manual_attestations,
            )


def render_csv_section() -> None:
    """Render Signal 04: raw CSV audit with LLM-assisted column mapping."""
    render_section_rule()
    render_signal_header(
        "Signal 04",
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

    with st.expander("Check this readout against the pre-registered plan"):
        csv_attestations = render_plan_attestations("csv")

    uploaded_file = st.file_uploader("Upload a results CSV", type="csv", key=CSV_UPLOAD)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        show_data_quality(df)
        st.write("Preview:", df.head(3))

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

                    test_results: FrequentistTestResult
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
                        small_sample = (
                            n_a <= SMALL_SAMPLE_THRESHOLD or n_b <= SMALL_SAMPLE_THRESHOLD
                        )
                        effect_size_method: EffectSizeMethod = (
                            "averaged" if small_sample else "pooled"
                        )
                        test_results = welch_t_test(
                            group_a, group_b, effect_size_method=effect_size_method
                        )
                        if small_sample:
                            ci_lower, ci_upper = bootstrap_ci_relative_lift_continuous(
                                group_a, group_b
                            )
                            st.caption(
                                f"Small sample (≤{SMALL_SAMPLE_THRESHOLD} in a group): using a "
                                "percentile bootstrap CI and the unequal-variance effect size, "
                                "which avoid the normal approximation."
                            )
                        else:
                            ci_lower, ci_upper = confidence_interval_continuous(group_a, group_b)

                    if csv_method in ["Frequentist (P-values)", "Both"]:
                        guardrails = build_frequentist_guardrails(
                            n_comparisons=csv_n_comparisons,
                            peeked_early=csv_peeked_early,
                        )
                        st.markdown("### Frequentist read")
                        show_frequentist_guardrails(guardrails)
                        show_frequentist_results(
                            test_results,
                            ci_lower,
                            ci_upper,
                            float(mean_a),
                            float(mean_b),
                            [str(groups[0]), str(groups[1])],
                            alpha_threshold=guardrails["adjusted_alpha"],
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
                                baseline_for_relative_tolerance=float(mean_a),
                            )
                            show_bayesian_decision(
                                recommendation,
                                confidence,
                                group_name=str(groups[1]),
                                expected_loss=csv_bayes["expected_loss"],
                            )
                        else:
                            st.info("Bayesian analysis is only available for binary metrics.")

                    render_plan_check(
                        n_control=n_a,
                        n_treatment=n_b,
                        baseline_rate=float(mean_a),
                        observed_rate=float(mean_b),
                        ci_relative=(ci_lower, ci_upper),
                        metrics_tested=csv_n_comparisons,
                        attestations=csv_attestations,
                    )

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


def render_causal_section() -> None:
    """Render Signal 05: causal fallback method selector and analysis."""
    render_section_rule()
    render_signal_header(
        "Signal 05",
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
        key=CAUSAL_HAS_CUTOFF,
    )
    has_control_choice = selector_center.selectbox(
        "Clean control?",
        ["No", "Yes (unaffected users)"],
        key=CAUSAL_HAS_CONTROL,
    )
    is_opt_in_choice = selector_right.selectbox(
        "User self-selection?",
        ["No (forced)", "Yes (opt-in)"],
        key=CAUSAL_IS_OPT_IN,
    )

    recommended_method = select_causal_method(
        has_cutoff=has_cutoff_choice.startswith("Yes"),
        has_clean_control=has_control_choice.startswith("Yes"),
        is_opt_in=is_opt_in_choice.startswith("Yes"),
    )
    st.success(f"Recommended method: {recommended_method}")

    if recommended_method == "Difference-in-Differences (DiD)":
        _render_did_analysis()
    elif recommended_method == "Regression Discontinuity (RDD)":
        _render_rdd_analysis()
    else:
        _render_causal_codegen(recommended_method)


def _render_did_analysis() -> None:
    """Upload, map, and run a Difference-in-Differences analysis."""
    st.markdown(
        "Upload panel data with a unit ID, time period, treatment flag, and outcome metric."
    )
    did_file = st.file_uploader("Upload CSV for DiD", type="csv", key=DID_UPLOAD)

    if did_file is None:
        return

    df_did = pd.read_csv(did_file)
    show_data_quality(df_did)
    st.write("Preview:", df_did.head(3))

    if not st.button("Run DiD analysis", key="did_analyze"):
        return

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

    if not mapping:
        return

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

        if not st.button("Calculate DiD effect", key="did_calc"):
            return

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


def _render_rdd_analysis() -> None:
    """Upload, map, and run a Regression Discontinuity analysis."""
    st.markdown(
        "Upload data with a running variable, treatment flag, and outcome metric."
    )
    rdd_file = st.file_uploader("Upload CSV for RDD", type="csv", key=RDD_UPLOAD)

    if rdd_file is None:
        return

    df_rdd = pd.read_csv(rdd_file)
    show_data_quality(df_rdd)
    st.write("Preview:", df_rdd.head(3))

    if not st.button("Run RDD analysis", key="rdd_analyze"):
        return

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

    if not mapping:
        return

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

        if not st.button("Calculate RDD effect", key="rdd_calc"):
            return

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
            st.success(f"Significant discontinuity (p={rdd_result['p_value']:.4f})")
        else:
            st.warning(f"No significant discontinuity (p={rdd_result['p_value']:.4f})")

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


def _render_causal_codegen(recommended_method: str) -> None:
    """Render code-generation fallback for PSM and CausalImpact."""
    st.info(
        "This path is still a code-generation fallback. The tool will suggest a script, "
        "but it will not pretend the estimator is fully productized in-app."
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


# ── Main page execution ───────────────────────────────────────────────────────

review_focus = render_sidebar()
page_snapshot = build_page_snapshot(review_focus, ai_enabled)

render_hero_card(
    kicker=page_snapshot["kicker"],
    title=page_snapshot["title"],
    body=page_snapshot["body"],
    pills=[str(pill) for pill in page_snapshot["pills"] if pill],
)
render_summary_cards(page_snapshot["cards"])

if not any(
    read_uploaded_dataframe(key) is not None
    for key in UPLOAD_KEYS
):
    render_empty_state()

render_design_section()
render_power_section()
render_manual_section()
render_csv_section()
render_causal_section()
