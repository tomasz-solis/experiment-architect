"""Hero and summary-card content for each review lens.

Each builder reads the relevant Streamlit session state and returns the
top-of-page snapshot (kicker, title, body, pills, cards) for one lens. They are
kept out of ``app.py`` so the lens logic is cohesive and the app script stays a
thin orchestrator. ``ai_enabled`` is passed in rather than read from a global so
the builders depend only on their inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from config import ALPHA, DEFAULT_POWER
from stats.bayesian import beta_binomial_analysis, get_decision_recommendation
from stats.causal import select_causal_method
from stats.decision_cards import (
    build_bayesian_card,
    build_count_mismatch_card,
    build_input_mismatch_summary,
    build_manual_frequentist_card,
    build_weakest_signal_card,
)
from stats.frequentist import (
    build_frequentist_guardrails,
    calculate_lift,
    calculate_sample_size,
    check_srm,
    chi_squared_test,
)
from stats.power import design_effect, plan_duration, sample_size_continuous
from stats.prereg import PreRegistration
from stats.sanity import run_all_checks, severity_rank
from ui.formatting import build_card, duration_tone, first_sentence
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
    MANUAL_N_COMPARISONS,
    MANUAL_PEEKED_EARLY,
    MANUAL_VISITORS_A,
    MANUAL_VISITORS_B,
    POWER_ALPHA,
    POWER_CLUSTER_SIZE,
    POWER_DAILY_NEW,
    POWER_ICC,
    POWER_MATURATION,
    POWER_MDE_ABS,
    POWER_POWER,
    POWER_RAMP,
    POWER_RHO,
    POWER_SD,
    PREREG_PLAN,
    RDD_UPLOAD,
    read_uploaded_dataframe,
)

REVIEW_FOCI = [
    "Experiment design",
    "Power and plan",
    "Manual result read",
    "Raw CSV audit",
    "Causal fallback",
]


def design_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the design-review lens."""
    baseline = float(st.session_state.get(MAIN_BASELINE, 10.0)) / 100
    mde = float(st.session_state.get(MAIN_MDE, 10.0)) / 100
    daily_traffic = int(st.session_state.get(MAIN_TRAFFIC, 5000))
    split_ratio = float(st.session_state.get(MAIN_SPLIT, 50)) / 100
    alpha = float(st.session_state.get(POWER_ALPHA, ALPHA))
    power = float(st.session_state.get(POWER_POWER, DEFAULT_POWER))
    rho = float(st.session_state.get(POWER_RHO, 0.0))
    avg_cluster_size = float(st.session_state.get(POWER_CLUSTER_SIZE, 1.0))
    icc = float(st.session_state.get(POWER_ICC, 0.0))
    cluster_effect = design_effect(avg_cluster_size, icc)

    size = calculate_sample_size(baseline, mde, daily_traffic, split_ratio)
    weeks_required = max(1, int(np.ceil(size["days"] / 7)))
    checks = run_all_checks(
        baseline,
        mde,
        daily_traffic,
        weeks_required,
        split_ratio=split_ratio,
        alpha=alpha,
        power=power,
        rho=rho,
        cluster_design_effect=cluster_effect,
    )
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


def power_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the power-and-plan lens."""
    sd = float(st.session_state.get(POWER_SD, 120.0))
    mde_absolute = float(st.session_state.get(POWER_MDE_ABS, 4.0))
    rho = float(st.session_state.get(POWER_RHO, 0.0))
    split_ratio = float(st.session_state.get(MAIN_SPLIT, 50)) / 100
    daily_new = float(st.session_state.get(POWER_DAILY_NEW, 900))
    maturation = int(st.session_state.get(POWER_MATURATION, 30))
    ramp = int(st.session_state.get(POWER_RAMP, 0))
    avg_cluster_size = float(st.session_state.get(POWER_CLUSTER_SIZE, 1.0))
    icc = float(st.session_state.get(POWER_ICC, 0.0))

    try:
        cluster_effect = design_effect(avg_cluster_size, icc)
        size = sample_size_continuous(
            sd=sd,
            mde_absolute=mde_absolute,
            split_ratio=split_ratio,
            rho=rho,
            cluster_design_effect=cluster_effect,
        )
        duration = plan_duration(
            n_total=size["n_total"],
            daily_new_eligible=daily_new,
            maturation_days=maturation,
            ramp_days=ramp,
        )
    except ValueError as error:
        return {
            "kicker": "Power and plan",
            "title": "The planning inputs do not resolve yet.",
            "body": str(error),
            "pills": ["Continuous outcome", "Variance reduction", "Duration"],
            "cards": [
                build_card("Review lens", "Power and plan", "Fix the inputs to size the test.", "amber", anchor=True),
            ],
        }

    saved = 1 - size["variance_retained"]
    plan_locked = st.session_state.get(PREREG_PLAN) is not None

    return {
        "kicker": "Power, variance and duration",
        "title": "Buy the effect with variance reduction before you buy it with traffic.",
        "body": (
            "Sample scales with the square of the noise and the inverse square of the effect. "
            "That makes the standard deviation, the pre-period covariate, and the maturation "
            "window the three numbers that actually set the launch date."
        ),
        "pills": [
            "Continuous outcome",
            f"SD {sd:,.0f}",
            f"MDE {mde_absolute:,.2f}",
            f"CUPED rho {rho:.2f}",
        ],
        "cards": [
            build_card("Review lens", "Power and plan", "Size on variance, not on hope.", "blue", anchor=True),
            build_card("Required sample", f"{size['n_total']:,}", "Both arms, at the current split.", "mint"),
            build_card(
                "Calendar time",
                f"{duration['total_days']} days",
                duration["binding_constraint"],
                duration_tone(duration["total_days"]),
            ),
            build_card(
                "Traffic saved by CUPED" if saved > 0 else "Pre-registration",
                f"{saved:.0%}" if saved > 0 else ("Locked" if plan_locked else "Not locked"),
                "Variance removed by the pre-period covariate." if saved > 0
                else "Lock the plan so the readout can verify it.",
                "mint" if (saved > 0 or plan_locked) else "amber",
            ),
        ],
    }


def manual_snapshot() -> dict[str, Any]:
    """Build hero and summary content for the manual-result lens."""
    visitors_a = int(st.session_state.get(MANUAL_VISITORS_A, 1000))
    conversions_a = int(st.session_state.get(MANUAL_CONVERSIONS_A, 100))
    visitors_b = int(st.session_state.get(MANUAL_VISITORS_B, 1000))
    conversions_b = int(st.session_state.get(MANUAL_CONVERSIONS_B, 115))
    n_comparisons = int(st.session_state.get(MANUAL_N_COMPARISONS, 1))
    peeked_early = bool(st.session_state.get(MANUAL_PEEKED_EARLY, False))

    if conversions_a > visitors_a or conversions_b > visitors_b:
        mismatch = build_input_mismatch_summary()
        mismatch_card = build_count_mismatch_card()
        return {
            **mismatch,
            "cards": [
                build_card("Review lens", "Manual read", "Use counts when you do not need raw rows.", "blue", anchor=True),
                build_card("Validation", mismatch_card["value"], mismatch_card["meta"], mismatch_card["tone"]),
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
        baseline_for_relative_tolerance=cr_a,
    )
    guardrails = build_frequentist_guardrails(
        n_comparisons=n_comparisons,
        peeked_early=peeked_early,
    )
    adjusted_alpha = guardrails["adjusted_alpha"]
    plan: PreRegistration | None = st.session_state.get(PREREG_PLAN)
    expected_share_b = plan["split_ratio"] if plan is not None else 0.5
    has_srm = check_srm(visitors_a, visitors_b, expected_share_b=expected_share_b)["has_mismatch"]

    frequentist_card = build_manual_frequentist_card(
        p_value=test_results["p_value"],
        adjusted_alpha=adjusted_alpha,
        has_srm=has_srm,
        peeked_early=peeked_early,
        chi_square_valid=test_results["chi_square_valid"],
    )
    weakest_card = build_weakest_signal_card(
        has_srm=has_srm,
        chi_square_valid=test_results["chi_square_valid"],
        peeked_early=peeked_early,
        alpha_adjusted=guardrails["alpha_adjusted"],
        adjusted_alpha=adjusted_alpha,
    )
    bayesian_card = build_bayesian_card(
        recommendation=recommendation,
        confidence=confidence,
        expected_loss=bayes["expected_loss"],
    )

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
            build_card("Frequentist read", frequentist_card["value"], frequentist_card["meta"], frequentist_card["tone"]),
            build_card("Bayesian read", bayesian_card["value"], bayesian_card["meta"], bayesian_card["tone"]),
            build_card("Weakest signal", weakest_card["value"], weakest_card["meta"], weakest_card["tone"]),
        ],
    }


def csv_snapshot(ai_enabled: bool) -> dict[str, Any]:
    """Build hero and summary content for the raw-CSV lens."""
    df = read_uploaded_dataframe(CSV_UPLOAD)
    uploaded_file = st.session_state.get(CSV_UPLOAD)

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


def causal_snapshot(ai_enabled: bool) -> dict[str, Any]:
    """Build hero and summary content for the causal-fallback lens."""
    has_cutoff = str(st.session_state.get(CAUSAL_HAS_CUTOFF, "No")).startswith("Yes")
    has_clean_control = str(st.session_state.get(CAUSAL_HAS_CONTROL, "No")).startswith("Yes")
    is_opt_in = str(st.session_state.get(CAUSAL_IS_OPT_IN, "No (forced)")).startswith("Yes")
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

    did_df = read_uploaded_dataframe(DID_UPLOAD)
    rdd_df = read_uploaded_dataframe(RDD_UPLOAD)
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


def build_page_snapshot(review_focus: str, ai_enabled: bool) -> dict[str, Any]:
    """Build top-of-page hero and summary content for the selected lens."""
    if review_focus == "Experiment design":
        return design_snapshot()
    if review_focus == "Power and plan":
        return power_snapshot()
    if review_focus == "Manual result read":
        return manual_snapshot()
    if review_focus == "Raw CSV audit":
        return csv_snapshot(ai_enabled)
    if review_focus == "Causal fallback":
        return causal_snapshot(ai_enabled)
    raise KeyError(f"Unknown review lens: {review_focus!r}")
