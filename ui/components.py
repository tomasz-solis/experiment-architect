"""Shared UI components."""

import streamlit as st
import pandas as pd


def show_data_quality(df):
    """Display data quality metrics."""
    st.markdown("**Data Quality**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Missing Values", df.isnull().sum().sum())
    col3.metric("Duplicate Rows", df.duplicated().sum())

    with st.expander("View full dataset"):
        st.dataframe(df)


def show_data_preview(df, n_rows=3):
    """Show data preview."""
    st.write("Preview:", df.head(n_rows))


def show_srm_warning(ratio):
    """Show Sample Ratio Mismatch warning if needed."""
    if abs(ratio - 0.5) > 0.05:
        st.warning(f"Sample Ratio Mismatch: {ratio:.1%} vs expected 50%. Check your randomization.")


def show_frequentist_results(test_results, ci_lower, ci_upper, mean_a, mean_b, groups):
    """Display frequentist analysis results."""
    st.caption(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    st.caption(f"Effect Size ({test_results['effect_size_label']}): "
              f"{test_results['effect_size']:.3f} | P-value: {test_results['p_value']:.4f}")

    # Declare winner
    if test_results['p_value'] < 0.05:
        winner = groups[1] if mean_b > mean_a else groups[0]
        st.success(f"Winner: {winner} is statistically significant.")
    else:
        st.warning("Result is not statistically significant.")

    st.caption(f"Method: {test_results['test_name']}")


def show_bayesian_results(bayes_results, groups):
    """Display Bayesian analysis results."""
    col1, col2 = st.columns(2)
    col1.metric("P(Variant Beats Control)", f"{bayes_results['prob_b_wins']:.1%}")
    col2.metric("Expected Loss (if wrong)", f"{bayes_results['expected_loss']:.3%}")

    st.caption(f"Posterior: {groups[0]} ~ Beta({bayes_results['alpha_a']}, {bayes_results['beta_a']}), "
              f"{groups[1]} ~ Beta({bayes_results['alpha_b']}, {bayes_results['beta_b']})")


def show_bayesian_decision(prob_b_wins, group_name="Variant B"):
    """Show Bayesian decision recommendation."""
    if prob_b_wins > 0.95:
        st.success(f"High confidence: Ship {group_name}")
    elif prob_b_wins > 0.75:
        st.info(f"Moderate confidence: Consider shipping {group_name}")
    elif prob_b_wins < 0.25:
        st.error(f"High confidence: Keep Control")
    else:
        st.warning("Uncertain: Need more data")
