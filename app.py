import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy.stats import ttest_ind, chi2_contingency, beta
from dotenv import load_dotenv

from config import Z_ALPHA, Z_BETA, ALPHA, BAYESIAN_SAMPLES
from llm.client import create_llm_client, ask_agent as llm_ask_agent

load_dotenv()

st.set_page_config(page_title="Experiment Architect", page_icon="🏗️", layout="centered")

client, ai_enabled, llm_provider = create_llm_client()

def ask_agent(system_role, user_prompt, json_mode=False):
    return llm_ask_agent(client, llm_provider, ai_enabled, system_role, user_prompt, json_mode)
st.title("🏗️ The Experiment Architect")
st.markdown("**Plan. Execute. Analyze.** The Decision Scientist's Companion.")

tab_design, tab_analyze = st.tabs(["Design Experiment", "Analyze Results"])

with tab_design:
    st.header("Experiment Design")
    
    experiment_type = st.radio(
        "Can you randomize users into groups?",
        ["Yes (A/B Test)", "No (Observational Study)"],
        horizontal=True
    )
    
    st.divider()

    if experiment_type == "Yes (A/B Test)":
        st.subheader("A/B Test Calculator")

        with st.expander("What is a realistic MDE?"):
            st.markdown("**MDE (Minimum Detectable Effect)**: The smallest change you can reliably detect with your traffic and time constraints.")
            st.markdown("**Baseline Conversion**: Your current conversion rate before any changes (e.g., 10% of visitors buy).")
            st.markdown("---")
            st.markdown("Input your constraints below to calculate what's actually detectable:")

            c_wiz1, c_wiz2 = st.columns(2)
            wiz_weeks = c_wiz1.slider("Max Wait Time (Weeks)", 1, 12, 4)
            wiz_traffic = c_wiz2.number_input("Avg Daily Visitors", 100, 1000000, 5000)
            wiz_base = c_wiz2.number_input("Baseline Conversion (%)", 0.1, 99.0, 10.0, key="wiz_base") / 100
            
            if st.button("Calculate Minimum Detectable Effect"):
                total_n = wiz_traffic * (wiz_weeks * 7)
                z_score = (Z_ALPHA + Z_BETA)**2
                pooled_var = 2 * wiz_base * (1 - wiz_base)

                if total_n < 100:
                    st.error("⚠️ Not enough traffic. You need at least 100 total visitors.")
                elif wiz_base < 0.001 or wiz_base > 0.999:
                    st.error("⚠️ Baseline conversion must be between 0.1% and 99.9%")
                else:
                    feasible_mde = np.sqrt((z_score * pooled_var) / total_n) / wiz_base

                    if feasible_mde > 10.0:
                        st.error(f"📉 Your detectable lift is **{feasible_mde:.1%}** - way too high. Increase traffic or wait longer.")
                    else:
                        st.info(f"📉 In {wiz_weeks} weeks, the smallest lift you can detect is **{feasible_mde:.1%}**.")
                        st.caption("If your feature impact is smaller than this, you will never reach significance.")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            baseline = st.number_input("Baseline Conversion (%)", 0.1, 99.0, 10.0, step=0.5, key="main_base") / 100
            mde = st.number_input("Target Lift (Relative %)", 1.0, 500.0, 10.0, step=1.0) / 100
        with c2:
            daily_traffic = st.number_input("Daily Visitors (Total)", 100, 1000000, 5000, step=100, key="main_traffic")
            split_ratio = st.slider("Traffic Allocation (Variant %)", 1, 99, 50) / 100

        # AI sanity check
        feature_context = st.text_input("Context (Optional)", placeholder="e.g. 'Changing CTA color on checkout page'")
        
        if st.button("Sanity Check: Is this realistic?"):
            sys_prompt = """
            You are a Senior Product Data Scientist at a B2B Fintech.
            Review the user's experiment parameters.
            Return a JSON object with:
            - "status": "Green" (Realistic), "Yellow" (Aggressive), or "Red" (Unrealistic).
            - "reason": A 1-sentence explanation based on industry benchmarks.
            """
            user_prompt = f"Feature: {feature_context}\nBaseline: {baseline:.1%}\nTarget Lift: {mde:.1%}"
            
            response = ask_agent(sys_prompt, user_prompt, json_mode=True)
            if response:
                data = json.loads(response)
                status = data.get("status", "Yellow")
                reason = data.get("reason", "")
                
                if status == "Green": st.success(f"✅ **Realistic:** {reason}")
                elif status == "Yellow": st.warning(f"⚠️ **Aggressive:** {reason}")
                else: st.error(f"🛑 **Unrealistic:** {reason}")

        # Calculate sample size
        p2 = baseline * (1 + mde)
        delta = p2 - baseline
        split_factor = (1 / split_ratio) + (1 / (1 - split_ratio))
        pooled_var = (baseline * (1-baseline) + p2 * (1-p2)) / 2
        z_score = (Z_ALPHA + Z_BETA)**2
        n_total = z_score * pooled_var * split_factor / (delta**2)
        days = np.ceil(n_total / daily_traffic)

        st.info(f"⏱️ **Estimated Duration:** {int(days)} Days (Total Sample: {int(n_total):,})")

        if split_ratio != 0.5:
             loss_pct = int((((split_factor/4)-1)*100))
             st.warning(f"⚠️ **Trade-off:** This split is **{loss_pct}% slower** than a 50/50 split.")

    else:
        st.subheader("🕵️ Causal Method Selector")
        st.caption("Expert System for Quasi-Experiments.")

        c1, c2, c3 = st.columns(3)
        q_cutoff = c1.selectbox("Strict Cutoff?", ["No", "Yes (e.g. Credit Score > 600)"])
        q_control = c2.selectbox("Clean Control?", ["No", "Yes (Unaffected Users)"])
        q_optin = c3.selectbox("User Self-Selection?", ["No (Forced)", "Yes (Opt-in)"])
        
        # Logic Tree
        method = "CausalImpact" 
        if q_cutoff.startswith("Yes"): method = "Regression Discontinuity (RDD)"
        elif q_control.startswith("Yes") and q_optin.startswith("No"): method = "Difference-in-Differences (DiD)"
        elif q_optin.startswith("Yes"): method = "Propensity Score Matching (PSM)"
            
        st.success(f"Recommended Method: {method}")

        st.markdown("---")
        st.subheader("Analysis")

        # DiD Implementation
        if method == "Difference-in-Differences (DiD)":
            st.markdown("Upload panel data with: unit ID, time period, treatment indicator, outcome metric")
            did_file = st.file_uploader("Upload CSV for DiD", type="csv", key="did_upload")

            if did_file:
                df_did = pd.read_csv(did_file)

                # Data quality check
                st.markdown("**Data Preview**")
                col1, col2 = st.columns(2)
                col1.metric("Total Rows", f"{len(df_did):,}")
                col2.metric("Missing Values", df_did.isnull().sum().sum())

                with st.expander("View full dataset"):
                    st.dataframe(df_did)

                st.write("Preview:", df_did.head(3))

                if st.button("Run DiD Analysis", key="did_analyze"):
                    # AI column mapping
                    sys_prompt = """
                    You are a causal inference expert. Identify these columns:
                    - unit_col: user/entity ID column
                    - time_col: date or period column
                    - treatment_col: binary treatment indicator (0/1)
                    - outcome_col: outcome metric

                    Return JSON with these 4 keys. Use exact column names from the dataset.
                    """

                    response = ask_agent(sys_prompt, f"Columns: {list(df_did.columns)}\n\nPreview:\n{df_did.head(3).to_markdown()}", json_mode=True)

                    if response:
                        try:
                            config = json.loads(response)
                            unit_col = config['unit_col']
                            time_col = config['time_col']
                            treatment_col = config['treatment_col']
                            outcome_col = config['outcome_col']

                            st.success(f"Mapped: Treatment={treatment_col}, Outcome={outcome_col}, Time={time_col}, Unit={unit_col}")

                            # Ask for intervention date
                            st.markdown("**Define Intervention Point**")
                            unique_times = sorted(df_did[time_col].unique())
                            st.write(f"Available time periods: {unique_times}")

                            intervention_input = st.text_input("Enter intervention date/period",
                                                              placeholder="e.g., 2024-01-01 or period_2")

                            if intervention_input and st.button("Calculate DiD Effect", key="did_calc"):
                                try:
                                    import statsmodels.api as sm
                                except ImportError:
                                    st.error("statsmodels not installed. Run: pip install statsmodels")
                                    st.stop()

                                # Create treatment indicators
                                df_did['post'] = (df_did[time_col] >= intervention_input).astype(int)
                                df_did['treated'] = df_did[treatment_col].astype(int)
                                df_did['did'] = df_did['treated'] * df_did['post']

                                # Run DiD regression
                                X = df_did[['treated', 'post', 'did']]
                                X = sm.add_constant(X)
                                y = df_did[outcome_col]

                                model = sm.OLS(y, X).fit()

                                # Extract DiD coefficient
                                did_coef = model.params['did']
                                did_se = model.bse['did']
                                did_pval = model.pvalues['did']
                                did_ci = model.conf_int().loc['did']

                                # Display results
                                st.markdown("**DiD Treatment Effect**")
                                st.metric("Average Treatment Effect", f"{did_coef:.4f}")
                                st.caption(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")

                                if did_pval < 0.05:
                                    st.success(f"Statistically significant effect (p={did_pval:.4f})")
                                else:
                                    st.warning(f"Not statistically significant (p={did_pval:.4f})")

                                # Show full regression table
                                with st.expander("Full Regression Output"):
                                    st.text(model.summary())

                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

        # RDD Implementation
        elif method == "Regression Discontinuity (RDD)":
            st.markdown("Upload data with: running variable (e.g., credit score), treatment indicator, outcome")
            rdd_file = st.file_uploader("Upload CSV for RDD", type="csv", key="rdd_upload")

            if rdd_file:
                df_rdd = pd.read_csv(rdd_file)

                # Data quality check
                st.markdown("**Data Preview**")
                col1, col2 = st.columns(2)
                col1.metric("Total Rows", f"{len(df_rdd):,}")
                col2.metric("Missing Values", df_rdd.isnull().sum().sum())

                with st.expander("View full dataset"):
                    st.dataframe(df_rdd)

                st.write("Preview:", df_rdd.head(3))

                if st.button("Run RDD Analysis", key="rdd_analyze"):
                    # AI column mapping
                    sys_prompt = """
                    You are a causal inference expert. Identify these columns:
                    - running_var: the running variable (e.g., credit score, age, test score)
                    - treatment_col: binary treatment indicator (0/1)
                    - outcome_col: outcome metric

                    Return JSON with these 3 keys. Use exact column names.
                    """

                    response = ask_agent(sys_prompt, f"Columns: {list(df_rdd.columns)}\n\nPreview:\n{df_rdd.head(3).to_markdown()}", json_mode=True)

                    if response:
                        try:
                            config = json.loads(response)
                            running_var = config['running_var']
                            treatment_col = config['treatment_col']
                            outcome_col = config['outcome_col']

                            st.success(f"Mapped: Running Variable={running_var}, Treatment={treatment_col}, Outcome={outcome_col}")

                            # Ask for cutoff
                            cutoff = st.number_input("Treatment cutoff value",
                                                    value=float(df_rdd[running_var].median()))

                            if st.button("Calculate RDD Effect", key="rdd_calc"):
                                try:
                                    import statsmodels.api as sm
                                except ImportError:
                                    st.error("statsmodels not installed. Run: pip install statsmodels")
                                    st.stop()

                                # Center running variable around cutoff
                                df_rdd['centered'] = df_rdd[running_var] - cutoff
                                df_rdd['treated'] = df_rdd[treatment_col].astype(int)

                                # Interaction term
                                df_rdd['interaction'] = df_rdd['treated'] * df_rdd['centered']

                                # Fit model
                                X = df_rdd[['treated', 'centered', 'interaction']]
                                X = sm.add_constant(X)
                                y = df_rdd[outcome_col]

                                model = sm.OLS(y, X).fit()

                                # Extract treatment effect
                                rdd_coef = model.params['treated']
                                rdd_pval = model.pvalues['treated']
                                rdd_ci = model.conf_int().loc['treated']

                                # Display results
                                st.markdown("**RDD Treatment Effect**")
                                st.metric("Effect at Cutoff", f"{rdd_coef:.4f}")
                                st.caption(f"95% CI: [{rdd_ci[0]:.4f}, {rdd_ci[1]:.4f}]")

                                if rdd_pval < 0.05:
                                    st.success(f"Statistically significant discontinuity (p={rdd_pval:.4f})")
                                else:
                                    st.warning(f"No significant discontinuity (p={rdd_pval:.4f})")

                                # Show full regression
                                with st.expander("Full Regression Output"):
                                    st.text(model.summary())

                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

        # PSM and CausalImpact: Code generator only
        else:
            st.info("Code generation mode for this method")

            c_dwh = st.selectbox("Target Database:", ["BigQuery", "Snowflake", "Redshift", "Local CSV"])
            user_context = st.text_area("Paste Column Names or SQL Schema:", height=100,
                                        placeholder="e.g., user_id, transaction_date, treatment_flag, total_spend")

            if st.button("Generate Python Script"):
                prompt = f"""
                You are a Senior Data Scientist. Write a Python script for {method}.
                Target: {c_dwh}.
                User Context: {user_context}.

                Requirements:
                1. Use standard libraries (CausalImpact for time series, sklearn for PSM).
                2. Include specific connector code if needed.
                3. Include comments explaining statistical assumptions.
                4. Keep it practical and executable.
                """
                result = ask_agent(prompt, "Write the code.")
                if result:
                    st.code(result, language="python")

with tab_analyze:
    st.header("Results Analysis")
    
    source_type = st.radio("Input Method:", 
                           ["Summary Stats (Manual)", "Raw CSV (Automated Analysis)", "SQL Query (Code Gen)"], 
                           horizontal=True)
    
    st.divider()

    if source_type == "Summary Stats (Manual)":
        # Analysis method selection
        analysis_method = st.radio("Analysis Method:",
                                   ["Frequentist (P-values)", "Bayesian (Probability)", "Both"],
                                   horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Control")
            vis_a = st.number_input("Visitors A", 1000)
            conv_a = st.number_input("Conversions A", 100)
        with c2:
            st.markdown("### Variant")
            vis_b = st.number_input("Visitors B", 1000)
            conv_b = st.number_input("Conversions B", 115)

        if st.button("Calculate Result"):
            cr_a = conv_a / vis_a
            cr_b = conv_b / vis_b
            lift = (cr_b - cr_a) / cr_a

            # Sample Ratio Mismatch check
            expected_ratio = vis_a / (vis_a + vis_b)
            if abs(expected_ratio - 0.5) > 0.05:
                st.warning(f"Sample Ratio Mismatch: {expected_ratio:.1%} vs expected 50%. Check your randomization.")

            # Point estimate
            st.metric("Relative Lift", f"{lift:.2%}")

            # Frequentist Analysis
            if analysis_method in ["Frequentist (P-values)", "Both"]:
                st.markdown("### Frequentist Analysis")

                # Confidence intervals on lift
                se_a = np.sqrt(cr_a * (1 - cr_a) / vis_a)
                se_b = np.sqrt(cr_b * (1 - cr_b) / vis_b)
                se_diff = np.sqrt(se_a**2 + se_b**2)

                margin = Z_ALPHA * se_diff
                ci_lower = ((cr_b - margin) - cr_a) / cr_a
                ci_upper = ((cr_b + margin) - cr_a) / cr_a

                st.caption(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")

                # Statistical test
                contingency_table = [
                    [int(conv_a), int(vis_a - conv_a)],
                    [int(conv_b), int(vis_b - conv_b)]
                ]
                stat, p_val, _, _ = chi2_contingency(contingency_table)

                # Effect size (Cramér's V)
                n = vis_a + vis_b
                cramers_v = np.sqrt(stat / n)

                st.caption(f"Effect Size (Cramér's V): {cramers_v:.3f} | P-value: {p_val:.4f}")

                # Verdict
                if p_val < ALPHA:
                    if cr_b > cr_a:
                        st.success("Variant B is statistically significant.")
                    else:
                        st.error("Variant B is statistically worse.")
                else:
                    st.warning("Result is not statistically significant.")

            # Bayesian Analysis
            if analysis_method in ["Bayesian (Probability)", "Both"]:
                if analysis_method == "Both":
                    st.divider()
                st.markdown("### Bayesian Analysis")

                # Beta-Binomial conjugate prior
                alpha_prior, beta_prior = 1, 1

                # Posterior distributions
                alpha_a = alpha_prior + conv_a
                beta_a = beta_prior + (vis_a - conv_a)
                alpha_b = alpha_prior + conv_b
                beta_b = beta_prior + (vis_b - conv_b)

                # Sample from posteriors
                samples_a = beta.rvs(alpha_a, beta_a, size=BAYESIAN_SAMPLES)
                samples_b = beta.rvs(alpha_b, beta_b, size=BAYESIAN_SAMPLES)

                # P(B > A)
                prob_b_wins = (samples_b > samples_a).mean()

                # Expected loss if you ship variant B
                loss_if_ship_b = np.maximum(samples_a - samples_b, 0).mean()

                col1, col2 = st.columns(2)
                col1.metric("P(Variant Beats Control)", f"{prob_b_wins:.1%}")
                col2.metric("Expected Loss (if ship)", f"{loss_if_ship_b:.3%}")

                st.caption(f"Posterior: Control ~ Beta({alpha_a}, {beta_a}), Variant ~ Beta({alpha_b}, {beta_b})")

                # Decision recommendation
                if prob_b_wins > 0.95:
                    st.success("High confidence: Ship Variant B")
                elif prob_b_wins > 0.75:
                    st.info("Moderate confidence: Consider shipping Variant B")
                elif prob_b_wins < 0.25:
                    st.error("High confidence: Keep Control")
                else:
                    st.warning("Uncertain: Need more data")

    elif source_type == "Raw CSV (Automated Analysis)":
        st.info("Upload your results. Columns are mapped automatically, then Python runs the statistical test.")

        # Analysis method selection
        csv_analysis_method = st.radio("Analysis Method:",
                                       ["Frequentist (P-values)", "Bayesian (Probability)", "Both"],
                                       horizontal=True,
                                       key="csv_method")

        uploaded_file = st.file_uploader("Upload .csv", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # Data quality check
            st.markdown("**Data Quality**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(df):,}")
            col2.metric("Missing Values", df.isnull().sum().sum())
            col3.metric("Duplicate Rows", df.duplicated().sum())

            with st.expander("View full dataset"):
                st.dataframe(df)

            st.write("Preview:", df.head(3))
            
            if st.button("Run Statistical Test"):
                # Map columns using LLM
                headers = list(df.columns)
                data_snippet = df.head(3).to_markdown()

                sys_prompt = """
                You are a Data Scientist Helper.
                Analyze the dataset preview and identify the column names.

                OUTPUT JSON ONLY:
                {
                  "variant_col": "exact_name_of_group_column",
                  "metric_col": "exact_name_of_outcome_column",
                  "metric_type": "binary" (if 0/1 or True/False) or "continuous" (if revenue/time)
                }
                """

                response = ask_agent(sys_prompt, f"Headers: {headers}\nPreview: {data_snippet}", json_mode=True)

                if response:
                    try:
                        config = json.loads(response)
                        variant_col = config['variant_col']
                        metric_col = config['metric_col']
                        metric_type = config['metric_type']

                        st.success(f"📍 **Mapped:** Variant=`{variant_col}`, Metric=`{metric_col}` ({metric_type})")

                        # Run statistical test
                        groups = df[variant_col].unique()
                        
                        if len(groups) != 2:
                            st.error(f"Currently only supports 2 variants. Found: {groups}")
                        else:
                            group_a = df[df[variant_col] == groups[0]][metric_col]
                            group_b = df[df[variant_col] == groups[1]][metric_col]

                            # Sample Ratio Mismatch check
                            n_a = len(group_a)
                            n_b = len(group_b)
                            ratio = n_a / (n_a + n_b)

                            if abs(ratio - 0.5) > 0.05:
                                st.warning(f"Sample Ratio Mismatch: {ratio:.1%} vs expected 50%. Check your randomization.")

                            # Calculate Lift
                            mean_a = group_a.mean()
                            mean_b = group_b.mean()
                            lift = (mean_b - mean_a) / mean_a

                            st.metric(f"Lift ({groups[1]} vs {groups[0]})", f"{lift:.2%}")

                            # Run Statistical Test
                            if metric_type == "binary":
                                # Chi-Squared test
                                successes_a = int(group_a.sum())
                                successes_b = int(group_b.sum())
                                failures_a = len(group_a) - successes_a
                                failures_b = len(group_b) - successes_b

                                contingency_table = [
                                    [successes_a, failures_a],
                                    [successes_b, failures_b]
                                ]
                                stat, p_val, _, _ = chi2_contingency(contingency_table)
                                test_name = "Chi-Squared Test"

                                # Effect size (Cramér's V)
                                n = n_a + n_b
                                cramers_v = np.sqrt(stat / n)
                                effect_size_label = "Cramér's V"
                                effect_size = cramers_v

                                # Confidence intervals for conversion rates
                                cr_a = mean_a
                                cr_b = mean_b
                                se_a = np.sqrt(cr_a * (1 - cr_a) / n_a)
                                se_b = np.sqrt(cr_b * (1 - cr_b) / n_b)
                                se_diff = np.sqrt(se_a**2 + se_b**2)

                                margin = Z_ALPHA * se_diff
                                ci_lower = ((cr_b - margin) - cr_a) / cr_a
                                ci_upper = ((cr_b + margin) - cr_a) / cr_a

                            else:
                                # Welch's T-Test
                                stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
                                test_name = "Welch's T-Test"

                                # Effect size (Cohen's d)
                                pooled_std = np.sqrt(((n_a - 1) * group_a.std()**2 +
                                                     (n_b - 1) * group_b.std()**2) /
                                                    (n_a + n_b - 2))
                                cohens_d = (mean_b - mean_a) / pooled_std
                                effect_size_label = "Cohen's d"
                                effect_size = cohens_d

                                # Confidence intervals for means
                                se_a = group_a.std() / np.sqrt(n_a)
                                se_b = group_b.std() / np.sqrt(n_b)
                                se_diff = np.sqrt(se_a**2 + se_b**2)

                                margin = Z_ALPHA * se_diff
                                ci_lower = ((mean_b - margin) - mean_a) / mean_a
                                ci_upper = ((mean_b + margin) - mean_a) / mean_a

                            # Frequentist results
                            if csv_analysis_method in ["Frequentist (P-values)", "Both"]:
                                if csv_analysis_method == "Both":
                                    st.markdown("### Frequentist Analysis")

                                st.caption(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
                                st.caption(f"Effect Size ({effect_size_label}): {effect_size:.3f} | P-value: {p_val:.4f}")

                                # Declare winner based on actual means
                                if p_val < ALPHA:
                                    winner = groups[1] if mean_b > mean_a else groups[0]
                                    st.success(f"Winner: {winner} is statistically significant.")
                                else:
                                    st.warning("Result is not statistically significant.")

                                st.caption(f"Method: {test_name}")

                            # Bayesian Analysis (for binary metrics only)
                            if csv_analysis_method in ["Bayesian (Probability)", "Both"] and metric_type == "binary":
                                if csv_analysis_method == "Both":
                                    st.divider()
                                st.markdown("### Bayesian Analysis")

                                # Beta-Binomial conjugate prior
                                alpha_prior, beta_prior = 1, 1

                                # Posterior distributions
                                successes_a = int(group_a.sum())
                                successes_b = int(group_b.sum())

                                alpha_a = alpha_prior + successes_a
                                beta_a = beta_prior + (n_a - successes_a)
                                alpha_b = alpha_prior + successes_b
                                beta_b = beta_prior + (n_b - successes_b)

                                # Sample from posteriors
                                samples_a = beta.rvs(alpha_a, beta_a, size=BAYESIAN_SAMPLES)
                                samples_b = beta.rvs(alpha_b, beta_b, size=BAYESIAN_SAMPLES)

                                # P(B > A)
                                prob_b_wins = (samples_b > samples_a).mean()

                                # Expected loss if you ship variant B
                                loss_if_ship_b = np.maximum(samples_a - samples_b, 0).mean()

                                col1, col2 = st.columns(2)
                                col1.metric("P(Variant Beats Control)", f"{prob_b_wins:.1%}")
                                col2.metric("Expected Loss (if ship)", f"{loss_if_ship_b:.3%}")

                                st.caption(f"Posterior: {groups[0]} ~ Beta({alpha_a}, {beta_a}), {groups[1]} ~ Beta({alpha_b}, {beta_b})")

                                # Decision recommendation
                                if prob_b_wins > 0.95:
                                    st.success(f"High confidence: Ship {groups[1]}")
                                elif prob_b_wins > 0.75:
                                    st.info(f"Moderate confidence: Consider shipping {groups[1]}")
                                elif prob_b_wins < 0.25:
                                    st.error(f"High confidence: Keep {groups[0]}")
                                else:
                                    st.warning("Uncertain: Need more data")

                            elif csv_analysis_method in ["Bayesian (Probability)", "Both"] and metric_type == "continuous":
                                st.info("Bayesian analysis is currently only available for binary metrics.")

                    except Exception as e:
                        st.error(f"Mapping Failed. Error: {str(e)}")

    elif source_type == "SQL Query (Code Gen)":
        st.info("Paste your SQL to generate a Notebook snippet that runs the query and prints statistical results.")
        target_dwh = st.selectbox("DB Dialect:", ["BigQuery", "Snowflake", "Redshift"], key="an_dwh")
        sql_input = st.text_area("Your SQL:", "SELECT variant_id, user_id, revenue FROM logs")
        
        if st.button("Generate Analysis Notebook"):
            sys_prompt = f"""
            You are an Analytics Engineer. Write a Python Notebook snippet.
            1. Connect to {target_dwh} and run the user's SQL.
            2. Detect if the metric is Conversion (rate) or Revenue (mean).
            3. Run the appropriate stats test (Proportions Z-test or T-test).
            4. IMPORTANT: Add a final print statement that explicitly says: "Winner: [Variant] with p-value: [x]"
            """
            result = ask_agent(sys_prompt, f"SQL: {sql_input}")
            if result:
                st.code(result, language="python")