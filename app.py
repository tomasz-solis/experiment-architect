import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from openai import OpenAI
from scipy.stats import ttest_ind, chi2_contingency
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# Statistical constants - don't touch these unless you know what you're doing
Z_ALPHA = 1.96  # 95% confidence level
Z_BETA = 0.84   # 80% power
ALPHA = 0.05    # Significance threshold

st.set_page_config(page_title="Experiment Architect", page_icon="🏗️", layout="centered")

# Initialize OpenAI Client with safe failover
# Priority: 1. Environment Variable (.env), 2. Streamlit Secrets (secrets.toml)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

try:
    if api_key:
        client = OpenAI(api_key=api_key)
        ai_enabled = True
    else:
        ai_enabled = False
        client = None
except Exception as e:
    ai_enabled = False
    client = None
    print(f"OpenAI init failed: {e}")

def ask_agent(system_role, user_prompt, json_mode=False):
    """Call the LLM. Set json_mode=True to force JSON output."""
    if not ai_enabled:
        st.warning("⚠️ AI features disabled - check your API key in .env")
        return None

    try:
        response_format = {"type": "json_object"} if json_mode else None

        with st.spinner("Processing..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format=response_format
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI Service Error: {str(e)}")
        return None

# ==========================================
# 2. APP HEADER
# ==========================================
st.title("🏗️ The Experiment Architect")
st.markdown("**Plan. Execute. Analyze.** The Decision Scientist's Companion.")

tab_design, tab_analyze = st.tabs(["📐 Design Experiment", "📊 Analyze Results"])

# ==============================================================================
# TAB 1: DESIGN EXPERIMENT (Planning)
# ==============================================================================
with tab_design:
    st.header("Step 1: Strategy & Sizing")
    
    experiment_type = st.radio(
        "Can you randomize users into groups?",
        ["Yes (A/B Test)", "No (Observational Study)"],
        horizontal=True
    )
    
    st.divider()

    # --------------------------------------------------
    # PATH A: A/B TEST PLANNER
    # --------------------------------------------------
    if experiment_type == "Yes (A/B Test)":
        st.subheader("🧪 A/B Test Calculator")
        
        # --- FEATURE: REVERSE MDE CALCULATOR ---
        with st.expander("🤔 Help! What is a realistic MDE?"):
            st.markdown("Don't guess. Input your max wait time, and I'll calculate the smallest effect size you can statistically detect.")
            
            c_wiz1, c_wiz2 = st.columns(2)
            wiz_weeks = c_wiz1.slider("Max Wait Time (Weeks)", 1, 12, 4)
            wiz_traffic = c_wiz2.number_input("Avg Daily Visitors", 100, 1000000, 5000)
            wiz_base = c_wiz2.number_input("Baseline Conv (%)", 0.1, 99.0, 10.0, key="wiz_base") / 100
            
            if st.button("Calculate Minimum Detectable Effect"):
                total_n = wiz_traffic * (wiz_weeks * 7)
                z_score = (Z_ALPHA + Z_BETA)**2
                pooled_var = 2 * wiz_base * (1 - wiz_base)

                # Sanity checks - catch edge cases before the math explodes
                if total_n < 100:
                    st.error("⚠️ Not enough traffic. You need at least 100 total visitors.")
                elif wiz_base < 0.001 or wiz_base > 0.999:
                    st.error("⚠️ Baseline conversion must be between 0.1% and 99.9%")
                else:
                    # Formula: MDE = sqrt( (Z^2 * Var) / N ) / Baseline
                    feasible_mde = np.sqrt((z_score * pooled_var) / total_n) / wiz_base

                    if feasible_mde > 10.0:
                        st.error(f"📉 Your detectable lift is **{feasible_mde:.1%}** - way too high. Increase traffic or wait longer.")
                    else:
                        st.info(f"📉 In {wiz_weeks} weeks, the smallest lift you can detect is **{feasible_mde:.1%}**.")
                        st.caption("If your feature impact is smaller than this, you will never reach significance.")

        st.divider()

        # --- STANDARD INPUTS ---
        c1, c2 = st.columns(2)
        with c1:
            baseline = st.number_input("Baseline Conversion (%)", 0.1, 99.0, 10.0, step=0.5, key="main_base") / 100
            mde = st.number_input("Target Lift (Relative %)", 1.0, 500.0, 10.0, step=1.0) / 100
        with c2:
            daily_traffic = st.number_input("Daily Visitors (Total)", 100, 1000000, 5000, step=100, key="main_traffic")
            split_ratio = st.slider("Traffic Allocation (Variant %)", 1, 99, 50) / 100

        # --- FEATURE: AI SANITY CHECK ---
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

        # --- DETERMINISTIC MATH ENGINE ---
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

    # --------------------------------------------------
    # PATH B: CAUSAL METHOD SELECTOR
    # --------------------------------------------------
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
            
        st.success(f"👉 **Recommended Method:** {method}")

        # --- AGENTIC CODE GENERATOR ---
        st.markdown("---")
        st.subheader("🤖 Implementation Assistant")
        
        c_in, c_dwh = st.columns(2)
        input_format = c_in.radio("Data Source:", ["CSV File", "SQL Query"])
        dwh_flavor = c_dwh.selectbox("Target Database:", ["BigQuery", "Snowflake", "Redshift"])
        
        user_context = st.text_area("Paste Column Names or SQL Schema:", height=100, 
                                    placeholder="e.g., user_id, transaction_date, treatment_flag, total_spend")
        
        if st.button("Generate Python Script"):
            prompt = f"""
            You are a Senior Data Scientist. Write a Python script for {method}.
            Target: {dwh_flavor}. Input Type: {input_format}.
            User Context: {user_context}.
            
            Requirements:
            1. Use standard libraries (CausalImpact, statsmodels, DoWhy).
            2. If SQL, include the specific connector code for {dwh_flavor}.
            3. Include comments explaining statistical assumptions.
            """
            result = ask_agent(prompt, "Write the code.")
            if result:
                st.code(result, language="python")

# ==============================================================================
# TAB 2: ANALYZE RESULTS (Analysis)
# ==============================================================================
with tab_analyze:
    st.header("Step 2: Analysis & Decision")
    
    source_type = st.radio("Input Method:", 
                           ["Summary Stats (Manual)", "Raw CSV (Automated Analysis)", "SQL Query (Code Gen)"], 
                           horizontal=True)
    
    st.divider()

    # --- OPTION A: MANUAL STATS ---
    if source_type == "Summary Stats (Manual)":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🅰️ Control")
            vis_a = st.number_input("Visitors A", 1000)
            conv_a = st.number_input("Conversions A", 100)
        with c2:
            st.markdown("### 🅱️ Variant")
            vis_b = st.number_input("Visitors B", 1000)
            conv_b = st.number_input("Conversions B", 115)
            
        if st.button("Calculate Result"):
            cr_a = conv_a / vis_a
            cr_b = conv_b / vis_b
            lift = (cr_b - cr_a) / cr_a
            st.metric("Relative Lift", f"{lift:.2%}")
            
            if lift > 0: st.success("Variant B is winning.")
            else: st.error("Variant B is losing.")

    # --- OPTION B: CSV ANALYZER (SMART MAPPER + PYTHON MATH) ---
    elif source_type == "Raw CSV (Automated Analysis)":
        st.info("Upload your results. I will map the columns, and Python will run the statistical test locally.")
        uploaded_file = st.file_uploader("Upload .csv", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head(3))
            
            if st.button("Run Statistical Test"):
                # STEP 1: AI MAPS THE COLUMNS (No Hallucinated Math)
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
                        
                        # STEP 2: PYTHON DOES THE MATH (On Full Dataset)
                        groups = df[variant_col].unique()
                        
                        if len(groups) != 2:
                            st.error(f"⚠️ Currently only supports 2 variants. Found: {groups}")
                        else:
                            group_a = df[df[variant_col] == groups[0]][metric_col]
                            group_b = df[df[variant_col] == groups[1]][metric_col]

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
                            else:
                                # Welch's T-Test
                                stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
                                test_name = "Welch's T-Test"

                            # Declare winner based on actual means
                            if p_val < ALPHA:
                                winner = groups[1] if mean_b > mean_a else groups[0]
                                st.success(f"🏆 **Winner:** {winner} is statistically significant!")
                            else:
                                st.warning("🤷 **Inconclusive:** Result is not statistically significant.")

                            st.caption(f"Method: {test_name} | P-Value: {p_val:.4f}")

                    except Exception as e:
                        st.error(f"Mapping Failed. Error: {str(e)}")

    # --- OPTION C: SQL NOTEBOOK GENERATOR ---
    elif source_type == "SQL Query (Code Gen)":
        st.info("Paste your SQL. I will write a Notebook snippet that runs it and prints the final verdict.")
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