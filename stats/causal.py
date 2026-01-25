"""Causal inference methods (DiD, RDD)."""

from typing import Dict, Any
import pandas as pd


def difference_in_differences(
    df: pd.DataFrame,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    intervention_point: Any
) -> Dict[str, Any]:
    """Run Difference-in-Differences analysis.

    Args:
        df: DataFrame with panel data
        time_col: Time period column name
        treatment_col: Treatment indicator column name
        outcome_col: Outcome metric column name
        intervention_point: Intervention date/period

    Returns:
        dict: DiD results including coefficient, p-value, CI
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for DiD. Run: pip install statsmodels")

    df = df.copy()
    df['post'] = (df[time_col] >= intervention_point).astype(int)
    df['treated'] = df[treatment_col].astype(int)
    df['did'] = df['treated'] * df['post']

    X = df[['treated', 'post', 'did']]
    X = sm.add_constant(X)
    y = df[outcome_col]

    model = sm.OLS(y, X).fit()

    did_coef = model.params['did']
    did_pval = model.pvalues['did']
    did_ci = model.conf_int().loc['did']

    return {
        "coefficient": did_coef,
        "p_value": did_pval,
        "ci_lower": did_ci[0],
        "ci_upper": did_ci[1],
        "model": model
    }


def regression_discontinuity(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
    cutoff: float
) -> Dict[str, Any]:
    """Run Regression Discontinuity Design analysis.

    Args:
        df: DataFrame with observation data
        running_var: Running variable column name (e.g., credit score)
        treatment_col: Treatment indicator column name
        outcome_col: Outcome metric column name
        cutoff: Treatment assignment cutoff value

    Returns:
        dict: RDD results including coefficient, p-value, CI
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required for RDD. Run: pip install statsmodels")

    df = df.copy()
    df['centered'] = df[running_var] - cutoff
    df['treated'] = df[treatment_col].astype(int)
    df['interaction'] = df['treated'] * df['centered']

    X = df[['treated', 'centered', 'interaction']]
    X = sm.add_constant(X)
    y = df[outcome_col]

    model = sm.OLS(y, X).fit()

    rdd_coef = model.params['treated']
    rdd_pval = model.pvalues['treated']
    rdd_ci = model.conf_int().loc['treated']

    return {
        "coefficient": rdd_coef,
        "p_value": rdd_pval,
        "ci_lower": rdd_ci[0],
        "ci_upper": rdd_ci[1],
        "model": model
    }


def select_causal_method(has_cutoff: bool, has_clean_control: bool, is_opt_in: bool) -> str:
    """Expert system for selecting causal inference method.

    Args:
        has_cutoff: Whether there's a strict cutoff (e.g., credit score > 600)
        has_clean_control: Whether there's a clean control group
        is_opt_in: Whether users self-select into treatment

    Returns:
        str: Recommended causal method name
    """
    if has_cutoff:
        return "Regression Discontinuity (RDD)"
    elif has_clean_control and not is_opt_in:
        return "Difference-in-Differences (DiD)"
    elif is_opt_in:
        return "Propensity Score Matching (PSM)"
    else:
        return "CausalImpact"
