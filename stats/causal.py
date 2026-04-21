"""Causal inference methods: DiD, RDD, and method selection.

DiD implementation includes:
- Parallel-trends pre-test
- Cluster-robust standard errors by unit

RDD implementation includes:
- Density check around the cutoff
- Rule-of-thumb local bandwidth selection
- Bandwidth sweep diagnostics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def _build_post_indicator(time_values: pd.Series, intervention_point: Any) -> pd.Series:
    """Build a post-period indicator that works for numeric, datetime, or label periods."""
    if is_numeric_dtype(time_values):
        numeric_times = pd.to_numeric(time_values, errors="raise")
        cutoff = pd.to_numeric(pd.Series([intervention_point]), errors="coerce").iloc[0]
        if pd.isna(cutoff):
            raise ValueError("Intervention point must be numeric for a numeric time column.")
        return (numeric_times >= float(cutoff)).astype(int)

    if is_datetime64_any_dtype(time_values):
        cutoff = pd.to_datetime(intervention_point, errors="coerce")
        if pd.isna(cutoff):
            raise ValueError("Intervention point could not be parsed as a date.")
        return (time_values >= cutoff).astype(int)

    parsed_times = pd.to_datetime(time_values, errors="coerce")
    parsed_cutoff = pd.to_datetime(intervention_point, errors="coerce")
    if parsed_times.notna().all() and not pd.isna(parsed_cutoff):
        return (parsed_times >= parsed_cutoff).astype(int)

    ordered_values = pd.Index(pd.unique(time_values))
    if intervention_point not in ordered_values:
        raise ValueError(
            f"Intervention point '{intervention_point}' was not found in the time column."
        )

    value_to_order = {value: idx for idx, value in enumerate(ordered_values)}
    return time_values.map(value_to_order).ge(value_to_order[intervention_point]).astype(int)


def _encode_time_order(time_values: pd.Series) -> pd.Series:
    """Encode time values into a numeric sequence while preserving their order."""
    if is_numeric_dtype(time_values):
        return pd.to_numeric(time_values, errors="raise").astype(float)

    if is_datetime64_any_dtype(time_values):
        return time_values.astype("int64").astype(float)

    parsed_times = pd.to_datetime(time_values, errors="coerce")
    if parsed_times.notna().all():
        return parsed_times.astype("int64").astype(float)

    ordered_values = pd.Index(pd.unique(time_values))
    value_to_order = {value: float(idx) for idx, value in enumerate(ordered_values)}
    return time_values.map(value_to_order).astype(float)


def _fit_rdd_model(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
    cutoff: float,
) -> dict[str, Any]:
    """Fit a linear RDD model on a dataframe slice."""
    import statsmodels.api as sm

    if df.empty:
        raise ValueError("RDD requires at least one observation.")
    if df[treatment_col].nunique() < 2:
        raise ValueError("RDD requires observations on both sides of the cutoff.")

    working = df.copy()
    working["_centered"] = working[running_var] - cutoff
    working["_treated"] = working[treatment_col].astype(int)
    working["_interaction"] = working["_treated"] * working["_centered"]

    X = sm.add_constant(working[["_treated", "_centered", "_interaction"]])
    model = sm.OLS(working[outcome_col], X).fit()

    return {
        "coefficient": float(model.params["_treated"]),
        "p_value": float(model.pvalues["_treated"]),
        "ci_lower": float(model.conf_int().loc["_treated", 0]),
        "ci_upper": float(model.conf_int().loc["_treated", 1]),
        "model": model,
    }


def _minimum_bandwidth_for_side_counts(
    values: pd.Series,
    cutoff: float,
    min_side_observations: int,
) -> float:
    """Return the smallest symmetric window that keeps enough rows on both sides."""
    left_distances = np.sort((cutoff - values[values < cutoff]).to_numpy())
    right_distances = np.sort((values[values >= cutoff] - cutoff).to_numpy())

    if len(left_distances) == 0 or len(right_distances) == 0:
        return 0.0

    if len(left_distances) < min_side_observations or len(right_distances) < min_side_observations:
        return float(max(left_distances[-1], right_distances[-1]))

    return float(
        max(
            left_distances[min_side_observations - 1],
            right_distances[min_side_observations - 1],
        )
    )


def _select_rdd_bandwidth(
    df: pd.DataFrame,
    running_var: str,
    cutoff: float,
    min_side_observations: int = 30,
) -> float:
    """Choose a rule-of-thumb local bandwidth for RDD estimation.

    The bandwidth starts from a robust dispersion estimate and is widened when
    needed so both sides of the cutoff have enough observations for a stable
    local fit.
    """
    values = pd.to_numeric(df[running_var], errors="raise")
    distances = (values - cutoff).abs()

    std = float(values.std())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    robust_scale = min(std, iqr / 1.349) if iqr > 0 else std
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = std
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(distances.median())
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = 1.0

    rule_of_thumb = 1.84 * robust_scale * (len(values) ** (-1 / 5))
    side_count_floor = _minimum_bandwidth_for_side_counts(
        values=values,
        cutoff=cutoff,
        min_side_observations=min_side_observations,
    )
    max_distance = float(distances.max())

    bandwidth = max(rule_of_thumb, side_count_floor)
    if max_distance > 0:
        bandwidth = min(bandwidth, max_distance)

    return float(bandwidth)


def _fit_rdd_at_bandwidth(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
    cutoff: float,
    bandwidth: float,
) -> dict[str, Any] | None:
    """Fit RDD within a symmetric bandwidth window when both sides are present."""
    window_df = df[(df[running_var] - cutoff).abs() <= bandwidth].copy()
    if window_df.empty or window_df[treatment_col].nunique() < 2:
        return None

    result = _fit_rdd_model(
        df=window_df,
        running_var=running_var,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        cutoff=cutoff,
    )
    return {
        "label": "window",
        "bandwidth": float(bandwidth),
        "n_obs": int(len(window_df)),
        "coefficient": float(result["coefficient"]),
        "p_value": float(result["p_value"]),
        "result": result,
    }


def check_parallel_trends(
    df: pd.DataFrame,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    intervention_point: Any,
) -> dict[str, Any]:
    """Test the DiD parallel-trends assumption on the pre-period only."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels required for DiD. Run: pip install statsmodels") from exc

    working = df.copy()
    working["_post"] = _build_post_indicator(working[time_col], intervention_point)
    working["_treated"] = working[treatment_col].astype(int)
    pre = working[working["_post"] == 0].copy()

    if pre.empty or pre[time_col].nunique() < 2 or pre["_treated"].nunique() < 2:
        return {
            "passes": True,
            "trend_interaction_pvalue": None,
            "n_pre_periods": int(pre[time_col].nunique()) if not pre.empty else 0,
            "test_ran": False,
        }

    pre["_time_num"] = _encode_time_order(pre[time_col])
    pre["_pre_interact"] = pre["_treated"] * pre["_time_num"]
    X_pre = sm.add_constant(pre[["_treated", "_time_num", "_pre_interact"]])
    pre_model = sm.OLS(pre[outcome_col], X_pre).fit()
    p_value = float(pre_model.pvalues["_pre_interact"])

    return {
        "passes": p_value > 0.10,
        "trend_interaction_pvalue": p_value,
        "n_pre_periods": int(pre[time_col].nunique()),
        "test_ran": True,
    }


def difference_in_differences(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    intervention_point: Any,
) -> dict[str, Any]:
    """Run Difference-in-Differences with diagnostics and clustered standard errors."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels required for DiD. Run: pip install statsmodels") from exc

    working = df.copy()
    working["_post"] = _build_post_indicator(working[time_col], intervention_point)
    working["_treated"] = working[treatment_col].astype(int)
    working["_did"] = working["_treated"] * working["_post"]

    parallel_trends = check_parallel_trends(
        df=working,
        time_col=time_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        intervention_point=intervention_point,
    )

    X = sm.add_constant(working[["_treated", "_post", "_did"]])
    model = sm.OLS(working[outcome_col], X).fit(
        cov_type="cluster",
        cov_kwds={"groups": working[unit_col]},
    )
    did_ci = model.conf_int().loc["_did"]

    diagnostics = {
        "parallel_trends_pvalue": parallel_trends["trend_interaction_pvalue"],
        "parallel_trends_ok": parallel_trends["passes"],
        "parallel_trends_test_ran": parallel_trends["test_ran"],
        "cluster_robust": True,
        "n_units": int(working[unit_col].nunique()),
        "n_periods": int(working[time_col].nunique()),
        "n_pre_periods": parallel_trends["n_pre_periods"],
    }

    return {
        "coefficient": float(model.params["_did"]),
        "p_value": float(model.pvalues["_did"]),
        "ci_lower": float(did_ci[0]),
        "ci_upper": float(did_ci[1]),
        "diagnostics": diagnostics,
        "model": model,
    }


def regression_discontinuity(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
    cutoff: float,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    """Run RDD with density, local-bandwidth, and sensitivity diagnostics."""
    try:
        import statsmodels.api as sm  # noqa: F401
    except ImportError as exc:
        raise ImportError("statsmodels required for RDD. Run: pip install statsmodels") from exc

    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("Bandwidth must be greater than zero.")

    rv_std = float(df[running_var].std())
    effective_bandwidth = (
        float(bandwidth)
        if bandwidth is not None
        else _select_rdd_bandwidth(df=df, running_var=running_var, cutoff=cutoff)
    )
    bandwidth_method = "user_supplied" if bandwidth is not None else "rule_of_thumb"

    window = rv_std * 0.2
    n_below = int(((df[running_var] >= cutoff - window) & (df[running_var] < cutoff)).sum())
    n_above = int(((df[running_var] >= cutoff) & (df[running_var] < cutoff + window)).sum())
    density_ratio = n_above / n_below if n_below > 0 else float("inf")
    density_ok = 0.7 <= density_ratio <= 1.4

    full_result = _fit_rdd_model(df, running_var, treatment_col, outcome_col, cutoff)
    local_fit = _fit_rdd_at_bandwidth(
        df,
        running_var,
        treatment_col,
        outcome_col,
        cutoff,
        effective_bandwidth,
    )
    if local_fit is None:
        raise ValueError("RDD local bandwidth window does not contain both sides of the cutoff.")

    half_bandwidth = max(effective_bandwidth / 2, 1e-9)
    half_fit = _fit_rdd_at_bandwidth(
        df,
        running_var,
        treatment_col,
        outcome_col,
        cutoff,
        half_bandwidth,
    )

    bandwidth_sweep: list[dict[str, str | float | int]] = [
        {
            "label": "local",
            "bandwidth": round(float(effective_bandwidth), 4),
            "n_obs": int(local_fit["n_obs"]),
            "coefficient": round(float(local_fit["coefficient"]), 6),
            "p_value": round(float(local_fit["p_value"]), 6),
        }
    ]
    if half_fit is not None:
        bandwidth_sweep.append(
            {
                "label": "half_bandwidth",
                "bandwidth": round(float(half_bandwidth), 4),
                "n_obs": int(half_fit["n_obs"]),
                "coefficient": round(float(half_fit["coefficient"]), 6),
                "p_value": round(float(half_fit["p_value"]), 6),
            }
        )
    bandwidth_sweep.append(
        {
            "label": "full_sample",
            "bandwidth": round(float((df[running_var] - cutoff).abs().max()), 4),
            "n_obs": int(len(df)),
            "coefficient": round(float(full_result["coefficient"]), 6),
            "p_value": round(float(full_result["p_value"]), 6),
        }
    )

    reference_coef = float(local_fit["coefficient"])
    comparison_coefs = [float(full_result["coefficient"])]
    if half_fit is not None:
        comparison_coefs.append(float(half_fit["coefficient"]))
    max_shift = max(
        (
            abs(reference_coef - coef) / abs(reference_coef)
            if abs(reference_coef) > 1e-9
            else abs(reference_coef - coef)
        )
        for coef in comparison_coefs
    )

    diagnostics = {
        "density_ratio_at_cutoff": round(float(density_ratio), 3),
        "density_ok": density_ok,
        "n_below_cutoff": n_below,
        "n_above_cutoff": n_above,
        "bandwidth_method": bandwidth_method,
        "bandwidth_used": effective_bandwidth,
        "n_in_bandwidth": int(local_fit["n_obs"]),
        "half_bandwidth_used": float(half_bandwidth) if half_fit is not None else None,
        "half_band_coefficient": (
            round(float(half_fit["coefficient"]), 6) if half_fit is not None else None
        ),
        "full_sample_coefficient": round(float(full_result["coefficient"]), 6),
        "coefficient_shift_pct": round(float(max_shift) * 100, 1),
        "coefficient_stable_under_bandwidth": bool(max_shift < 0.30),
        "bandwidth_sweep": bandwidth_sweep,
    }

    return {
        "coefficient": float(local_fit["result"]["coefficient"]),
        "p_value": float(local_fit["result"]["p_value"]),
        "ci_lower": float(local_fit["result"]["ci_lower"]),
        "ci_upper": float(local_fit["result"]["ci_upper"]),
        "diagnostics": diagnostics,
        "model": local_fit["result"]["model"],
    }


def select_causal_method(
    has_cutoff: bool,
    has_clean_control: bool,
    is_opt_in: bool,
) -> str:
    """Select a quasi-experimental method based on study design properties."""
    if has_cutoff:
        return "Regression Discontinuity (RDD)"
    if has_clean_control and not is_opt_in:
        return "Difference-in-Differences (DiD)"
    if is_opt_in:
        return "Propensity Score Matching (PSM)"
    return "CausalImpact"
