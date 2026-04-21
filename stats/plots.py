"""Plot helpers that keep visualization logic out of the Streamlit app."""

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import beta as beta_dist

from stats.frequentist import calculate_sample_size


def plot_posterior_distributions(
    alpha_a: float,
    beta_a: float,
    alpha_b: float,
    beta_b: float,
    labels: Tuple[str, str] = ("Control", "Variant"),
) -> go.Figure:
    """Plot posterior beta distributions for two experiment groups."""
    x_values = np.linspace(0.0001, 0.9999, 500)
    density_a = beta_dist.pdf(x_values, alpha_a, beta_a)
    density_b = beta_dist.pdf(x_values, alpha_b, beta_b)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=density_a,
            mode="lines",
            name=labels[0],
            line=dict(color="#1f77b4", width=3),
            fill="tozeroy",
            opacity=0.45,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=density_b,
            mode="lines",
            name=labels[1],
            line=dict(color="#ff7f0e", width=3),
            fill="tozeroy",
            opacity=0.45,
        )
    )
    figure.update_layout(
        title="Posterior Belief: What the data says about each group",
        xaxis_title="Conversion rate",
        yaxis_title="Probability density",
        template="plotly_white",
        legend_title="Group",
    )
    return figure


def plot_power_curve(
    baseline: float,
    daily_traffic: int,
    mde_range: Tuple[float, float] = (0.02, 0.30),
) -> go.Figure:
    """Plot the trade-off between detectable lift and experiment duration."""
    lower_mde, upper_mde = mde_range
    candidate_mdes = np.linspace(lower_mde, upper_mde, 50)

    x_values = []
    y_values = []
    for mde in candidate_mdes:
        try:
            result = calculate_sample_size(baseline, float(mde), daily_traffic)
        except ValueError:
            continue
        x_values.append(float(mde * 100))
        y_values.append(result["days"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="Days required",
            line=dict(color="#2ca02c", width=3),
        )
    )
    for label, days, dash in (("2 weeks", 14, "dash"), ("4 weeks", 28, "dot")):
        figure.add_hline(y=days, line_dash=dash, line_color="#7f7f7f", annotation_text=label)

    figure.update_layout(
        title="How long do you need to wait?",
        xaxis_title="Minimum detectable effect (%)",
        yaxis_title="Days required",
        template="plotly_white",
    )
    return figure


def plot_rdd_discontinuity(
    df: pd.DataFrame,
    running_var: str,
    outcome_col: str,
    cutoff: float,
    treatment_col: str,
) -> go.Figure:
    """Plot an RDD scatter with separate smooth lines on each side of the cutoff."""
    control = df[df[treatment_col] == 0].sort_values(running_var)
    treated = df[df[treatment_col] == 1].sort_values(running_var)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=control[running_var],
            y=control[outcome_col],
            mode="markers",
            name="Control",
            marker=dict(color="#1f77b4", size=7, opacity=0.45),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=treated[running_var],
            y=treated[outcome_col],
            mode="markers",
            name="Treated",
            marker=dict(color="#ff7f0e", size=7, opacity=0.45),
        )
    )

    for subset, label, color in (
        (control, "Control trend", "#1f77b4"),
        (treated, "Treated trend", "#ff7f0e"),
    ):
        smooth_x, smooth_y = _smooth_series(subset, running_var, outcome_col)
        if len(smooth_x) == 0:
            continue
        figure.add_trace(
            go.Scatter(
                x=smooth_x,
                y=smooth_y,
                mode="lines",
                name=label,
                line=dict(color=color, width=3),
            )
        )

    figure.add_vline(
        x=cutoff,
        line_width=2,
        line_dash="dash",
        line_color="#111111",
        annotation_text=f"Cutoff = {cutoff}",
    )
    figure.update_layout(
        title="RDD: Outcome by running variable",
        xaxis_title=running_var,
        yaxis_title=outcome_col,
        template="plotly_white",
    )
    return figure


def _smooth_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth a scatter series with LOWESS when available."""
    if df.empty:
        return np.array([]), np.array([])

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        x_values = df[x_col].to_numpy()
        y_values = df[y_col].rolling(window=max(5, len(df) // 10), min_periods=1).mean()
        return x_values, y_values.to_numpy()

    smoothed = lowess(df[y_col], df[x_col], frac=0.3, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]
