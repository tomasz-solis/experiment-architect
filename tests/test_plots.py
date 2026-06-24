"""Unit tests for Plotly figure helpers."""

from __future__ import annotations

import builtins
from typing import Any

import pandas as pd
import pytest

from stats.plots import (
    _smooth_series,
    plot_posterior_distributions,
    plot_power_curve,
    plot_rdd_discontinuity,
)


def test_plot_posterior_distributions_uses_supplied_labels() -> None:
    """Posterior plots should render one density trace per group."""
    figure = plot_posterior_distributions(12, 88, 15, 85, labels=("A", "B"))

    assert len(figure.data) == 2
    assert [trace.name for trace in figure.data] == ["A", "B"]
    assert figure.layout.xaxis.title.text == "Conversion rate"


def test_plot_power_curve_skips_invalid_candidate_mdes() -> None:
    """Power curves should tolerate ranges that include invalid non-positive MDEs."""
    figure = plot_power_curve(baseline=0.10, daily_traffic=5000, mde_range=(-0.05, 0.10))

    assert len(figure.data) == 1
    assert all(x_value > 0 for x_value in figure.data[0].x)
    assert len(figure.layout.shapes) == 2
    assert figure.layout.yaxis.title.text == "Days required"


def test_plot_rdd_discontinuity_adds_markers_trends_and_cutoff() -> None:
    """RDD plots should include both groups, trend lines, and the cutoff marker."""
    frame = pd.DataFrame(
        {
            "score": [40, 42, 44, 46, 48, 50, 52, 54, 56, 58],
            "treated": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "outcome": [80, 84, 88, 92, 96, 103, 107, 111, 115, 119],
        }
    )

    figure = plot_rdd_discontinuity(frame, "score", "outcome", cutoff=50.0, treatment_col="treated")

    assert [trace.name for trace in figure.data] == [
        "Control",
        "Treated",
        "Control trend",
        "Treated trend",
    ]
    assert len(figure.layout.shapes) == 1


def test_smooth_series_returns_empty_arrays_for_empty_data() -> None:
    """Empty inputs should not produce trend traces."""
    smooth_x, smooth_y = _smooth_series(pd.DataFrame({"x": [], "y": []}), "x", "y")

    assert smooth_x.size == 0
    assert smooth_y.size == 0


def test_smooth_series_falls_back_to_rolling_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If statsmodels is unavailable, smoothing should still return aligned arrays."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "statsmodels.nonparametric.smoothers_lowess":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

    smooth_x, smooth_y = _smooth_series(frame, "x", "y")

    assert smooth_x.tolist() == [1, 2, 3, 4, 5]
    assert smooth_y.tolist() == [2, 3, 4, 5, 6]
