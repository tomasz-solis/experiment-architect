"""Tests for analysis input validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from stats.validation import (
    normalize_metric_type,
    prepare_ab_test_frame,
    prepare_did_frame,
    prepare_rdd_frame,
    validate_mapping_columns,
)


def test_validate_mapping_columns_rejects_missing_columns() -> None:
    """Mapped column names must exist in the dataframe."""
    df = pd.DataFrame({"group": ["A", "B"], "metric": [1, 0]})

    with pytest.raises(ValueError, match="not found"):
        validate_mapping_columns(
            {"variant_col": "group", "metric_col": "missing_metric"},
            df,
            ["variant_col", "metric_col"],
        )


def test_prepare_ab_test_frame_drops_missing_rows_and_coerces_binary() -> None:
    """Binary experiment input should drop incomplete rows and coerce 0/1 values."""
    df = pd.DataFrame(
        {
            "variant": ["A", "A", "B", "B"],
            "converted": ["1", None, "0", "1"],
        }
    )

    cleaned, dropped_rows = prepare_ab_test_frame(df, "variant", "converted", "binary")

    assert dropped_rows == 1
    assert cleaned["converted"].tolist() == [1, 0, 1]


def test_prepare_did_frame_rejects_non_binary_treatment() -> None:
    """DiD input must have a binary treatment column."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [0, 2, 0, 2],
            "outcome": [10.0, 11.0, 9.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="0/1"):
        prepare_did_frame(df, "unit", "period", "treated", "outcome")


def test_prepare_rdd_frame_requires_numeric_running_variable() -> None:
    """RDD input must have a numeric running variable."""
    df = pd.DataFrame(
        {
            "score": ["low", "high"],
            "treated": [0, 1],
            "outcome": [10, 12],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        prepare_rdd_frame(df, "score", "treated", "outcome")


def test_normalize_metric_type_is_case_insensitive() -> None:
    """Metric type normalization should accept mixed case text."""
    assert normalize_metric_type("BiNaRy") == "binary"
