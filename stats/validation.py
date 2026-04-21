"""Validation helpers for LLM-mapped analysis inputs."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def validate_mapping_columns(
    mapping: dict[str, Any],
    df: pd.DataFrame,
    column_keys: Iterable[str],
) -> dict[str, str]:
    """Return a cleaned mapping after checking that all mapped columns exist."""
    validated: dict[str, str] = {}
    missing_keys = [key for key in column_keys if key not in mapping]
    if missing_keys:
        raise ValueError(
            "The model response was missing required fields: " + ", ".join(missing_keys) + "."
        )

    invalid_fields: list[str] = []
    missing_columns: list[str] = []

    for key in column_keys:
        value = mapping[key]
        if not isinstance(value, str) or not value.strip():
            invalid_fields.append(key)
            continue
        column_name = value.strip()
        if column_name not in df.columns:
            missing_columns.append(f"{key} -> {column_name}")
            continue
        validated[key] = column_name

    if invalid_fields:
        raise ValueError(
            "These mapped fields were blank or not strings: " + ", ".join(invalid_fields) + "."
        )
    if missing_columns:
        raise ValueError(
            "These mapped columns were not found in the dataset: "
            + ", ".join(missing_columns)
            + "."
        )

    return validated


def normalize_metric_type(metric_type: Any) -> str:
    """Normalize and validate the metric type returned by the model."""
    normalized = str(metric_type).strip().lower()
    if normalized not in {"binary", "continuous"}:
        raise ValueError("metric_type must be either 'binary' or 'continuous'.")
    return normalized


def _coerce_numeric(series: pd.Series, label: str) -> pd.Series:
    """Coerce a series to numeric values or raise a clear error."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{label} must be numeric.")
    return numeric


def _coerce_binary(series: pd.Series, label: str) -> pd.Series:
    """Coerce a binary series to 0/1 integers or raise a clear error."""
    numeric = _coerce_numeric(series, label)
    unique_values = set(numeric.unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{label} must contain only 0/1 values.")
    return numeric.astype(int)


def _drop_missing_rows(
    df: pd.DataFrame,
    required_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """Drop rows that are missing any required analysis field."""
    cleaned = df.dropna(subset=required_columns).copy()
    dropped_rows = len(df) - len(cleaned)
    if cleaned.empty:
        raise ValueError("No rows remain after removing missing values in required columns.")
    return cleaned, dropped_rows


def prepare_ab_test_frame(
    df: pd.DataFrame,
    variant_col: str,
    metric_col: str,
    metric_type: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a two-group experiment dataframe for analysis."""
    cleaned, dropped_rows = _drop_missing_rows(df, [variant_col, metric_col])
    if cleaned[variant_col].nunique() != 2:
        raise ValueError(
            f"Only 2-group experiments are supported. Found {cleaned[variant_col].nunique()} groups."
        )

    normalized_metric_type = normalize_metric_type(metric_type)
    if normalized_metric_type == "binary":
        cleaned[metric_col] = _coerce_binary(cleaned[metric_col], metric_col)
    else:
        cleaned[metric_col] = _coerce_numeric(cleaned[metric_col], metric_col)

    return cleaned, dropped_rows


def prepare_did_frame(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a panel dataframe for Difference-in-Differences analysis."""
    cleaned, dropped_rows = _drop_missing_rows(df, [unit_col, time_col, treatment_col, outcome_col])
    cleaned[treatment_col] = _coerce_binary(cleaned[treatment_col], treatment_col)
    cleaned[outcome_col] = _coerce_numeric(cleaned[outcome_col], outcome_col)

    if cleaned[unit_col].nunique() < 2:
        raise ValueError("DiD requires at least two units.")
    if cleaned[time_col].nunique() < 2:
        raise ValueError("DiD requires at least two time periods.")

    return cleaned, dropped_rows


def prepare_rdd_frame(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a dataframe for regression discontinuity analysis."""
    cleaned, dropped_rows = _drop_missing_rows(df, [running_var, treatment_col, outcome_col])
    cleaned[running_var] = _coerce_numeric(cleaned[running_var], running_var)
    cleaned[treatment_col] = _coerce_binary(cleaned[treatment_col], treatment_col)
    cleaned[outcome_col] = _coerce_numeric(cleaned[outcome_col], outcome_col)

    if cleaned[running_var].nunique() < 2:
        raise ValueError("RDD requires variation in the running variable.")

    return cleaned, dropped_rows
