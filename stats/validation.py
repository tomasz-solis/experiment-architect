"""Validation helpers for LLM-mapped analysis inputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypedDict

import pandas as pd

# Normalised forms that plausibly name the control/baseline arm. Row order in
# the source CSV is not a design decision, so it must never be what decides
# which arm is "control" for the lift direction, the confidence interval, and
# the sample-ratio-mismatch check.
_CONTROL_ALIASES = {"control", "c", "a", "0", "off", "baseline", "old", "existing"}

# Above this many rows per randomised unit, a CSV reads as one row per
# repeated observation rather than one row per unit. A little above 1.0
# tolerates a handful of accidental duplicate rows without firing on them.
CLUSTERED_ROWS_PER_UNIT_THRESHOLD = 1.05


def _normalise_group_label(value: object) -> str:
    """Normalise a group label for control-arm matching."""
    return str(value).strip().lower().replace("_", "").replace("-", "")


def resolve_control_group(values: Sequence[object]) -> tuple[str, str]:
    """Pick which of two group values is the control arm, deterministically.

    First-appearance order in an uploaded CSV is an accident of how the file
    was exported, not a statement of which arm is control. Deciding it that
    way silently flips the lift direction and the SRM check whenever the
    treatment rows happen to come first. This picks the control by name when
    the name says so (e.g. "control", "A", "baseline"), and otherwise falls
    back to alphabetical order so the choice is at least stated and stable.
    """
    distinct = sorted({str(value) for value in values})
    if len(distinct) != 2:
        raise ValueError(
            f"resolve_control_group needs exactly 2 distinct values, found {len(distinct)}: "
            f"{distinct}."
        )

    for candidate in distinct:
        if _normalise_group_label(candidate) in _CONTROL_ALIASES:
            variant = distinct[1] if candidate == distinct[0] else distinct[0]
            return candidate, variant

    control, variant = distinct[0], distinct[1]
    return control, variant


_VARIANT_NAME_CANDIDATES = (
    "variant",
    "group",
    "arm",
    "bucket",
    "treatment",
    "test_group",
    "cohort",
)
_METRIC_NAME_CANDIDATES = (
    "converted",
    "conversion",
    "outcome",
    "revenue",
    "value",
    "metric",
    "target",
)


def _normalise_column_name(name: object) -> str:
    """Normalise a column name for case-insensitive candidate matching."""
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def guess_mapping(columns: Sequence[str]) -> dict[str, str]:
    """Best-guess the variant and metric columns from column names alone.

    This lets the mapping widgets default to something sensible without a
    model call, so the bundled sample file (and similarly named CSVs) maps
    itself in one click. Matching is case-insensitive on a normalised name.
    Returns only the keys it is confident about, and a column already picked
    for one role is never also offered for the other.
    """
    normalised_variant_names = {_normalise_column_name(c) for c in _VARIANT_NAME_CANDIDATES}
    normalised_metric_names = {_normalise_column_name(c) for c in _METRIC_NAME_CANDIDATES}

    guess: dict[str, str] = {}
    for column in columns:
        normalised = _normalise_column_name(column)
        if "variant_col" not in guess and normalised in normalised_variant_names:
            guess["variant_col"] = column
        elif "metric_col" not in guess and normalised in normalised_metric_names:
            guess["metric_col"] = column

    return guess


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
            "The model response was missing required fields: "
            + ", ".join(missing_keys)
            + "."
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
            "These mapped fields were blank or not strings: "
            + ", ".join(invalid_fields)
            + "."
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
    unique_values = sorted(numeric.unique().tolist())
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"{label} must contain only 0/1 values. Found: {unique_values}."
        )
    return numeric.astype(int)


def _drop_missing_rows(
    df: pd.DataFrame,
    required_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """Drop rows missing any required analysis field and report what was dropped."""
    cleaned = df.dropna(subset=required_columns).copy()
    dropped_rows = len(df) - len(cleaned)
    if cleaned.empty:
        raise ValueError(
            f"No rows remain after removing missing values in required columns: "
            f"{required_columns}. Check that the source dataset has values for these fields."
        )
    return cleaned, dropped_rows


def dropped_rows_by_arm(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    variant_col: str,
) -> dict[str, int]:
    """Count rows removed by cleaning, broken out by arm.

    Cleaning that drops one arm's rows disproportionately is invisible if you
    only look at the total drop count: it can manufacture or mask a sample
    ratio mismatch depending on which arm loses more. Rows with a missing or
    unmapped variant value are excluded from both counts (``value_counts``
    drops ``NaN`` by default), since they were never attributable to an arm in
    the first place.
    """
    original_counts = original[variant_col].value_counts()
    cleaned_counts = cleaned[variant_col].value_counts()
    return {
        str(arm): int(original_counts[arm] - cleaned_counts.get(arm, 0))
        for arm in original_counts.index
    }


class AnalysisUnitResult(TypedDict):
    """How many rows a dataset holds per randomised unit."""

    rows: int
    units: int
    rows_per_unit: float
    is_clustered: bool


def check_analysis_unit(df: pd.DataFrame, unit_col: str) -> AnalysisUnitResult:
    """Check whether the analysis is running on one row per randomised unit.

    Welch's t-test and the chi-squared test both assume every row is an
    independent observation. If randomisation actually happened per user but
    the file holds one row per session, per order, or per event, those rows
    are repeated measurements from the same person, not independent draws.
    Treating them as independent understates the standard error and overstates
    significance, silently, because nothing about the numbers looks wrong.
    This does not fix that by aggregating; it only measures whether the
    problem is there, so the analyst can decide.
    """
    if unit_col not in df.columns:
        raise ValueError(f"Column '{unit_col}' was not found in the dataset.")

    rows = len(df)
    if rows == 0:
        raise ValueError("The dataset has no rows to check.")

    units = int(df[unit_col].nunique(dropna=True))
    if units == 0:
        raise ValueError(f"Column '{unit_col}' has no non-missing values.")

    rows_per_unit = rows / units
    return {
        "rows": rows,
        "units": units,
        "rows_per_unit": rows_per_unit,
        "is_clustered": rows_per_unit > CLUSTERED_ROWS_PER_UNIT_THRESHOLD,
    }


def prepare_ab_test_frame(
    df: pd.DataFrame,
    variant_col: str,
    metric_col: str,
    metric_type: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a two-group experiment dataframe for analysis."""
    cleaned, dropped_rows = _drop_missing_rows(df, [variant_col, metric_col])
    n_groups = cleaned[variant_col].nunique()
    if n_groups != 2:
        found_groups = sorted(cleaned[variant_col].unique().tolist())
        raise ValueError(
            f"Only 2-group experiments are supported. Found {n_groups} groups: "
            f"{found_groups}. Filter the dataset to two groups before uploading."
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
    cleaned, dropped_rows = _drop_missing_rows(
        df, [unit_col, time_col, treatment_col, outcome_col]
    )
    cleaned[treatment_col] = _coerce_binary(cleaned[treatment_col], treatment_col)
    cleaned[outcome_col] = _coerce_numeric(cleaned[outcome_col], outcome_col)

    n_units = cleaned[unit_col].nunique()
    if n_units < 2:
        raise ValueError(
            f"DiD requires at least two units. Column '{unit_col}' has "
            f"{n_units} unique value(s)."
        )

    n_periods = cleaned[time_col].nunique()
    if n_periods < 2:
        found_periods = sorted(cleaned[time_col].unique().tolist())
        raise ValueError(
            f"DiD requires at least two time periods. Column '{time_col}' "
            f"has only: {found_periods}."
        )

    return cleaned, dropped_rows


def prepare_rdd_frame(
    df: pd.DataFrame,
    running_var: str,
    treatment_col: str,
    outcome_col: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a dataframe for regression discontinuity analysis."""
    cleaned, dropped_rows = _drop_missing_rows(
        df, [running_var, treatment_col, outcome_col]
    )
    cleaned[running_var] = _coerce_numeric(cleaned[running_var], running_var)
    cleaned[treatment_col] = _coerce_binary(cleaned[treatment_col], treatment_col)
    cleaned[outcome_col] = _coerce_numeric(cleaned[outcome_col], outcome_col)

    if cleaned[running_var].nunique() < 2:
        raise ValueError(
            f"RDD requires variation in the running variable '{running_var}'. "
            f"Found only one unique value."
        )

    return cleaned, dropped_rows
