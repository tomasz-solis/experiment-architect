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


class TestValidateMappingColumns:
    """Coverage for the LLM mapping contract."""

    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame({"group": ["A", "B"], "metric": [1, 0]})
        with pytest.raises(ValueError, match="not found"):
            validate_mapping_columns(
                {"variant_col": "group", "metric_col": "missing_metric"},
                df,
                ["variant_col", "metric_col"],
            )

    def test_reports_missing_required_keys(self) -> None:
        """Missing required keys raise before any column-existence check runs."""
        df = pd.DataFrame({"variant": ["A", "B"]})
        with pytest.raises(ValueError, match="missing required fields"):
            validate_mapping_columns(
                {"variant_col": "variant"},
                df,
                ["variant_col", "metric_col"],
            )

    def test_rejects_blank_string_values(self) -> None:
        """Blank string mappings are not valid even though the key exists."""
        df = pd.DataFrame({"variant": ["A", "B"], "metric": [1, 0]})
        with pytest.raises(ValueError, match="blank or not strings"):
            validate_mapping_columns(
                {"variant_col": "variant", "metric_col": "  "},
                df,
                ["variant_col", "metric_col"],
            )

    def test_rejects_non_string_values(self) -> None:
        """Non-string mappings get caught with a clear error."""
        df = pd.DataFrame({"variant": ["A", "B"], "metric": [1, 0]})
        with pytest.raises(ValueError, match="blank or not strings"):
            validate_mapping_columns(
                {"variant_col": "variant", "metric_col": 42},
                df,
                ["variant_col", "metric_col"],
            )

    def test_accumulates_multiple_invalid_fields(self) -> None:
        """When several fields are invalid, the error names all of them."""
        df = pd.DataFrame({"variant": ["A", "B"], "metric": [1, 0]})
        with pytest.raises(ValueError) as exc_info:
            validate_mapping_columns(
                {"variant_col": "", "metric_col": None},
                df,
                ["variant_col", "metric_col"],
            )
        assert "variant_col" in str(exc_info.value)
        assert "metric_col" in str(exc_info.value)

    def test_strips_whitespace_from_valid_names(self) -> None:
        """Mapped names are stripped of surrounding whitespace before lookup."""
        df = pd.DataFrame({"variant": ["A", "B"], "metric": [1, 0]})
        result = validate_mapping_columns(
            {"variant_col": " variant ", "metric_col": "metric"},
            df,
            ["variant_col", "metric_col"],
        )
        assert result == {"variant_col": "variant", "metric_col": "metric"}


class TestPrepareAbTestFrame:
    """Coverage for two-group experiment frame preparation."""

    def test_drops_missing_rows_and_coerces_binary(self) -> None:
        df = pd.DataFrame(
            {
                "variant": ["A", "A", "B", "B"],
                "converted": ["1", None, "0", "1"],
            }
        )
        cleaned, dropped_rows = prepare_ab_test_frame(df, "variant", "converted", "binary")
        assert dropped_rows == 1
        assert cleaned["converted"].tolist() == [1, 0, 1]

    def test_raises_when_all_rows_drop(self) -> None:
        """If every row has a missing required column, the helper raises."""
        df = pd.DataFrame({"variant": [None, None], "metric": [None, None]})
        with pytest.raises(ValueError, match="No rows remain"):
            prepare_ab_test_frame(df, "variant", "metric", "binary")

    def test_rejects_three_groups_and_names_them(self) -> None:
        """Three-group experiments are not supported and the error names what was found."""
        df = pd.DataFrame(
            {
                "variant": ["A", "B", "C"],
                "converted": [1, 0, 1],
            }
        )
        with pytest.raises(ValueError) as exc_info:
            prepare_ab_test_frame(df, "variant", "converted", "binary")
        error_message = str(exc_info.value)
        assert "2-group" in error_message
        assert "A" in error_message
        assert "B" in error_message
        assert "C" in error_message

    def test_rejects_binary_with_non_01_values(self) -> None:
        """Binary metric must contain only 0/1 after coercion."""
        df = pd.DataFrame(
            {
                "variant": ["A", "B"],
                "converted": [1, 2],
            }
        )
        with pytest.raises(ValueError, match="0/1"):
            prepare_ab_test_frame(df, "variant", "converted", "binary")

    def test_coerces_continuous_to_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "variant": ["A", "B"],
                "revenue": ["12.5", "13.7"],
            }
        )
        cleaned, _ = prepare_ab_test_frame(df, "variant", "revenue", "continuous")
        assert cleaned["revenue"].dtype.kind in {"f", "i"}


class TestPrepareDidFrame:
    """Coverage for panel data preparation."""

    def test_rejects_non_binary_treatment(self) -> None:
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

    def test_rejects_single_unit(self) -> None:
        """DiD needs at least two units to identify the treatment effect."""
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 1],
                "period": [0, 1, 2, 3],
                "treated": [1, 1, 1, 1],
                "outcome": [10.0, 11.0, 12.0, 15.0],
            }
        )
        with pytest.raises(ValueError, match="at least two units"):
            prepare_did_frame(df, "unit", "period", "treated", "outcome")

    def test_rejects_single_period_and_names_periods(self) -> None:
        """The single-period error message must surface what was found."""
        df = pd.DataFrame(
            {
                "unit": [1, 2, 3, 4],
                "period": ["Q1", "Q1", "Q1", "Q1"],
                "treated": [0, 0, 1, 1],
                "outcome": [10.0, 11.0, 12.0, 15.0],
            }
        )
        with pytest.raises(ValueError) as exc_info:
            prepare_did_frame(df, "unit", "period", "treated", "outcome")
        assert "Q1" in str(exc_info.value)


class TestPrepareRddFrame:
    """Coverage for RDD frame preparation."""

    def test_requires_numeric_running_variable(self) -> None:
        df = pd.DataFrame(
            {
                "score": ["low", "high"],
                "treated": [0, 1],
                "outcome": [10, 12],
            }
        )
        with pytest.raises(ValueError, match="numeric"):
            prepare_rdd_frame(df, "score", "treated", "outcome")

    def test_rejects_constant_running_variable(self) -> None:
        """RDD needs variation in the running variable on both sides of the cutoff."""
        df = pd.DataFrame(
            {
                "score": [50.0, 50.0, 50.0],
                "treated": [0, 0, 1],
                "outcome": [10.0, 11.0, 12.0],
            }
        )
        with pytest.raises(ValueError, match="variation"):
            prepare_rdd_frame(df, "score", "treated", "outcome")


class TestNormalizeMetricType:
    def test_is_case_insensitive(self) -> None:
        assert normalize_metric_type("BiNaRy") == "binary"

    def test_strips_whitespace(self) -> None:
        assert normalize_metric_type("  continuous  ") == "continuous"

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            normalize_metric_type("ordinal")
