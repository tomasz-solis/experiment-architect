"""Tests for analysis input validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from stats.validation import (
    check_analysis_unit,
    dropped_rows_by_arm,
    guess_mapping,
    normalize_metric_type,
    prepare_ab_test_frame,
    prepare_did_frame,
    prepare_rdd_frame,
    resolve_control_group,
    validate_mapping_columns,
)


class TestDroppedRowsByArm:
    """Coverage for per-arm attrition counts between raw and cleaned frames."""

    def test_counts_dropped_rows_per_arm(self) -> None:
        original = pd.DataFrame(
            {
                "variant": ["A", "A", "A", "B", "B", "B"],
                "converted": [1, None, 0, 1, None, None],
            }
        )
        cleaned, _ = prepare_ab_test_frame(original, "variant", "converted", "binary")
        counts = dropped_rows_by_arm(original, cleaned, "variant")
        assert counts == {"A": 1, "B": 2}

    def test_no_drops_reports_zero_for_every_arm(self) -> None:
        original = pd.DataFrame({"variant": ["A", "A", "B", "B"], "converted": [1, 0, 1, 0]})
        cleaned, _ = prepare_ab_test_frame(original, "variant", "converted", "binary")
        counts = dropped_rows_by_arm(original, cleaned, "variant")
        assert counts == {"A": 0, "B": 0}

    def test_rows_with_missing_variant_are_not_attributed_to_an_arm(self) -> None:
        original = pd.DataFrame(
            {
                "variant": ["A", "A", None, "B", "B"],
                "converted": [1, 0, 1, 1, 0],
            }
        )
        cleaned, _ = prepare_ab_test_frame(original, "variant", "converted", "binary")
        counts = dropped_rows_by_arm(original, cleaned, "variant")
        assert counts == {"A": 0, "B": 0}


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


class TestCheckAnalysisUnit:
    """One row per randomised unit, or the tests below overstate significance."""

    def test_one_row_per_unit_is_not_clustered(self) -> None:
        df = pd.DataFrame({"user_id": range(100)})
        result = check_analysis_unit(df, "user_id")
        assert result["rows"] == 100
        assert result["units"] == 100
        assert result["rows_per_unit"] == pytest.approx(1.0)
        assert not result["is_clustered"]

    def test_exactly_at_the_tolerance_threshold_is_not_clustered(self) -> None:
        """20 units, 21 rows: exactly 1.05 rows per unit, the tolerance boundary itself."""
        df = pd.DataFrame({"user_id": [*range(20), 0]})
        result = check_analysis_unit(df, "user_id")
        assert result["rows_per_unit"] == pytest.approx(1.05)
        assert not result["is_clustered"]

    def test_above_the_threshold_is_clustered(self) -> None:
        df = pd.DataFrame({"user_id": [*range(20), 0, 1]})
        result = check_analysis_unit(df, "user_id")
        assert result["rows_per_unit"] == pytest.approx(1.1)
        assert result["is_clustered"]

    def test_ignores_missing_unit_values_when_counting_units(self) -> None:
        df = pd.DataFrame({"user_id": [1, 1, 2, None]})
        result = check_analysis_unit(df, "user_id")
        assert result["units"] == 2

    def test_rejects_a_missing_column(self) -> None:
        df = pd.DataFrame({"user_id": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            check_analysis_unit(df, "missing_col")

    def test_rejects_an_empty_dataframe(self) -> None:
        df = pd.DataFrame({"user_id": pd.Series([], dtype="int64")})
        with pytest.raises(ValueError, match="no rows"):
            check_analysis_unit(df, "user_id")

    def test_rejects_a_column_with_only_missing_values(self) -> None:
        df = pd.DataFrame({"user_id": [None, None, None]})
        with pytest.raises(ValueError, match="no non-missing values"):
            check_analysis_unit(df, "user_id")


class TestResolveControlGroup:
    """Coverage for deterministic control/variant resolution, independent of row order."""

    def test_recognises_named_control_aliases(self) -> None:
        assert resolve_control_group(["treatment", "control"]) == ("control", "treatment")

    def test_recognises_single_letter_a_as_control(self) -> None:
        assert resolve_control_group(["B", "A"]) == ("A", "B")

    def test_recognises_zero_as_control(self) -> None:
        assert resolve_control_group(["1", "0"]) == ("0", "1")

    def test_recognises_off_as_control(self) -> None:
        assert resolve_control_group(["on", "off"]) == ("off", "on")

    def test_recognises_baseline_as_control(self) -> None:
        assert resolve_control_group(["new_flow", "baseline"]) == ("baseline", "new_flow")

    def test_recognises_old_as_control(self) -> None:
        assert resolve_control_group(["new", "old"]) == ("old", "new")

    def test_recognises_existing_as_control(self) -> None:
        assert resolve_control_group(["redesign", "existing"]) == ("existing", "redesign")

    def test_alias_matching_ignores_case_and_separators(self) -> None:
        """"Control-Group" style labels normalise the same as "control"."""
        assert resolve_control_group(["Variant_B", "Control-Group"]) == (
            "Control-Group",
            "Variant_B",
        )

    def test_falls_back_to_sorted_order_when_no_alias_matches(self) -> None:
        assert resolve_control_group(["blue", "green"]) == ("blue", "green")
        assert resolve_control_group(["green", "blue"]) == ("blue", "green")

    def test_row_order_never_decides_control_when_no_alias_matches(self) -> None:
        """Whichever value appears first in the file must not change the outcome."""
        assert resolve_control_group(["zebra", "apple"]) == resolve_control_group(
            ["apple", "zebra"]
        )

    def test_rejects_more_than_two_distinct_values(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 distinct values"):
            resolve_control_group(["A", "B", "C"])

    def test_rejects_a_single_distinct_value(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 distinct values"):
            resolve_control_group(["A", "A"])


class TestNormalizeMetricType:
    def test_is_case_insensitive(self) -> None:
        assert normalize_metric_type("BiNaRy") == "binary"

    def test_strips_whitespace(self) -> None:
        assert normalize_metric_type("  continuous  ") == "continuous"

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            normalize_metric_type("ordinal")


class TestGuessMapping:
    """Coverage for the no-model default guess behind the CSV mapping widgets."""

    def test_maps_the_bundled_sample_csv(self) -> None:
        assert guess_mapping(["user_id", "variant", "converted", "revenue"]) == {
            "variant_col": "variant",
            "metric_col": "converted",
        }

    def test_is_case_insensitive_and_ignores_separators(self) -> None:
        assert guess_mapping(["Test_Group", "Conversion"]) == {
            "variant_col": "Test_Group",
            "metric_col": "Conversion",
        }

    def test_recognises_other_candidate_names(self) -> None:
        assert guess_mapping(["arm", "outcome"]) == {
            "variant_col": "arm",
            "metric_col": "outcome",
        }

    def test_never_maps_both_roles_to_the_same_column_first_match_wins(self) -> None:
        guess = guess_mapping(["converted", "revenue"])
        assert guess == {"metric_col": "converted"}

    def test_returns_only_the_keys_it_is_confident_about(self) -> None:
        assert guess_mapping(["user_id", "timestamp"]) == {}

    def test_returns_partial_guess_when_only_one_role_matches(self) -> None:
        assert guess_mapping(["user_id", "group"]) == {"variant_col": "group"}
