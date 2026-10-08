"""Unit tests for standalone UI component behaviour that does not need a full app run."""

from __future__ import annotations

import pytest

from stats.frequentist import SRMResult
from stats.prereg import GuardrailReading
from ui.components import show_guardrail_reading, show_srm_warning


def make_mismatch(expected_share: float = 0.5, observed_share: float = 0.6) -> SRMResult:
    return {
        "observed_share": observed_share,
        "expected_share": expected_share,
        "p_value": 0.001,
        "has_mismatch": True,
    }


class TestShowSrmWarning:
    """Two SRM checks on one page must be told apart by their stage label."""

    def test_no_stage_label_renders_the_plain_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr("ui.components.st.warning", messages.append)

        show_srm_warning(make_mismatch())

        assert len(messages) == 1
        assert messages[0].startswith("Sample ratio mismatch")

    def test_stage_label_prefixes_the_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr("ui.components.st.warning", messages.append)

        show_srm_warning(make_mismatch(), stage_label="Before cleaning")

        assert messages[0].startswith("Before cleaning: Sample ratio mismatch")

    def test_two_calls_with_different_labels_are_distinguishable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr("ui.components.st.warning", messages.append)

        show_srm_warning(make_mismatch(), stage_label="Before cleaning")
        show_srm_warning(make_mismatch(), stage_label="After cleaning")

        assert messages[0] != messages[1]
        assert "Before cleaning" in messages[0]
        assert "After cleaning" in messages[1]

    def test_clean_split_renders_nothing_regardless_of_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr("ui.components.st.warning", messages.append)

        clean: SRMResult = {
            "observed_share": 0.5,
            "expected_share": 0.5,
            "p_value": 0.9,
            "has_mismatch": False,
        }
        show_srm_warning(clean, stage_label="Before cleaning")

        assert messages == []


class TestShowGuardrailReading:
    """Rendering a guardrail reading safely without "inf" or "nan" in the output."""

    def test_infinite_relative_change_renders_without_inf_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captions: list[str] = []
        monkeypatch.setattr("ui.components.st.caption", captions.append)
        monkeypatch.setattr("ui.components.st.columns", lambda n: tuple([type("obj", (), {"metric": lambda *a, **kw: None})() for _ in range(n)]))
        monkeypatch.setattr("ui.components.st.error", lambda x: None)

        reading: GuardrailReading = {
            "observed_control_rate": 0.0,
            "observed_variant_rate": 0.001,
            "relative_change": float("inf"),
            "ci_relative": (float("nan"), float("nan")),
            "detectable_harm": None,
            "status": "fail",
            "note": "Test note",
        }
        show_guardrail_reading(reading)

        caption_text = " ".join(captions)
        assert "inf%" not in caption_text.lower()
        assert "infinite" in caption_text.lower()

    def test_nan_ci_renders_without_nan_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captions: list[str] = []
        monkeypatch.setattr("ui.components.st.caption", captions.append)
        monkeypatch.setattr("ui.components.st.columns", lambda n: tuple([type("obj", (), {"metric": lambda *a, **kw: None})() for _ in range(n)]))
        monkeypatch.setattr("ui.components.st.success", lambda x: None)

        reading: GuardrailReading = {
            "observed_control_rate": 0.0,
            "observed_variant_rate": 0.0,
            "relative_change": 0.0,
            "ci_relative": (float("nan"), float("nan")),
            "detectable_harm": None,
            "status": "ok",
            "note": "Test note",
        }
        show_guardrail_reading(reading)

        caption_text = " ".join(captions)
        assert "nan%" not in caption_text.lower()
        assert "cannot be computed" in caption_text.lower()
