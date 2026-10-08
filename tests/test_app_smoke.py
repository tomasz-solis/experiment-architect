"""End-to-end smoke tests that run the Streamlit app through AppTest.

The app script and its snapshot builders execute Streamlit at import time, so
they cannot be imported directly. AppTest runs the real script in a simulated
context and surfaces any exception, which gives the snapshot builders and the
per-lens rendering genuine regression coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stats.prereg import PreRegistration, build_preregistration, serialise_preregistration
from ui.state import PREREG_UPLOAD

LENSES = [
    "Experiment design",
    "Power and plan",
    "Manual result read",
    "Raw CSV audit",
    "Causal fallback",
]

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "examples" / "sample_ab_test.csv"
SAMPLE_CSV_BYTES = SAMPLE_CSV_PATH.read_bytes()

DID_CSV_BYTES = pd.DataFrame(
    {
        "unit": [1, 1, 2, 2, 3, 3, 4, 4],
        "period": [0, 1, 0, 1, 0, 1, 0, 1],
        "treated": [0, 0, 0, 0, 1, 1, 1, 1],
        "outcome": [10.0, 11.0, 9.5, 10.5, 10.0, 15.0, 9.0, 14.0],
    }
).to_csv(index=False).encode("utf-8")

RDD_CSV_BYTES = pd.DataFrame(
    {
        "score": [10, 20, 30, 40, 60, 70, 80, 90],
        "treated": [0, 0, 0, 0, 1, 1, 1, 1],
        "outcome": [5.0, 5.5, 6.0, 6.5, 9.0, 9.5, 10.0, 10.5],
    }
).to_csv(index=False).encode("utf-8")

# 20 users, 3 rows each: three times the rows per unit the clustering check
# tolerates, so a session-level or event-level export like this must warn.
CLUSTERED_CSV_BYTES = pd.DataFrame(
    {
        "user_id": [user for user in range(20) for _ in range(3)],
        "variant": (["control", "treatment"] * 30)[:60],
        "converted": ([0, 1, 0, 1, 1, 0] * 10)[:60],
    }
).to_csv(index=False).encode("utf-8")


class FakeMappingResponses:
    """Deterministic column-mapping stand-in for ``llm.client.ask_agent_json``.

    Dispatches on ``expected_keys`` so the same fake serves the CSV, DiD, and
    RDD mapping prompts, and records every call so a test can assert the
    prompt-building path actually ran.
    """

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        client: object,
        provider: str,
        ai_enabled: bool,
        system_role: str,
        user_prompt: str,
        expected_keys: Sequence[str],
        max_attempts: int = 2,
    ) -> dict[str, str] | None:
        """Return a fixed mapping for the prompt's expected keys."""
        self.calls.append(tuple(expected_keys))
        if "variant_col" in expected_keys:
            return {"variant_col": "variant", "metric_col": "converted", "metric_type": "binary"}
        if "unit_col" in expected_keys:
            return {
                "unit_col": "unit",
                "time_col": "period",
                "treatment_col": "treated",
                "outcome_col": "outcome",
            }
        if "running_var" in expected_keys:
            return {"running_var": "score", "treatment_col": "treated", "outcome_col": "outcome"}
        return None


@pytest.fixture
def app() -> AppTest:
    """Run the app once with no datasets uploaded (AI features disabled)."""
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


@pytest.fixture
def ai_app(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, FakeMappingResponses]:
    """Run the app with a stubbed LLM client so the AI-enabled mapping paths execute.

    Patches ``llm.client.create_llm_client`` and ``llm.client.ask_agent_json`` before
    the script runs; ``app.py`` imports both by name at module scope, so the
    patched objects are what it binds to on this run.
    """
    fake_responses = FakeMappingResponses()
    monkeypatch.setattr("llm.client.create_llm_client", lambda: (object(), True, "openai"))
    monkeypatch.setattr("llm.client.ask_agent_json", fake_responses)
    app_test = AppTest.from_file(APP_PATH, default_timeout=60).run()
    return app_test, fake_responses


def test_app_runs_without_exception(app: AppTest) -> None:
    assert not app.exception
    assert app.markdown  # hero and section copy rendered


@pytest.mark.parametrize("lens", LENSES)
def test_each_lens_renders(app: AppTest, lens: str) -> None:
    """Switching the review lens rebuilds the hero/summary without error."""
    app.radio(key="review_focus").set_value(lens).run()
    assert not app.exception


def test_design_snapshot_recomputes_on_aggressive_inputs(app: AppTest) -> None:
    """An aggressive MDE drives the sanity checks down the 'fail' branch."""
    app.number_input(key="main_base").set_value(10.0).run()
    app.number_input(key="main_mde").set_value(60.0).run()
    app.number_input(key="main_traffic").set_value(100).run()
    assert not app.exception


def test_design_review_and_design_tab_agree_on_an_uneven_split(app: AppTest) -> None:
    """Regression test for the design review hardcoding a 50/50 split while the
    design tab used the real one, which understated the required sample and let
    an underpowered plan read as a comfortable 'caution'.

    With a 30/70 split, the review's "needed" figure must equal the tab's own
    "Total sample" metric, since both must size the same plan.
    """
    app.slider(key="main_split").set_value(30).run()
    app.button(key="sanity_button").click().run()
    assert not app.exception

    total_sample = next(m.value for m in app.metric if m.label == "Total sample")

    rendered = " ".join(
        block.value for block in list(app.success) + list(app.warning) + list(app.error)
    )
    assert "Traffic vs MDE" in rendered
    assert f"{total_sample} needed" in rendered


def test_manual_lens_handles_count_mismatch(app: AppTest) -> None:
    """More conversions than visitors must be caught, not crash the lens."""
    app.radio(key="review_focus").set_value("Manual result read").run()
    app.number_input(key="manual_visitors_a").set_value(100).run()
    app.number_input(key="manual_conversions_a").set_value(500).run()
    assert not app.exception


def test_power_lens_sizes_a_continuous_metric(app: AppTest) -> None:
    """The variance-driven sizing path recomputes without error."""
    app.radio(key="review_focus").set_value("Power and plan").run()
    app.number_input(key="power_sd").set_value(240.0).run()
    app.number_input(key="power_mde_abs").set_value(2.0).run()
    assert not app.exception


def test_locking_a_plan_makes_the_readout_verify_against_it(app: AppTest) -> None:
    """Locking in Signal 02 turns on the plan-versus-delivery check downstream."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.number_input(key="manual_visitors_a").set_value(1000).run()
    app.number_input(key="manual_visitors_b").set_value(1400).run()
    app.button(key="manual_result_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.markdown)
    assert "What you promised, and what you got" in rendered


def test_material_not_floor_result_does_not_render_green(app: AppTest) -> None:
    """A point estimate above the bar with a pessimistic floor below it is info, not success."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.radio(key="review_focus").set_value("Manual result read").run()
    app.number_input(key="manual_visitors_a").set_value(4000).run()
    app.number_input(key="manual_conversions_a").set_value(400).run()
    app.number_input(key="manual_visitors_b").set_value(4000).run()
    app.number_input(key="manual_conversions_b").set_value(460).run()
    app.button(key="manual_result_button").click().run()
    assert not app.exception

    assert any("Probably worth shipping" in block.value for block in app.info)
    assert not any("Probably worth shipping" in block.value for block in app.success)


def test_open_downside_adds_a_caption(app: AppTest) -> None:
    """When the interval floor still reaches a real loss, the readout must say so."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.radio(key="review_focus").set_value("Manual result read").run()
    app.number_input(key="manual_visitors_a").set_value(500).run()
    app.number_input(key="manual_conversions_a").set_value(50).run()
    app.number_input(key="manual_visitors_b").set_value(500).run()
    app.number_input(key="manual_conversions_b").set_value(48).run()
    app.button(key="manual_result_button").click().run()
    assert not app.exception

    assert any("has not ruled out a loss" in block.value for block in app.caption)


def test_guardrail_inputs_appear_once_a_plan_is_locked(app: AppTest) -> None:
    """Sizing a guardrail is not the same as checking one: the readout must be
    able to collect what the guardrail actually did once a plan exists."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    for suffix in (
        "guardrail_control_events",
        "guardrail_control_n",
        "guardrail_variant_events",
        "guardrail_variant_n",
    ):
        widget = app.number_input(key=f"manual_{suffix}")
        assert widget.value == 0


def test_guardrail_reading_flags_a_broken_guardrail(app: AppTest) -> None:
    """Filling in a guardrail that clearly got worse must say the ship decision
    does not rest on the primary metric alone."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.number_input(key="manual_guardrail_control_events").set_value(200).run()
    app.number_input(key="manual_guardrail_control_n").set_value(10_000).run()
    app.number_input(key="manual_guardrail_variant_events").set_value(400).run()
    app.number_input(key="manual_guardrail_variant_n").set_value(10_000).run()
    app.button(key="manual_result_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.markdown)
    assert "What happened to the guardrail" in rendered
    assert any(
        "does not belong to the primary metric alone" in block.value for block in app.error
    )


def test_manual_snapshot_checks_srm_against_the_locked_plan_not_5050(app: AppTest) -> None:
    """A 70/30 plan must not make the manual summary card flag a matching 70/30 read as SRM.

    Regression test for the manual-lens hero card judging the split against a
    hardcoded 50/50 instead of the locked pre-registration split.
    """
    app.slider(key="main_split").set_value(70).run()
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.number_input(key="manual_visitors_a").set_value(300).run()
    app.number_input(key="manual_visitors_b").set_value(700).run()
    app.radio(key="review_focus").set_value("Manual result read").run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.markdown)
    assert "Sample ratio mismatch" not in rendered


def test_csv_lens_runs_end_to_end_with_manual_columns_and_no_llm(app: AppTest) -> None:
    """The raw CSV path must not need an LLM: picking columns by hand is enough to run it.

    Regression test for the analysis being wired up entirely inside the
    ``ask_agent_json`` branch, so the button did nothing on a deployment
    without an API key.
    """
    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "sample_ab_test.csv", SAMPLE_CSV_BYTES, "text/csv"
    ).run()
    app.button(key="csv_analysis_button").click().run()
    assert not app.exception

    assert any("Mapped: Variant=" in block.value for block in app.success)


def test_csv_lens_states_the_resolved_control_arm(app: AppTest) -> None:
    """The caption must name which value was treated as control and which as variant."""
    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "sample_ab_test.csv", SAMPLE_CSV_BYTES, "text/csv"
    ).run()
    app.button(key="csv_analysis_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.caption)
    assert "Treating `control` as control and `treatment` as variant." in rendered


def test_csv_lens_flags_a_metric_mismatch_against_the_locked_plan(app: AppTest) -> None:
    """A locked plan's primary metric is checked against the chosen outcome column."""
    app.button(key="prereg_lock_button").click().run()
    assert not app.exception

    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "sample_ab_test.csv", SAMPLE_CSV_BYTES, "text/csv"
    ).run()
    app.selectbox(key="csv_metric_col").set_value("revenue").run()
    app.radio(key="csv_metric_type").set_value("continuous").run()
    app.button(key="csv_analysis_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.warning)
    assert "This plan was written for 'Checkout conversion', but this readout is on 'revenue'" in rendered


def test_csv_lens_warns_when_the_file_holds_several_rows_per_unit(app: AppTest) -> None:
    """A session- or event-level export must warn that the tests below treat every
    row as independent, once the analyst names which column is the randomised unit."""
    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "clustered.csv", CLUSTERED_CSV_BYTES, "text/csv"
    ).run()
    app.selectbox(key="csv_unit_col").set_value("user_id").run()
    app.button(key="csv_analysis_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.warning)
    assert "3.0 rows per" in rendered
    assert "independent observation" in rendered


def test_csv_lens_skips_the_clustering_check_when_no_unit_column_is_picked(app: AppTest) -> None:
    """Leaving the unit column at its "(none)" default must not fire the warning."""
    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "clustered.csv", CLUSTERED_CSV_BYTES, "text/csv"
    ).run()
    app.button(key="csv_analysis_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.warning)
    assert "rows per" not in rendered


def test_csv_ai_suggestion_fills_the_selectboxes(
    ai_app: tuple[AppTest, FakeMappingResponses],
) -> None:
    """Regression test for the missing ``tabulate`` dependency crashing this button.

    ``df.head(3).to_markdown()`` used to raise ``ImportError`` here, and every
    AppTest fixture ran with AI disabled, so nothing ever exercised this path.
    """
    app, fake_responses = ai_app
    app.radio(key="review_focus").set_value("Raw CSV audit").run()
    app.file_uploader(key="csv_upload").upload(
        "sample_ab_test.csv", SAMPLE_CSV_BYTES, "text/csv"
    ).run()
    app.button(key="csv_suggest_mapping_button").click().run()
    assert not app.exception

    assert ("variant_col", "metric_col", "metric_type") in fake_responses.calls
    assert app.selectbox(key="csv_variant_col").value == "variant"
    assert app.selectbox(key="csv_metric_col").value == "converted"


def test_did_mapping_prompt_builds_without_raising(
    ai_app: tuple[AppTest, FakeMappingResponses],
) -> None:
    """Regression test for ``to_markdown()`` crashing the DiD mapping prompt."""
    app, fake_responses = ai_app
    app.selectbox(key="causal_has_cutoff").set_value("No").run()
    app.selectbox(key="causal_has_control").set_value("Yes (unaffected users)").run()
    app.selectbox(key="causal_is_opt_in").set_value("No (forced)").run()
    assert not app.exception

    app.file_uploader(key="did_upload").upload("panel.csv", DID_CSV_BYTES, "text/csv").run()
    app.button(key="did_analyze").click().run()
    assert not app.exception

    assert ("unit_col", "time_col", "treatment_col", "outcome_col") in fake_responses.calls


def test_rdd_mapping_prompt_builds_without_raising(
    ai_app: tuple[AppTest, FakeMappingResponses],
) -> None:
    """Regression test for ``to_markdown()`` crashing the RDD mapping prompt."""
    app, fake_responses = ai_app
    app.selectbox(key="causal_has_cutoff").set_value("Yes (e.g. score > 600)").run()
    assert not app.exception

    app.file_uploader(key="rdd_upload").upload("rdd.csv", RDD_CSV_BYTES, "text/csv").run()
    app.button(key="rdd_analyze").click().run()
    assert not app.exception

    assert ("running_var", "treatment_col", "outcome_col") in fake_responses.calls


def _make_plan_for_restore() -> PreRegistration:
    """Build a small locked plan to round-trip through the restore uploader."""
    return build_preregistration(
        primary_metric="Checkout conversion",
        metric_layer="conversion",
        baseline=0.12,
        mde_relative=0.05,
        n_total=40_000,
        ramp_days=0,
        enrolment_days=44,
        maturation_days=0,
        daily_new_eligible=900.0,
    )


def test_restoring_a_downloaded_plan_makes_the_readout_verify_against_it(app: AppTest) -> None:
    """A plan restored from a downloaded file must reach the same downstream checks as a locked one.

    Covers Item 3: the plan must survive a page refresh, which in this app
    means it can be re-uploaded and picked up exactly like a freshly locked one.
    """
    plan_bytes = serialise_preregistration(_make_plan_for_restore()).encode("utf-8")

    app.file_uploader(key=PREREG_UPLOAD).upload("plan.json", plan_bytes, "application/json").run()
    assert not app.exception
    assert any("Plan restored" in block.value for block in app.success)

    app.number_input(key="manual_visitors_a").set_value(1000).run()
    app.number_input(key="manual_visitors_b").set_value(1400).run()
    app.button(key="manual_result_button").click().run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.markdown)
    assert "What you promised, and what you got" in rendered


def test_restoring_a_malformed_plan_shows_the_validation_error(app: AppTest) -> None:
    """A corrupted or hand-edited upload must surface the plain-language error, not crash."""
    app.file_uploader(key=PREREG_UPLOAD).upload(
        "plan.json", b"not json", "application/json"
    ).run()
    assert not app.exception

    rendered = " ".join(block.value for block in app.error)
    assert "not valid JSON" in rendered
