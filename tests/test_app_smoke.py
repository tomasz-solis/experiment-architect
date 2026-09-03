"""End-to-end smoke tests that run the Streamlit app through AppTest.

The app script and its snapshot builders execute Streamlit at import time, so
they cannot be imported directly. AppTest runs the real script in a simulated
context and surfaces any exception, which gives the snapshot builders and the
per-lens rendering genuine regression coverage.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

LENSES = [
    "Experiment design",
    "Power and plan",
    "Manual result read",
    "Raw CSV audit",
    "Causal fallback",
]


@pytest.fixture
def app() -> AppTest:
    """Run the app once with no datasets uploaded (AI features disabled)."""
    return AppTest.from_file("app.py", default_timeout=60).run()


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
