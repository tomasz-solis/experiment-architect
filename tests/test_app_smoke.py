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
