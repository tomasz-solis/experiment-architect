"""Streamlit session-state keys and shared readers.

The widget keys live here so the lens snapshot builders (``ui/snapshots.py``)
and the widget definitions (``app.py``) reference the same constant. Before
this, a key like ``"main_base"`` was written as a literal in both places, and a
rename in one spot would silently desync the snapshot from the widget.
"""

from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Design lens
MAIN_BASELINE = "main_base"
MAIN_MDE = "main_mde"
MAIN_TRAFFIC = "main_traffic"
MAIN_SPLIT = "main_split"

# Power-and-plan lens
POWER_METRIC_LAYER = "power_metric_layer"
POWER_SD = "power_sd"
POWER_BASELINE_MEAN = "power_baseline_mean"
POWER_MDE_ABS = "power_mde_abs"
POWER_ALPHA = "power_alpha"
POWER_POWER = "power_power"
POWER_RHO = "power_rho"
POWER_DAILY_NEW = "power_daily_new"
POWER_MATURATION = "power_maturation"
POWER_RAMP = "power_ramp"
POWER_GUARDRAIL_BASELINE = "power_guardrail_baseline"
POWER_CLUSTER_SIZE = "power_cluster_size"
POWER_ICC = "power_icc"
POWER_UPLOAD = "power_upload"
PREREG_PLAN = "prereg_plan"
PREREG_UPLOAD = "prereg_upload"

# Manual-result lens
MANUAL_VISITORS_A = "manual_visitors_a"
MANUAL_CONVERSIONS_A = "manual_conversions_a"
MANUAL_VISITORS_B = "manual_visitors_b"
MANUAL_CONVERSIONS_B = "manual_conversions_b"
MANUAL_N_COMPARISONS = "manual_n_comparisons"
MANUAL_PEEKED_EARLY = "manual_peeked_early"

# Causal-fallback lens
CAUSAL_HAS_CUTOFF = "causal_has_cutoff"
CAUSAL_HAS_CONTROL = "causal_has_control"
CAUSAL_IS_OPT_IN = "causal_is_opt_in"

# Dataset uploads
CSV_UPLOAD = "csv_upload"
DID_UPLOAD = "did_upload"
RDD_UPLOAD = "rdd_upload"

# Raw CSV audit lens: manual column mapping and the model's optional suggestion
CSV_VARIANT_COL = "csv_variant_col"
CSV_METRIC_COL = "csv_metric_col"
CSV_METRIC_TYPE = "csv_metric_type"
CSV_UNIT_COL = "csv_unit_col"
CSV_SUGGESTED_VARIANT_COL = "csv_suggested_variant_col"
CSV_SUGGESTED_METRIC_COL = "csv_suggested_metric_col"
CSV_SUGGESTED_METRIC_TYPE = "csv_suggested_metric_type"

UPLOAD_KEYS = (CSV_UPLOAD, DID_UPLOAD, RDD_UPLOAD, POWER_UPLOAD, PREREG_UPLOAD)


def read_uploaded_dataframe(widget_key: str) -> pd.DataFrame | None:
    """Read a CSV uploaded via a Streamlit file uploader key, or ``None``."""
    uploaded_file = st.session_state.get(widget_key)
    if uploaded_file is None:
        return None

    try:
        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception:
        logger.warning("Could not parse uploaded file for key '%s'.", widget_key)
        return None
