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

UPLOAD_KEYS = (CSV_UPLOAD, DID_UPLOAD, RDD_UPLOAD)


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
