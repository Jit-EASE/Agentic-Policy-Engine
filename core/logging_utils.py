# logging_utils.py

import traceback
from typing import Any, Dict
import streamlit as st


def log_error(context: str, error: Exception) -> None:
    """Log error details into Streamlit sidebar error panel."""
    with st.sidebar:
        st.markdown("### ⚠️ Engine Error Log")
        st.error(f"Context: `{context}`")
        st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))


def log_info(message: str, data: Dict[str, Any] | None = None) -> None:
    """Optional structured info logging into sidebar."""
    with st.sidebar:
        st.markdown("### ℹ️ Engine Signal")
        st.caption(message)
        if data:
            st.json(data, expanded=False)
