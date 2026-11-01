"""Helpers for keeping typed API keys in Streamlit session state."""

from __future__ import annotations

from typing import Any, MutableMapping


def persist_api_key(provider: str, state: MutableMapping[str, Any]) -> None:
    """Copy a widget value to a non-widget key before Streamlit cleans widget state."""
    widget_key = f"api_key_input_{provider}"
    session_key = f"api_key_session_{provider}"
    value = state.get(widget_key)
    if value:
        state[session_key] = value
    else:
        state.pop(session_key, None)
