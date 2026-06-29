"""Centralized configuration — resolves values from st.secrets, env vars, or .env."""

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version

APP_NAME = "trident"


@lru_cache(maxsize=1)
def app_version() -> str:
    """The installed trident version, read from package metadata (pyproject).

    Single source of truth for the version string used everywhere (User-Agent,
    provenance). Returns 'unknown' if the package metadata is not resolvable.
    """
    try:
        return _pkg_version("trident")
    except PackageNotFoundError:
        return "unknown"


# Config keys
CONTACT_EMAIL = "CONTACT_EMAIL"
TRIDENT_LOG_LEVEL = "TRIDENT_LOG_LEVEL"

_dotenv_loaded = False


def _in_streamlit_run() -> bool:
    """True only when executing inside a live `streamlit run` script context.

    Touching `st.session_state` / `st.secrets` without a ScriptRunContext makes
    Streamlit log "missing ScriptRunContext" / "Session state does not function"
    warnings. Config is also read at package import (setup_logging) and from
    notebooks/tests, where no session exists, so we skip the Streamlit lookups
    there and fall back to env / .env.
    """
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


def _load_dotenv_once() -> None:
    """Load .env file once if python-dotenv is installed."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
    except ImportError:
        pass


def get(key: str, default: str | None = None) -> str | None:
    """Get a config value: st.secrets → session_state → env var → default.

    On first call, loads a .env file if python-dotenv is installed.
    """
    if _in_streamlit_run():
        try:
            import streamlit as st

            if key in st.secrets:
                return st.secrets[key]
            val = st.session_state.get(f"_config_{key}")
            if val:
                return val
        except Exception:
            pass
    _load_dotenv_once()
    return os.getenv(key, default)


def set(key: str, value: str) -> None:
    """Store a config value in session state (and env when running locally)."""
    if not is_streamlit_cloud():
        os.environ[key] = value
    if _in_streamlit_run():
        try:
            import streamlit as st

            st.session_state[f"_config_{key}"] = value
        except Exception:
            pass


def is_streamlit_cloud() -> bool:
    """Return True if running on Streamlit Cloud."""
    return os.path.exists("/mount/src")


def contact_email() -> str | None:
    """Return the configured contact email."""
    return get(CONTACT_EMAIL)


def user_agent(email: str | None = None) -> str:
    """Return a User-Agent string, e.g. 'trident/<version> (user@example.com)'."""
    email = email or contact_email()
    base = f"{APP_NAME}/{app_version()}"
    return f"{base} ({email})" if email else base
