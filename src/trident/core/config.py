"""Centralized configuration: resolves values from st.secrets, env vars, or the
repository's .streamlit/secrets.toml."""

import os
import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

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
RESULTS_DIR = "RESULTS_DIR"


def _in_streamlit_run() -> bool:
    """True only when executing inside a live `streamlit run` script context.

    Touching `st.session_state` / `st.secrets` without a ScriptRunContext makes
    Streamlit log "missing ScriptRunContext" / "Session state does not function"
    warnings. Config is also read at package import (setup_logging) and from
    notebooks/tests, where no session exists, so we skip the Streamlit lookups
    there and fall back to env / secrets.toml.
    """
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


def _source_root() -> Path | None:
    """The repository root in a source checkout (git clone or downloaded
    archive, as documented), None for an installed package (this file then
    sits in site-packages)."""
    root = Path(__file__).resolve().parents[3]
    return root if (root / "pyproject.toml").exists() else None


@lru_cache(maxsize=1)
def _repo_secrets() -> dict:
    """The repository's .streamlit/secrets.toml, read directly.

    Streamlit reads it only from the folder the app starts in, and not at all
    in notebooks or scripts: this makes the same file work for both.
    """
    root = _source_root()
    path = root / ".streamlit" / "secrets.toml" if root else None
    if not path or not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def get(key: str, default: str | None = None) -> str | None:
    """Get a config value: st.secrets → session_state → env var → the
    repository's secrets.toml → default."""
    if _in_streamlit_run():
        try:
            import streamlit as st

            from streamlit.runtime.scriptrunner import get_script_run_ctx

            if key in st.secrets:
                return st.secrets[key]
            # Session state belongs to a script run: worker threads (step jobs,
            # BLAST batches) have none, and reading it there only logs warnings.
            if get_script_run_ctx(suppress_warning=True) is not None:
                val = st.session_state.get(f"_config_{key}")
                if val:
                    return val
        except Exception:
            pass
    if key in os.environ:
        return os.environ[key]
    return _repo_secrets().get(key, default)


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


def results_dir() -> Path:
    """Folder of the saved analyses, the same whatever folder TRIDENT starts from.

    The RESULTS_DIR setting wins. Otherwise it is results/ at the repository
    root, or in the user's home for an installed package.
    """
    configured = get(RESULTS_DIR)
    if configured:
        return Path(configured).expanduser()
    root = _source_root()
    return root / "results" if root else Path.home() / "TRIDENT" / "results"


def contact_email() -> str | None:
    """Return the configured contact email."""
    return get(CONTACT_EMAIL)


def user_agent(email: str | None = None) -> str:
    """Return a User-Agent string, e.g. 'trident/<version> (user@example.com)'."""
    email = email or contact_email()
    base = f"{APP_NAME}/{app_version()}"
    return f"{base} ({email})" if email else base
