import sys
from pathlib import Path

from loguru import logger

from trident.core import config

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

LOG_LEVELS = ("INFO", "DEBUG")

# Tracks whether the default (no-arg) setup has already run. Lives in this
# imported module, so it survives Streamlit script reruns: setup_logging() can
# be called from the entry point on every rerun and stays a no-op after the
# first, matching the old once-per-process behavior. An explicit level/log_file
# (e.g. set_log_level) always reconfigures.
_configured = False


def get_log_level() -> str:
    """Return the current log level from config, defaulting to INFO."""
    return config.get(config.TRIDENT_LOG_LEVEL, "INFO").upper()


def set_log_level(level: str) -> None:
    """Set the log level and reconfigure logging."""
    config.set(config.TRIDENT_LOG_LEVEL, level)
    setup_logging(level)


def setup_logging(level: str | None = None, log_file: str | Path | None = None) -> None:
    """Configure logging for the trident package.

    Sets up a console handler and an optional rotating file handler.
    The log level is resolved in order: function argument →
    TRIDENT_LOG_LEVEL environment variable → default "INFO".

    Importing `trident` does NOT call this (the package keeps loguru silent via
    `logger.disable("trident")`); the entry points (the Streamlit app, the
    `trident` launcher) and notebooks call it explicitly to turn logging on.

    Args:
        level: Log level string (e.g. "DEBUG", "INFO"). Case-insensitive.
        log_file: Path to the log file. File logging is disabled when None.
    """
    global _configured
    explicit = level is not None or log_file is not None
    if _configured and not explicit:
        return

    if level is None:
        level = get_log_level()

    logger.remove()
    logger.enable("trident")  # undo the import-time disable in __init__

    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
    )

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            level="DEBUG",
            format=LOG_FORMAT,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )

    _configured = True
