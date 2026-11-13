from loguru import logger

from .logging import setup_logging

# Library default: stay silent and add no handlers on import. The Streamlit
# app (app.py) and notebooks call setup_logging()
# to turn logging on. This keeps `import trident` free of global side effects
# (no hijacking a consumer's loguru config, no Streamlit bare-mode warnings).
logger.disable("trident")

__all__ = ["setup_logging"]
