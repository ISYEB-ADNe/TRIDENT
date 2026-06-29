from functools import wraps
from typing import Callable

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trident.core import config


def create_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    allowed_methods: set[str] | None = None,
    user_agent: str | None = None,
    pool_connections: int = 10,
    pool_maxsize: int = 20,
) -> requests.Session:
    """Create a requests.Session with retry logic and optional User-Agent.

    Args:
        retries: Max retry attempts per error type.
        backoff_factor: Multiplier for exponential backoff between retries.
        status_forcelist: HTTP status codes that trigger a retry.
        allowed_methods: HTTP methods eligible for retry. Defaults to GET only.
        user_agent: User-Agent header value. Falls back to
            ``config.user_agent()``.
        pool_connections: Number of pooled connections per host.
        pool_maxsize: Max connections in the pool.

    Returns:
        Configured Session with retry adapter mounted on http(s).
    """
    if allowed_methods is None:
        allowed_methods = {"GET"}

    session = requests.Session()

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    user_agent = user_agent or config.user_agent()
    session.headers.update({"User-Agent": user_agent})
    logger.debug(f"Session created (User-Agent: {user_agent})")

    return session


def with_optional_session(
    *,
    retries: int = 5,
    backoff_factor: float = 0.5,
) -> Callable:
    """Decorator that ensures a function receives a ``session`` kwarg.

    If the caller does not pass one, a temporary session is created
    and closed after the call.

    Args:
        retries: Retry attempts for the auto-created session.
        backoff_factor: Backoff multiplier for the auto-created session.

    Returns:
        Decorator that injects a session into the wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session: requests.Session | None = kwargs.get("session")
            created_here = session is None

            if created_here:
                session = create_session(
                    retries=retries,
                    backoff_factor=backoff_factor,
                )
                kwargs["session"] = session

            try:
                return func(*args, **kwargs)
            finally:
                if created_here:
                    session.close()

        return wrapper

    return decorator
