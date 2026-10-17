"""HTTP helpers shared by the API clients: retries and session handling.

Two retry levels exist, and a client needs exactly one of them:

- Per HTTP request (``create_session`` / ``with_optional_session``): a
  ``requests`` session whose transport retries one request on connection
  errors, read timeouts and HTTP 429/500/502/503/504, with exponential backoff
  and ``Retry-After`` support. Used by the WoRMS, BOLD and GBIF clients. Retries
  are fast (seconds) and invisible to the calling code.
- Per call (``retry_call``): re-runs a whole function when it raises. For
  clients whose unit of work is a long job rather than one request: NCBI BLAST
  (a job is submitted, then polled for minutes; the client tolerates a few
  failed polls itself). Retries are slow (tens of seconds).

A call that still fails after its retries raises to the client, which reports
the item to the cache's ``failure_sink`` so it is retried on the next run
rather than cached as empty (see ``trident.core.database.save_to_db``).
"""

import time
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
    """Decorator that ensures a function receives a retrying ``session`` kwarg.

    Per-request retry level (see the module docstring). If the caller passes
    ``session=...``, that session is used as is, with its own retry settings;
    otherwise a temporary one is built with ``create_session(retries,
    backoff_factor)`` and closed after the call. Passing one session to many
    calls reuses its connection pool.

    The retries happen inside each ``session.get``: the wrapped function only
    sees the final response, or the exception once retries are exhausted. It
    does not re-run the wrapped function itself; for that, use ``retry_call``.

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


def retry_call(
    *,
    delays: tuple[float, ...],
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Decorator that re-runs the whole function when it raises.

    Per-call retry level (see the module docstring), for long jobs that cannot
    go through a retrying ``requests`` session, such as an NCBI BLAST
    submission. The function is called once, then once more after each delay
    in ``delays``, so ``delays=(30, 60)`` means up to 3 attempts: wait 30 s
    after the first failure, 60 s after the second. Each failure is logged as
    a warning. If the last attempt raises, its exception propagates unchanged.

    Only use it on functions that are safe to repeat (queries, not writes).

    Args:
        delays: Seconds to wait before each retry; ``len(delays)`` retries.
            Required: the right spacing depends on the service.
        exceptions: Exception types that trigger a retry; others propagate
            immediately.

    Returns:
        Decorator that adds the retries to the wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = len(delays) + 1
            for attempt, delay in enumerate(delays, start=1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{attempts}): "
                        f"{e}. Retrying in {delay:g} s."
                    )
                    time.sleep(delay)
            return func(*args, **kwargs)

        return wrapper

    return decorator
