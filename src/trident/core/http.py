"""HTTP helpers shared by the API clients: retries and session handling.

Two retry levels exist, and a client needs exactly one of them:

- Per HTTP request (``create_session`` / ``with_optional_session``): a
  ``requests`` session whose transport retries one request on connection
  errors, read timeouts and HTTP 429/500/502/503/504, with exponential backoff
  and ``Retry-After`` support. Used by the WoRMS, BOLD and GBIF clients. Retries
  are fast (seconds) and invisible to the calling code, but each one is logged
  (labelled by ``request_label``).
- Per call (``retry_call``): re-runs a whole function when it raises. For
  clients whose unit of work is a long job rather than one request: NCBI BLAST
  (a job is submitted, then polled for minutes; the client tolerates a few
  failed polls itself). Retries are slow (tens of seconds).

A call that still fails after its retries raises to the client, which reports
the item to the cache's ``failure_sink`` so it is retried on the next run
rather than cached as empty (see ``trident.core.database.save_to_db``).
"""

import contextlib
import contextvars
from functools import wraps
from typing import Callable

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trident.core import config, stop

# What the current thread's requests are for ("BOLD Gadus morhua"), set by the
# clients with request_label, so a retry line says which item it is about.
_request_label: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_label", default=None
)

# Statuses by which a server asks clients to slow down: retrying them is
# routine, so they are logged at DEBUG only.
_SLOW_DOWN_STATUSES = {429, 503}


@contextlib.contextmanager
def request_label(label: str):
    """Label the requests made in this block, for their retry lines."""
    token = _request_label.set(label)
    try:
        yield
    finally:
        _request_label.reset(token)


class _StoppableRetry(Retry):
    """urllib3 retries whose wait (backoff or Retry-After) is cut short by a
    stop, which then ends the request instead of retrying it. urllib3 calls
    ``sleep`` once the failed response is released, so no connection is left
    checked out. Each retry is logged, so a source answering slowly does not
    look stuck."""

    def increment(
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        # Raises once the retries are used up: the caller reports the failure.
        new = super().increment(method, url, response, error, _pool, _stacktrace)
        what = _request_label.get() or (_pool.host if _pool else "Request")
        if response is not None:
            reason = f"HTTP {response.status}"
            wait = new.get_retry_after(response) or new.get_backoff_time()
        else:
            # urllib3 errors start with the connection pool: keep what follows.
            reason = str(error).split(": ", 1)[-1].rstrip(".")
            wait = new.get_backoff_time()
        attempt = len(new.history)
        attempts = attempt + 1 + new.total if isinstance(new.total, int) else None
        of = f"/{attempts}" if attempts else ""
        slow_down = response is not None and response.status in _SLOW_DOWN_STATUSES
        logger.log(
            "DEBUG" if slow_down else "WARNING",
            f"{what} failed (attempt {attempt}{of}): {reason}. "
            + (f"Retrying in {wait:g} s." if wait else "Retrying now."),
        )
        return new

    def sleep(self, response=None) -> None:
        seconds = 0.0
        if self.respect_retry_after_header and response:
            seconds = self.get_retry_after(response) or 0.0
        if not seconds:
            seconds = self.get_backoff_time()
        if seconds > 0:
            stop.wait(seconds)
        stop.check_stop()


class _StoppableAdapter(HTTPAdapter):
    """HTTP adapter that sends nothing once the step is stopped (raises
    ``StepStopped``, see ``trident.core.stop``)."""

    def send(self, request, *args, **kwargs):
        stop.check_stop()
        return super().send(request, *args, **kwargs)


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
        Configured Session with retry adapter mounted on http(s). Once the
        step is stopped, its requests raise ``StepStopped``.
    """
    if allowed_methods is None:
        allowed_methods = {"GET"}

    session = requests.Session()

    retry = _StoppableRetry(
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

    adapter = _StoppableAdapter(
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
    no_retry: tuple[type[BaseException], ...] = (),
    describe: Callable[[], str] | None = None,
) -> Callable:
    """Decorator that re-runs the whole function when it raises.

    Per-call retry level (see the module docstring), for long jobs that cannot
    go through a retrying ``requests`` session, such as an NCBI BLAST
    submission. The function is called once, then once more after each delay
    in ``delays``, so ``delays=(30, 60)`` means up to 3 attempts: wait 30 s
    after the first failure, 60 s after the second. Each failure is logged as
    a warning. If the last attempt raises, its exception propagates unchanged.
    A stopped step (``StepStopped``) is not retried, and a stop cuts the wait
    between attempts short.

    Only use it on functions that are safe to repeat (queries, not writes).

    Args:
        delays: Seconds to wait before each retry; ``len(delays)`` retries.
            Required: the right spacing depends on the service.
        exceptions: Exception types that trigger a retry; others propagate
            immediately.
        no_retry: Subtypes of ``exceptions`` that propagate immediately, for
            failures a repeat cannot fix.
        describe: Returns what the call is, for the retry warning (e.g. the
            BLAST batch being run); the function name by default.

    Returns:
        Decorator that adds the retries to the wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = len(delays) + 1
            for attempt, delay in enumerate((*delays, None), start=1):
                # Also for functions without a stop check of their own (qblast).
                stop.check_stop()
                try:
                    return func(*args, **kwargs)
                except (stop.StepStopped, *no_retry):
                    raise
                except exceptions as e:
                    if delay is None:
                        raise
                    what = describe() if describe else func.__name__
                    logger.warning(
                        f"{what} failed (attempt {attempt}/{attempts}): "
                        f"{str(e).rstrip('.')}. Retrying in {delay:g} s."
                    )
                    stop.wait(delay)

        return wrapper

    return decorator
