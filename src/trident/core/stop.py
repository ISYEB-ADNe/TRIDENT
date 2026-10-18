"""Stopping a running step.

The UI runs each job of a step inside ``stoppable(event)``, with one
``threading.Event`` per job. Once the event is set, every request the job
makes raises ``StepStopped`` (checked by ``core.http`` sessions and the BLAST
client), and waits between requests are cut short (``wait``). The clients'
per-item error handling then reports each remaining item as failed, so it is
not cached, while the finished items are cached as usual: running the step
again continues from there. Outside the UI (notebook, CLI) no event is set
and nothing changes.

The event is held in a context variable, so it reaches the threads a job
starts only when they run in a copy of its context: start them with
``run_threads``.
"""

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterable, Iterator

_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "trident_stop_event", default=None
)


class StepStopped(Exception):
    """Raised by a request made after the user stopped the step."""

    def __init__(self):
        super().__init__("step stopped by the user")


@contextlib.contextmanager
def stoppable(event: threading.Event) -> Iterator[None]:
    """Make *event* the stop signal of the code run in this block."""
    token = _event.set(event)
    try:
        yield
    finally:
        _event.reset(token)


def run_threads(fn: Callable, calls: Iterable[tuple], n_jobs: int) -> list:
    """``[fn(*args) for args in calls]``, on up to *n_jobs* threads (at most
    one per call; joblib's -1 means one per core), each call in a copy of the
    caller's context: the stop signal and the log context (the UI log sink tag)
    then reach the threads. The copies are made here, in the caller's thread
    (joblib would consume a generator from its own).
    """
    from joblib import Parallel, delayed

    tasks = [delayed(contextvars.copy_context().run)(fn, *args) for args in calls]
    if not tasks:
        return []
    if n_jobs > 0:
        n_jobs = min(n_jobs, len(tasks))
    return Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(tasks)


def stop_requested() -> bool:
    """True once the current step was stopped."""
    event = _event.get()
    return event is not None and event.is_set()


def check_stop() -> None:
    """Raise ``StepStopped`` if the current step was stopped."""
    if stop_requested():
        raise StepStopped()


def wait(seconds: float) -> None:
    """Sleep for *seconds*, or less if the step is stopped meanwhile. Never
    raises: the next request raises instead, so a wait outside a client's
    per-item error handling cannot abort the job."""
    event = _event.get()
    if event is None:
        time.sleep(seconds)
    else:
        event.wait(seconds)


def log_filter(record) -> bool:
    """Loguru filter that drops the records of a stopped job: its remaining
    items all fail at once, and their lines (one or more per item) would
    flood the log. Used by the console and the app's log panels."""
    return not stop_requested()
