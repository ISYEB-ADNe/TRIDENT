"""Per-call retries (``retry_call``): the call is re-run after each delay and
only the last failure is raised. A stopped step sends nothing and is not
retried."""

import threading
import time
from types import SimpleNamespace

import pytest
import trident.core.stop as stop
from trident.core.http import create_session, retry_call


@pytest.fixture
def sleeps(monkeypatch):
    """Record the waits instead of sleeping."""
    recorded = []
    monkeypatch.setattr(stop, "wait", recorded.append)
    return recorded


def _flaky(failures: int, exc: type[Exception] = ConnectionError):
    """A function that raises ``failures`` times, then returns "ok"."""
    calls = []

    def func():
        calls.append(1)
        if len(calls) <= failures:
            raise exc(f"failure {len(calls)}")
        return "ok"

    return func, calls


def test_retry_call_returns_once_an_attempt_succeeds(sleeps):
    func, calls = _flaky(failures=2)
    assert retry_call(delays=(30, 60))(func)() == "ok"
    assert len(calls) == 3
    assert sleeps == [30, 60]


def test_retry_call_raises_the_last_error_when_all_attempts_fail(sleeps):
    func, calls = _flaky(failures=5)
    with pytest.raises(ConnectionError, match="failure 3"):
        retry_call(delays=(30, 60))(func)()
    assert len(calls) == 3
    assert sleeps == [30, 60]


def test_retry_call_does_not_retry_unlisted_exceptions(sleeps):
    func, calls = _flaky(failures=1, exc=KeyError)
    with pytest.raises(KeyError):
        retry_call(delays=(30,), exceptions=(ConnectionError,))(func)()
    assert len(calls) == 1
    assert sleeps == []


def test_stopped_step_sends_nothing_and_is_not_retried(sleeps):
    """Once the step is stopped, a session request raises StepStopped before
    reaching the network, and retry_call raises it without calling or
    waiting."""
    event = threading.Event()
    event.set()
    func, calls = _flaky(failures=0)

    with stop.stoppable(event):
        with pytest.raises(stop.StepStopped):
            # An address that cannot resolve: a broken check fails, offline.
            create_session().get("https://trident.invalid/")
        with pytest.raises(stop.StepStopped):
            retry_call(delays=(30, 60))(func)()
    assert calls == []
    assert sleeps == []


def test_stop_ends_a_request_waiting_to_be_retried():
    """A stop cuts urllib3's wait before a retry (Retry-After) short and ends
    the request instead of retrying it."""
    event = threading.Event()
    retry = create_session().get_adapter("https://").max_retries
    response = SimpleNamespace(headers={"Retry-After": "120"})
    event.set()

    with stop.stoppable(event), pytest.raises(stop.StepStopped):
        start = time.monotonic()
        retry.sleep(response)
    assert time.monotonic() - start < 1
