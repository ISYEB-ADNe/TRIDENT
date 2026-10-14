"""Per-call retries (``retry_call``): the call is re-run after each delay and
only the last failure is raised."""

import pytest
import trident.core.http as http
from trident.core.http import retry_call


@pytest.fixture
def sleeps(monkeypatch):
    """Record the waits instead of sleeping."""
    recorded = []
    monkeypatch.setattr(http.time, "sleep", recorded.append)
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
