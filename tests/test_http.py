"""Per-call retries (``retry_call``): the call is re-run after each delay and
only the last failure is raised. A stopped step sends nothing and is not
retried. A function taking an optional session gets one built for the call
unless given one."""

import threading
import time
from types import SimpleNamespace

import pytest
import trident.core.stop as stop
import trident.core.http as http
from trident.core.http import create_session, retry_call, with_optional_session


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


def test_retry_call_retries_listed_errors_after_each_delay(sleeps):
    """It returns once an attempt succeeds, raises the last error when all
    fail, and does not retry an error it was not given."""
    func, calls = _flaky(failures=2)
    assert retry_call(delays=(30, 60))(func)() == "ok"
    assert len(calls) == 3 and sleeps == [30, 60]

    sleeps.clear()
    func, calls = _flaky(failures=5)
    with pytest.raises(ConnectionError, match="failure 3"):
        retry_call(delays=(30, 60))(func)()
    assert len(calls) == 3 and sleeps == [30, 60]

    sleeps.clear()
    func, calls = _flaky(failures=1, exc=KeyError)
    with pytest.raises(KeyError):
        retry_call(delays=(30,), exceptions=(ConnectionError,))(func)()
    assert len(calls) == 1 and sleeps == []


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


def test_a_session_is_built_for_the_call_only_when_none_is_given(monkeypatch):
    """Given by position or by keyword, the caller's session is used as is;
    otherwise one is built with the call's user agent and closed after it."""
    built = []

    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    def fake_create_session(**kwargs):
        built.append(kwargs["user_agent"])
        return FakeSession()

    monkeypatch.setattr(http, "create_session", fake_create_session)

    @with_optional_session()
    def fetch(name, session=None, user_agent=None):
        return session

    mine = FakeSession()
    assert fetch("a", mine) is mine
    assert fetch("a", session=mine) is mine
    assert built == []

    own = fetch("a", None, "trident-test")
    assert built == ["trident-test"]
    assert own.closed and not mine.closed


@pytest.mark.parametrize("wait, requests_sent", [("60", 3), ("3600", 1)])
def test_a_long_retry_after_fails_at_once_and_a_short_one_is_waited(
    sleeps, wait, requests_sent
):
    """A server asking to wait longer than RETRY_AFTER_MAX (a block) is not
    asked again: the answer comes back at once for the caller to fail the
    item. A shorter wait is honoured before each retry."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Busy(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(429)
            self.send_header("Retry-After", wait)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Busy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        answer = create_session(retries=2).get(
            f"http://127.0.0.1:{server.server_port}/", timeout=5
        )
    finally:
        server.shutdown()
    assert answer.status_code == 429
    assert len(seen) == requests_sent
    assert sleeps == [60.0] * (requests_sent - 1)
