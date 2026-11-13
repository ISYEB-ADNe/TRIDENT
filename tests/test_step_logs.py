"""A step's log takes the lines of the threads it runs on (AppTest, offline):
sessions share one process on Cloud, so a line from another thread, even one
given the id of a thread the step used and that has ended, is not taken."""

from streamlit.testing.v1 import AppTest


def _page():
    import threading
    from types import SimpleNamespace

    import streamlit as st
    from loguru import logger

    from trident.ui.ui import StreamlitLogSink

    sink = StreamlitLogSink(prefix="t_")
    worker = threading.Thread(target=sink.allow_current_thread)
    worker.start()
    worker.join()

    def under_trident(record):
        record["name"] = "trident.test"

    log = logger.patch(under_trident)
    log.info("from the step")
    # Another thread that got the ended worker's id.
    log.patch(
        lambda r: r.update(thread=SimpleNamespace(id=worker.ident, name="other"))
    ).info("from another session")
    sink.flush_to_ui()
    sink.close()
    st.session_state.lines = [line for _, line in sink.lines]


def test_a_step_log_takes_no_line_from_an_ended_threads_id():
    at = AppTest.from_function(_page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    lines = at.session_state.lines
    assert any("from the step" in line for line in lines)
    assert not any("from another session" in line for line in lines)
