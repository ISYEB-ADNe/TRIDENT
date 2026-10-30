"""The inputs of an analysis: the FASTA file read into sequences, and an
uploaded analysis already in the results folder opened without a copy."""

import sqlite3

import pandas as pd

from streamlit.testing.v1 import AppTest

from trident.pipelines.fasta_pipeline import run_fasta_workflow
from trident.ui.inputs import _fasta_analysis_name, _fasta_handle, _saved_copy


def _analysis_file(path, seq_id="s1", with_sequences=True):
    """A minimal analysis file: the sequences table it is opened from."""
    with sqlite3.connect(path) as conn:
        table = "sequences" if with_sequences else "other"
        conn.execute(f"CREATE TABLE {table} (seq_id TEXT, dna_sequence TEXT)")
        conn.execute(f"INSERT INTO {table} VALUES (?, 'ACGT')", (seq_id,))
    conn.close()
    return path.read_bytes()


def test_upload_opens_the_saved_analysis_with_the_same_content(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    data = _analysis_file(tmp_path / "AbuDhabi.db")
    (tmp_path / "AbuDhabi (2).db").write_bytes(data)
    changed = _analysis_file(tmp_path / "Changed.db", seq_id="s2")
    (tmp_path / "Changed.db").unlink()
    not_analysis = _analysis_file(tmp_path / "Other.db", with_sequences=False)

    # The copy named like the upload first, whatever the folder order...
    assert _saved_copy(data, "AbuDhabi (2)") == tmp_path / "AbuDhabi (2).db"
    # ...but the active one before it, so its upload is seen as already active.
    assert _saved_copy(data, "AbuDhabi (2)", "AbuDhabi") == tmp_path / "AbuDhabi.db"
    # An analysis changed since is imported as a new one.
    assert _saved_copy(changed, "AbuDhabi") is None
    # A file without sequences is never opened, even when it is in the folder.
    assert _saved_copy(not_analysis, "Other") is None
    # Looking at the saved analyses leaves their bytes as they were...
    assert (tmp_path / "AbuDhabi.db").read_bytes() == data
    assert (tmp_path / "AbuDhabi (2).db").read_bytes() == data
    # ...and a copy committed elsewhere without a change to its rows is still
    # the same analysis.
    committed = tmp_path / "elsewhere" / "AbuDhabi.db"
    committed.parent.mkdir()
    committed.write_bytes(data)
    with sqlite3.connect(committed) as conn:
        conn.execute("PRAGMA user_version = 0")
    conn.close()
    assert committed.read_bytes() != data
    assert _saved_copy(committed.read_bytes(), "AbuDhabi") == tmp_path / "AbuDhabi.db"


def _upload(text, name):
    return _fasta_handle(name, text.encode())


def test_fasta_file_gives_one_row_per_record_saved_by_filename(tmp_path):
    """Each record becomes a row with its id, length and sequence (Windows line
    endings and a UTF-8 byte order mark too), read back the same from the
    analysis; a file without records gives none."""
    text = "\ufeff>m1 first\r\nACGT\r\nAC\r\n>m2\r\nGGG\r\n>m3\r\nT\r\n"
    db_path = str(tmp_path / "analysis.db")

    live, _ = run_fasta_workflow(_upload(text, "motus.fasta"), db_path=db_path)
    cached, _ = run_fasta_workflow(_upload(text, "motus.fasta"), db_path=db_path)
    assert live.to_dict("list") == {
        "seq_id": ["m1", "m2", "m3"],
        "seq_length": [6, 3, 1],
        "dna_sequence": ["ACGTAC", "GGG", "T"],
    }
    pd.testing.assert_frame_equal(live, cached)
    assert run_fasta_workflow(_upload("", "empty.fasta"), db_path=db_path)[0].empty


def test_edited_fasta_under_a_saved_name_starts_a_new_analysis(tmp_path, monkeypatch):
    """The sequences are saved by filename: a file whose sequences changed
    since gets "name (2)" instead of the old analysis, and reopens it later."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    first = ">m1\nACGT\n>m2\nGGG\n"
    edited = ">m1\nACGT\n>m3\nTTT\n"
    run_fasta_workflow(_upload(first, "demo.fasta"), db_path=str(tmp_path / "demo.db"))

    def name(text):
        return _fasta_analysis_name(
            "demo", run_fasta_workflow(_upload(text, "demo.fasta"))[0]
        )

    assert name(first) == "demo"
    assert name(edited) == "demo (2)"
    run_fasta_workflow(
        _upload(edited, "demo.fasta"), db_path=str(tmp_path / "demo (2).db")
    )
    assert name(edited) == "demo (2)"
    assert name(">m4\nCCC\n") == "demo (3)"


def test_unreadable_fasta_is_reported_and_saves_nothing(tmp_path, monkeypatch):
    """A file that is not UTF-8, not FASTA or without sequences shows an error
    instead of a traceback, and leaves no analysis behind."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))

    def page(data):
        from trident.ui.inputs import _start_fasta_analysis

        _start_fasta_analysis("bad.fasta", data)

    for data in [b">m1 \xe9t\xe9\nACGT\n", b"notes\n>m1\nACGT\n", b""]:
        at = AppTest.from_function(page, args=(data,))
        at.run()
        assert not at.exception, data
        assert len(at.error) == 1, data
    assert list(tmp_path.iterdir()) == []


def test_same_sequences_under_another_filename_open_the_analysis_unchanged(
    tmp_path, monkeypatch
):
    """x.fa with the sequences of the analysis x (made from x.fasta) opens it
    without adding them a second time."""
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    text = ">m1\nACGT\n>m2\nGGG\n"
    run_fasta_workflow(_upload(text, "x.fasta"), db_path=str(tmp_path / "x.db"))

    def page():
        from trident.ui.inputs import _start_fasta_analysis

        _start_fasta_analysis("x.fa", b">m1\nACGT\n>m2\nGGG\n")

    at = AppTest.from_function(page)
    at.run()
    assert at.session_state["analysis_name"] == "x"
    with sqlite3.connect(tmp_path / "x.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0] == 2
    conn.close()
