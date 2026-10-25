"""Uploading an analysis already in the results folder opens it, without a copy."""

import sqlite3

from trident.ui.inputs import _saved_copy


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
