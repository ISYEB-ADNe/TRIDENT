"""The inputs of an analysis: the FASTA file read into sequences, and an
uploaded analysis already in the results folder opened without a copy."""

import sqlite3

import pandas as pd

from trident.pipelines.fasta_pipeline import run_fasta_workflow
from trident.ui.inputs import _fasta_handle, _saved_copy


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


def _upload(text, name):
    return _fasta_handle(name, text.encode())


def test_fasta_file_gives_one_row_per_record_saved_by_filename(tmp_path):
    """Each record becomes a row with its id, length and sequence (Windows line
    endings too), read back the same from the analysis; a file without
    records gives none."""
    text = ">m1 first\r\nACGT\r\nAC\r\n>m2\r\nGGG\r\n>m3\r\nT\r\n"
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
