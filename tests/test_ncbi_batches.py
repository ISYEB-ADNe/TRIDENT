"""Failure handling of NCBI BLAST batches (``run_blast_batches``).

NCBI is faked at its boundary (``Blast.qblast`` and the XML2 archive reader):
each query sequence gets one real hit (from the XML2 test fixture), a job
holding a "bad" sequence is rejected, and ``down`` makes every job fail. Waits
between retries are skipped.
"""

import contextvars
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import trident.clients.ncbi as ncbi
import trident.core.http as http


def _batch(start: int, n: int) -> list[SeqRecord]:
    return [SeqRecord(Seq("ACGT"), id=f"s{i}") for i in range(start, start + n)]


_FIXTURE = Path(__file__).parent / "fixtures" / "blast_xml2_one_query.zip"
_TEMPLATE = ncbi.read_xml2_archive(_FIXTURE.read_bytes())[0]


class _Stream:
    """What Blast.qblast returns: a response to read, usable with ``with``."""

    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _record(seq_id: str):
    """A Bio.Blast record for ``seq_id`` with a single hit."""
    record = copy.deepcopy(_TEMPLATE)
    record.query.description = seq_id
    del record[1:]
    return record


@pytest.fixture
def ncbi_fake(monkeypatch):
    fake = SimpleNamespace(bad=set(), down=False, jobs=0)

    def qblast(sequence, **kwargs):
        fake.jobs += 1
        if fake.down:
            raise ConnectionError("NCBI unreachable")
        ids = [
            line[1:].split()[0]
            for line in sequence.splitlines()
            if line.startswith(">")
        ]
        if fake.bad & set(ids):
            raise ValueError("NCBI rejected the query")
        return _Stream(ids)  # the archive "bytes"

    monkeypatch.setattr(ncbi.Blast, "qblast", qblast)
    monkeypatch.setattr(ncbi, "read_xml2_archive", lambda ids: list(map(_record, ids)))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    return fake


def test_bad_sequence_fails_alone(ncbi_fake):
    ncbi_fake.bad = {"s3"}

    df, failed = ncbi.run_blast_batches([_batch(0, 10)], n_threads=1)

    assert failed == ["s3"]
    assert sorted(df["seq_id"]) == sorted(f"s{i}" for i in range(10) if i != 3)


def test_bad_sequences_in_many_batches_do_not_stop_retries(ncbi_fake):
    """A batch failed by its own sequence does not count as NCBI being down,
    so every batch still has its bad sequence isolated."""
    n_batches = ncbi.BLAST_CIRCUIT_THRESHOLD + 2
    batches = [_batch(b * 10, 10) for b in range(n_batches)]
    ncbi_fake.bad = {f"s{b * 10}" for b in range(n_batches)}

    df, failed = ncbi.run_blast_batches(batches, n_threads=1)

    assert sorted(failed) == sorted(ncbi_fake.bad)
    assert len(df) == n_batches * 9


def test_ncbi_down_fails_everything_at_bounded_cost(ncbi_fake):
    """With NCBI down, every sequence is reported failed. Only the first
    batches get retries and a split; once the circuit opens, each remaining
    batch costs a single job."""
    ncbi_fake.down = True
    n_batches = ncbi.BLAST_CIRCUIT_THRESHOLD + 5
    batches = [_batch(b * 2, 2) for b in range(n_batches)]

    df, failed = ncbi.run_blast_batches(batches, n_threads=1)

    assert df.empty
    assert len(failed) == n_batches * 2
    retried_cost = len(ncbi.BLAST_RETRY_DELAYS) + 1 + 2  # attempts + 2 halves
    max_jobs = ncbi.BLAST_CIRCUIT_THRESHOLD * retried_cost + (
        n_batches - ncbi.BLAST_CIRCUIT_THRESHOLD
    )
    assert ncbi_fake.jobs <= max_jobs


def test_no_batch_returns_empty_hits_with_columns():
    df, failed = ncbi.run_blast_batches([])
    assert df.empty and failed == []
    assert list(df.columns) == ncbi.BLAST_RAW_COLUMNS


def test_every_worker_runs_in_the_callers_context(ncbi_fake, monkeypatch):
    """Log context set by the caller (the UI log sink tag) reaches every
    BLAST worker thread."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "caller", default=None
    )
    seen = []
    fake_qblast = ncbi.Blast.qblast

    def qblast(**kwargs):
        seen.append(var.get())
        return fake_qblast(**kwargs)

    monkeypatch.setattr(ncbi.Blast, "qblast", qblast)
    var.set("ui-run")

    ncbi.run_blast_batches([_batch(i, 1) for i in range(12)], n_threads=3)

    assert seen == ["ui-run"] * 12


def test_archive_missing_a_query_fails_the_job(monkeypatch):
    """A result without one record per query is an error (retried), not
    queries without hits."""
    monkeypatch.setattr(ncbi.Blast, "qblast", lambda **k: _Stream(["s0"]))
    monkeypatch.setattr(ncbi, "read_xml2_archive", lambda ids: list(map(_record, ids)))
    with pytest.raises(ValueError, match="1 of 2 queries"):
        ncbi._run_qblast_once(">s0\nACGT\n>s1\nACGT\n", 1e-10, 10, None)
