"""Failure handling of NCBI BLAST batches (``run_blast_batches``).

NCBI is faked at its boundary (``NCBIWWW.qblast`` and the XML parser): each
query sequence gets one hit, a job holding a "bad" sequence is rejected, and
``down`` makes every job fail. Waits between retries are skipped.
"""

import contextvars
from types import SimpleNamespace

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import trident.clients.ncbi as ncbi
import trident.core.http as http


def _batch(start: int, n: int) -> list[SeqRecord]:
    return [SeqRecord(Seq("ACGT"), id=f"s{i}") for i in range(start, start + n)]


def _record(seq_id: str) -> SimpleNamespace:
    """A BLAST record with a single hit, as NCBIXML.parse yields it."""
    hsp = SimpleNamespace(
        score=1,
        bits=1.0,
        expect=0.0,
        align_length=4,
        identities=4,
        gaps=0,
        query_start=1,
        query_end=4,
    )
    hit = SimpleNamespace(hit_def="hit", hit_id="hit", accession="A1", hsps=[hsp])
    return SimpleNamespace(query=seq_id, query_length=4, alignments=[hit])


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
        return ids  # the "handle" handed to the parser

    monkeypatch.setattr(ncbi.NCBIWWW, "qblast", qblast)
    monkeypatch.setattr(ncbi.NCBIXML, "parse", lambda ids: map(_record, ids))
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


def test_every_worker_runs_in_the_callers_context(ncbi_fake, monkeypatch):
    """Log context set by the caller (the UI log sink tag) reaches every
    BLAST worker thread."""
    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "caller", default=None
    )
    seen = []
    fake_qblast = ncbi.NCBIWWW.qblast

    def qblast(**kwargs):
        seen.append(var.get())
        return fake_qblast(**kwargs)

    monkeypatch.setattr(ncbi.NCBIWWW, "qblast", qblast)
    var.set("ui-run")

    ncbi.run_blast_batches([_batch(i, 1) for i in range(12)], n_threads=3)

    assert seen == ["ui-run"] * 12
