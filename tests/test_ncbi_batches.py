"""Failure handling of NCBI BLAST jobs and batches (``run_blast_batches``).

NCBI is faked at its boundary (the URL API requests and the XML2 archive
reader): each query sequence gets one real hit (from the XML2 test fixture), a
job holding a "bad" sequence is rejected, a "stuck" one stays waiting, and
``down`` makes every request fail. Waits are skipped. The polling of a job is tested against a fake that
serves the real XML2 fixture (``url_api_fake``).
"""

import contextvars
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import trident.clients.ncbi as ncbi
import trident.core.stop as stop


def _batch(start: int, n: int) -> list[SeqRecord]:
    return [SeqRecord(Seq("ACGT"), id=f"s{i}") for i in range(start, start + n)]


_FIXTURE = Path(__file__).parent / "fixtures" / "blast_xml2_one_query.zip"
_TEMPLATE = ncbi.read_xml2_archive(_FIXTURE.read_bytes())[0]


def _record(seq_id: str):
    """A Bio.Blast record for ``seq_id`` with a single hit."""
    record = copy.deepcopy(_TEMPLATE)
    record.query.description = seq_id
    del record[1:]
    return record


def _query_ids(fasta: str) -> list[str]:
    return [line[1:].split()[0] for line in fasta.splitlines() if line.startswith(">")]


def _reply(text="", content=b""):
    return SimpleNamespace(text=text, content=content, raise_for_status=lambda: None)


@pytest.fixture
def ncbi_fake(monkeypatch):
    """NCBI's URL API, one job per submission, done at the first poll. The
    first ``stuck[id]`` jobs holding sequence ``id`` wait forever."""
    fake = SimpleNamespace(
        bad=set(), stuck={}, down=False, jobs=0, queries={}, max_queries=None
    )
    stuck_rids = set()

    def post(url, data, **kwargs):
        if data["CMD"] == "Put":
            fake.jobs += 1
        if fake.down:
            raise requests.ConnectionError("NCBI unreachable")
        if data["CMD"] == "Put":
            ids = _query_ids(data["QUERY"])
            if fake.bad & set(ids):
                return _reply("<p>Message ID#24 Error: NCBI rejected the query</p>")
            rid = f"R{fake.jobs}"
            fake.queries[rid] = ids
            for i in set(ids) & {i for i, n in fake.stuck.items() if n > 0}:
                fake.stuck[i] -= 1
                stuck_rids.add(rid)
            return _reply(f"QBlastInfoBegin\n    RID = {rid}\n    RTOE = 30\n")
        if data["RID"] in stuck_rids:
            return _reply("QBlastInfoBegin\n\tStatus=WAITING\nQBlastInfoEnd")
        return _reply(content=b"PK" + data["RID"].encode())  # the archive "bytes"

    def read_archive(data):
        ids = fake.queries[data[2:].decode()]
        if fake.max_queries and len(ids) > fake.max_queries:
            raise ncbi.BlastCpuLimit("CPU usage limit was exceeded")
        return [_record(i) for i in ids]

    monkeypatch.setattr(ncbi.requests, "post", post)
    monkeypatch.setattr(ncbi, "read_xml2_archive", read_archive)
    monkeypatch.setattr(stop, "wait", lambda s: None)
    monkeypatch.setattr(ncbi, "BLAST_REQUEST_INTERVAL", 0.0)
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


def test_batch_over_the_cpu_limit_is_split_until_it_fits(ncbi_fake):
    """A search NCBI stops at its CPU limit is halved until the parts fit,
    without retries, and does not count as NCBI being down: every batch of a
    run gets the same treatment and every sequence gets its hits."""
    ncbi_fake.max_queries = 2
    n_batches = ncbi.BLAST_CIRCUIT_THRESHOLD + 2
    batches = [_batch(b * 8, 8) for b in range(n_batches)]

    df, failed = ncbi.run_blast_batches(batches, n_threads=1)

    assert failed == []
    assert len(df) == n_batches * 8
    assert ncbi_fake.jobs == n_batches * (1 + 2 + 4)  # 8, then 4 + 4, then 4 x 2


def test_bad_sequence_in_a_heavy_batch_fails_alone(ncbi_fake):
    """A batch both too heavy and holding a bad sequence still loses only that
    sequence: each part is split on its own cause."""
    ncbi_fake.max_queries = 2
    ncbi_fake.bad = {"s3"}

    df, failed = ncbi.run_blast_batches([_batch(0, 8)], n_threads=1)

    assert failed == ["s3"]
    assert sorted(df["seq_id"]) == sorted(f"s{i}" for i in range(8) if i != 3)


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
    fake_post = ncbi.requests.post

    def post(url, data, **kwargs):
        if data["CMD"] == "Put":
            seen.append(var.get())
        return fake_post(url, data, **kwargs)

    monkeypatch.setattr(ncbi.requests, "post", post)
    var.set("ui-run")

    ncbi.run_blast_batches([_batch(i, 1) for i in range(12)], n_threads=3)

    assert seen == ["ui-run"] * 12


def test_stop_keeps_finished_batches_and_skips_the_rest(ncbi_fake, monkeypatch):
    """After a stop, the batches already done are returned and the others are
    reported failed at once, for the cache to retry: no retry, no split."""
    event = threading.Event()
    fake_post = ncbi.requests.post

    def post(url, data, **kwargs):
        reply = fake_post(url, data, **kwargs)
        if data["CMD"] == "Get":
            event.set()  # the user stops once the first batch is done
        return reply

    monkeypatch.setattr(ncbi.requests, "post", post)
    batches = [_batch(b * 2, 2) for b in range(3)]

    with stop.stoppable(event):
        df, failed = ncbi.run_blast_batches(batches, n_threads=1)

    assert sorted(df["seq_id"]) == ["s0", "s1"]
    assert failed == ["s2", "s3", "s4", "s5"]
    assert ncbi_fake.jobs == 1


def test_archive_missing_a_query_fails_the_job(ncbi_fake, monkeypatch):
    """A result without one record per query is an error (retried), not
    queries without hits."""
    read_archive = ncbi.read_xml2_archive
    monkeypatch.setattr(ncbi, "read_xml2_archive", lambda data: read_archive(data)[:1])
    with pytest.raises(ValueError, match="1 of 2 queries"):
        ncbi._run_blast_job_once(">s0\nACGT\n>s1\nACGT\n", 1e-10, 10, None)


@pytest.fixture
def url_api_fake(monkeypatch):
    """NCBI's URL API: a job is WAITING for ``polls`` polls, then ends with
    ``status`` (READY serves the one-query XML2 fixture). The first
    ``glitches`` polls get a page without a status."""
    fake = SimpleNamespace(polls=2, status="READY", glitches=0, requests=[])

    def post(url, data, **kwargs):
        fake.requests.append(data)
        if data["CMD"] == "Put":
            return _reply("QBlastInfoBegin\n    RID = R1\n    RTOE = 20\nQBlastInfoEnd")
        if fake.glitches:
            fake.glitches -= 1
            return _reply("<html>Service temporarily unavailable</html>")
        if len(fake.requests) - 1 <= fake.polls:
            status = "WAITING"
        elif fake.status == "READY":
            return _reply(content=_FIXTURE.read_bytes())
        else:
            status = fake.status
        return _reply(f"QBlastInfoBegin\n\tStatus={status}\nQBlastInfoEnd")

    monkeypatch.setattr(ncbi.requests, "post", post)
    monkeypatch.setattr(stop, "wait", lambda s: None)
    monkeypatch.setattr(ncbi, "BLAST_REQUEST_INTERVAL", 0.0)
    return fake


def test_url_api_client_polls_until_ready(url_api_fake):
    """The job is polled while WAITING, then its XML2 result is parsed."""
    records = ncbi._run_blast_job_once(">s0\nACGT\n", 1e-10, 10, None)

    assert len(records) == 1 and len(records[0]) > 0
    assert [d["CMD"] for d in url_api_fake.requests] == ["Put"] + ["Get"] * 3


def test_url_api_client_fails_a_failed_job(url_api_fake):
    """A job NCBI reports FAILED raises (so it is retried)."""
    url_api_fake.status = "FAILED"
    with pytest.raises(ValueError, match="status FAILED"):
        ncbi._run_blast_job_once(">s0\nACGT\n", 1e-10, 10, None)


def test_url_api_client_keeps_the_job_over_transient_poll_errors(url_api_fake):
    """Failed polls below the limit keep polling the same job; it is not
    resubmitted."""
    url_api_fake.glitches = ncbi.BLAST_POLL_FAILURES - 1
    records = ncbi._run_blast_job_once(">s0\nACGT\n", 1e-10, 10, None)

    assert len(records) == 1
    assert sum(d["CMD"] == "Put" for d in url_api_fake.requests) == 1


def test_slow_job_makes_way_for_the_next_batch(ncbi_fake, monkeypatch):
    """With one thread, a job slow at NCBI lets the next batch run meanwhile,
    on a spare thread."""
    monkeypatch.setattr(ncbi, "BLAST_SLOW_JOB", -1.0)  # slow at its first poll
    second_done = threading.Event()
    first_polls, done = [], []
    fake_post = ncbi.requests.post

    def post(url, data, **kwargs):
        if data["CMD"] == "Get":
            ids = ncbi_fake.queries[data["RID"]]
            if ids[0] == "s0":  # first batch: WAITING, then done after the second
                first_polls.append(data["RID"])
                if len(first_polls) == 1:
                    return _reply("QBlastInfoBegin\n\tStatus=WAITING\nQBlastInfoEnd")
                second_done.wait(timeout=5)  # times out if the second never ran
            done.append(ids[0])
            if ids[0] == "s2":
                second_done.set()
        return fake_post(url, data, **kwargs)

    monkeypatch.setattr(ncbi.requests, "post", post)

    df, failed = ncbi.run_blast_batches([_batch(0, 2), _batch(2, 2)], n_threads=1)

    assert done == ["s2", "s0"]
    assert failed == [] and len(df) == 4


def test_stuck_job_is_resubmitted_once_then_fails(ncbi_fake):
    """A job past the give-up time is resubmitted once. Stuck again, its
    sequences fail without a split (each part would add a search to NCBI's
    queue), and NCBI does not count as down."""
    ncbi_fake.stuck = {"s0": 1, "s2": 2}  # first batch stuck once, second twice
    circuit = ncbi.BlastCircuit(threshold=1)  # one failure would open it

    df, failed = ncbi.run_blast_batches(
        [_batch(0, 2), _batch(2, 2)], n_threads=1, circuit=circuit, give_up=-1.0
    )

    assert sorted(df["seq_id"]) == ["s0", "s1"]
    assert sorted(failed) == ["s2", "s3"]
    assert ncbi_fake.jobs == 4
    assert not circuit.is_open


def test_give_up_while_splitting_sends_nothing_more(ncbi_fake, monkeypatch):
    """A part of a split batch given up stops the split: the parts not sent
    yet fail with it, and the parts done keep their hits."""
    ncbi_fake.bad = {"s3"}  # the batch fails, then its half s2 + s3
    ncbi_fake.stuck = {"s2": 2}  # then s2 alone is given up twice
    submitted = []
    fake_post = ncbi.requests.post

    def post(url, data, **kwargs):
        if data["CMD"] == "Put":
            submitted.append(_query_ids(data["QUERY"]))
        return fake_post(url, data, **kwargs)

    monkeypatch.setattr(ncbi.requests, "post", post)

    df, failed = ncbi.run_blast_batches([_batch(0, 4)], n_threads=1, give_up=-1.0)

    assert sorted(df["seq_id"]) == ["s0", "s1"]
    assert sorted(failed) == ["s2", "s3"]
    assert ["s3"] not in submitted  # never sent alone
