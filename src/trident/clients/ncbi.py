"""NCBI BLAST client: submit jobs, parse hits, apply the MOL filter (barcoding
gap or similarity range), plot the hits."""

import contextvars
import io
import os
import re
import threading
import time
import zipfile
from collections.abc import Callable
from functools import wraps
from typing import NamedTuple

import pandas as pd
import requests
from Bio import Blast
from Bio.SeqRecord import SeqRecord
from loguru import logger

from .fasta import blast_record, clean_sequence, records_to_fasta_string
from trident.core import config, stop
from trident.core.http import retry_call
from trident.core.constants import (
    BLAST_DATABASE,
    NCBI_GAP_SIZE,
    NCBI_GAP_MIN_TOP,
    NCBI_QUERY_COVER,
    NCBI_METHOD,
)
from trident.core.utils import (
    duration,
    extract_specific_epithet,
    notify_progress,
    to_binomial,
    pluralize,
)


### CREDENTIALS ###


def set_ncbi_email(email: str | None) -> None:
    """Set ``Blast.email``, sent with each BLAST job (a process-level global)."""
    if not email:
        return
    Blast.email = email
    logger.debug(f"NCBI contact email set: {email}")


def setup_ncbi_credentials() -> None:
    """Configure Biopython with the contact email from ``core.config``.

    Sets ``Blast.email``. Optional but recommended: NCBI may throttle anonymous
    requests.
    """
    email = config.contact_email()

    if not email:
        logger.debug("CONTACT_EMAIL not set: NCBI requests will be anonymous")
    else:
        set_ncbi_email(email)


_ncbi_credentials_loaded = False
# The parallel BLAST batches all call a require_ncbi_auth function at once.
_ncbi_credentials_lock = threading.Lock()


def reload_ncbi_credentials() -> None:
    """Force re-reading credentials from config (e.g. after Settings change)."""
    global _ncbi_credentials_loaded
    with _ncbi_credentials_lock:
        setup_ncbi_credentials()
        _ncbi_credentials_loaded = True


def require_ncbi_auth(func: Callable) -> Callable:
    """Decorator that loads NCBI credentials once before the first call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _ncbi_credentials_loaded
        with _ncbi_credentials_lock:
            if not _ncbi_credentials_loaded:
                setup_ncbi_credentials()
                _ncbi_credentials_loaded = True
        return func(*args, **kwargs)

    return wrapper


### BATCH UTILITIES ###


def create_sequence_batches(
    sequences: list[SeqRecord],
    batch_size: int | None = None,
    n_batches: int | None = None,
) -> list[list[SeqRecord]]:
    """Split a sequence list into evenly-sized batches for parallel processing.

    Batches are balanced: any remainder is distributed one element at a time
    across the first batches, so sizes differ by at most 1.

    Provide either batch_size OR n_batches, not both. If neither is given,
    defaults to batch_size=50.

    Args:
        sequences: List of sequences to split.
        batch_size: Target size of each batch.
        n_batches: Desired number of batches.

    Returns:
        List of batches, each batch being a sublist of sequences.

    Raises:
        ValueError: If both batch_size and n_batches are provided.
    """
    if batch_size is not None and n_batches is not None:
        raise ValueError("Provide either batch_size or n_batches, not both")

    total = len(sequences)
    if total == 0:
        return []

    if n_batches is None:
        if batch_size is None:
            batch_size = 50
        n_batches = (total + batch_size - 1) // batch_size

    base_size = total // n_batches
    remainder = total % n_batches

    batches = []
    start = 0
    for i in range(n_batches):
        end = start + base_size + (1 if i < remainder else 0)
        batches.append(sequences[start:end])
        start = end

    return batches


### BLAST SUBMISSION ###

# Output columns of parse_blast_records, also present when there are no hits.
BLAST_RAW_COLUMNS = [
    "seq_id",
    "query_length",
    "hit_def",
    "hit_id",
    "hit_accession",
    "score",
    "bits",
    "evalue",
    "align_length",
    "identities",
    "gaps",
    "query_start",
    "query_end",
    "ncbi_taxid",
    "sciname",
]


def parse_blast_records(records: list) -> pd.DataFrame:
    """Parse Bio.Blast records (XML2) into a flat DataFrame, one row per hit.

    A hit is one database record; when the query aligns to it in several
    pieces (HSPs), the row describes the best one (highest score, then bit
    score), so the record counts once.
    A hit that merges several identical database records is described by its
    first record: its title (hit_def), accession, NCBI taxid and the scientific
    name NCBI's taxonomy gives that taxid (sciname). Records with no hits
    produce no rows.

    Args:
        records: Bio.Blast Record objects, as returned by Blast.parse().

    Returns:
        DataFrame with the columns of BLAST_RAW_COLUMNS.
    """
    rows = []

    for record in records:
        # In XML2 the query id is NCBI's (Query_123); the FASTA id is its title.
        seq_id = record.query.description
        query_length = len(record.query.seq)

        for hit in record:
            target = hit.target
            hsp = max(
                hit, key=lambda h: (h.score, h.annotations["bit score"]), default=None
            )
            if hsp is None:
                continue
            # Bio.Blast coordinates are 0-based and end-exclusive; BLAST
            # positions are 1-based and inclusive.
            query_coords = hsp.coordinates[1]
            rows.append(
                {
                    "seq_id": seq_id,
                    "query_length": query_length,
                    "hit_def": target.description,
                    "hit_id": target.id,
                    "hit_accession": target.name,
                    "score": hsp.score,
                    "bits": hsp.annotations["bit score"],
                    "evalue": hsp.annotations["evalue"],
                    "align_length": hsp.shape[1],
                    "identities": hsp.annotations["identity"],
                    "gaps": hsp.annotations.get("gaps", 0),
                    "query_start": int(min(query_coords[0], query_coords[-1])) + 1,
                    "query_end": int(max(query_coords[0], query_coords[-1])),
                    "ncbi_taxid": target.annotations.get("taxid"),
                    "sciname": target.annotations.get("sciname"),
                }
            )

    return pd.DataFrame(rows, columns=BLAST_RAW_COLUMNS)


# Label of the batch the current thread is running ("Batch 3/40"),
# set by submit_blast_batch, so the client's log lines (submission, slow job,
# retry) say which batch they are about.
_job_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "blast_job_label", default="BLAST job"
)

# Threads a parallel run has beyond its n_threads, for batches to start while
# slow jobs wait (see _BlastRun).
BLAST_SLOW_THREADS = 2


class _BlastRun:
    """Scheduling shared by the workers of one parallel run (``run_blast_pool``).

    At most *n_threads* batches hold a place, that is run at their normal
    pace. A job waiting long at NCBI is slow (see ``_Job``): it is still
    polled, less often, but its batch frees its place, so one of the
    BLAST_SLOW_THREADS spare threads can start the next batch. The batch takes
    a place again before it submits another job (a resubmission, a retry, a
    split half). A free place goes to the waiting batch first in the queue, so
    batches start in queue order, and a batch resubmitting a job goes before
    the batches after it. *give_up* is the run's give-up time in seconds (see
    ``BlastGaveUp``).
    """

    def __init__(self, n_batches: int, n_threads: int, give_up: float):
        self.give_up = give_up
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count() or 1
        self.pool_size = min(n_batches, self.n_threads + BLAST_SLOW_THREADS)
        self._changed = threading.Condition()
        self._normal = 0  # batches holding a place
        self._waiting: set[int] = set()  # queue positions of waiting batches

    def take_place(self, position: int) -> None:
        """Wait for a place for the batch at *position* in the queue (or for a
        stop: the batch then fails at once)."""
        with self._changed:
            # By queue position, not first come: otherwise the thread that
            # freed a place takes it again for its next batch, and a waiting
            # batch can be passed over for the whole run.
            self._waiting.add(position)
            while (
                position != min(self._waiting) or self._normal >= self.n_threads
            ) and not stop.stop_requested():
                self._changed.wait(timeout=1.0)
            self._waiting.discard(position)
            self._normal += 1
            self._changed.notify_all()  # the next batch may have a place too

    def free_place(self) -> None:
        with self._changed:
            self._normal -= 1
            self._changed.notify_all()


class _Place:
    """The place of the batch run by the current thread, if it holds one."""

    def __init__(self, run: _BlastRun, position: int):
        self.run = run
        self.position = position  # of the batch in the run's queue
        self.held = False

    def take(self) -> None:
        if not self.held:
            self.run.take_place(self.position)
            self.held = True

    def free(self) -> None:
        if self.held:
            self.run.free_place()
            self.held = False


# Place of the batch the current thread runs, set by run_blast_pool (None
# outside a parallel run).
_place: contextvars.ContextVar[_Place | None] = contextvars.ContextVar(
    "blast_place", default=None
)


class _Job:
    """One job attempt: takes its batch's place again if a slow job freed it,
    and frees it once this job is slow (``set_slow``). A job still waiting at
    NCBI is slow after BLAST_SLOW_JOB (or half its give-up time if shorter),
    and given up after its run's give-up time (BLAST_GIVE_UP outside a run)."""

    def __init__(self):
        self.place = _place.get()
        self.slow = False
        self.give_up = BLAST_GIVE_UP if self.place is None else self.place.run.give_up
        self.slow_after = min(BLAST_SLOW_JOB, self.give_up / 2)
        if self.place is not None:
            self.place.take()

    def set_slow(self) -> None:
        self.slow = True
        if self.place is not None:
            self.place.free()


def run_blast_pool(
    fn: Callable, calls: list[tuple], n_threads: int, give_up: float | None = None
) -> list:
    """``[fn(*args) for args in calls]`` as one parallel BLAST run: each call is
    a batch whose jobs are scheduled by a shared ``_BlastRun`` (slow jobs make
    way for the next batches), on up to *n_threads* + BLAST_SLOW_THREADS
    threads (see ``stop.run_threads``). A job still waiting at NCBI after
    *give_up* seconds (default BLAST_GIVE_UP) is given up (see
    ``BlastGaveUp``)."""
    run = _BlastRun(
        len(calls), n_threads, BLAST_GIVE_UP if give_up is None else give_up
    )
    spare = run.pool_size - min(run.n_threads, len(calls))
    logger.debug(
        f"Starting {len(calls)} BLAST {pluralize(len(calls), 'batch')} "
        f"on {run.n_threads} {pluralize(run.n_threads, 'thread')}, "
        f"+{spare} while jobs are slow"
    )

    def batch(position, *args):
        place = _Place(run, position)
        # Taken before the batch starts, so it reads the circuit and the clock
        # once it can run.
        place.take()
        token = _place.set(place)
        try:
            return fn(*args)
        finally:
            _place.reset(token)
            place.free()

    calls = [(position, *args) for position, args in enumerate(calls)]
    return stop.run_threads(batch, calls, run.pool_size)


# Waits before each retry of a failed BLAST job. NCBI asks clients to poll a
# job at most once a minute, so retries are spaced in tens of seconds.
BLAST_RETRY_DELAYS = (30.0, 60.0)

# After this many jobs in a row have failed all their retries, the remaining
# jobs of the run get a single attempt (see BlastCircuit).
BLAST_CIRCUIT_THRESHOLD = 3


# Messages in XML2 query files (see read_xml2_archive).
_XML2_MESSAGE = re.compile(rb"<message>(.*?)</message>", re.DOTALL)
_XML2_QUERY_TITLE = re.compile(rb"<query-title>(.*?)</query-title>", re.DOTALL)
_NO_HITS_MESSAGE = b"No hits found"
# NCBI gives each search a fixed CPU budget; a search that exceeds it stops
# without the hits of its queries.
_CPU_LIMIT_MESSAGE = "CPU usage limit"


def _is_cpu_limit(text: str) -> bool:
    """Whether an NCBI reply or message says the search hit its CPU budget."""
    return _CPU_LIMIT_MESSAGE in text


class BlastGaveUp(ValueError):
    """A job still waiting at NCBI after its run's give-up time (BLAST_GIVE_UP
    by default, see ``run_blast_pool``). In practice such a job rarely ends,
    while the same search resubmitted usually takes a minute or two, so
    ``submit_blast_batch`` resubmits it once and raises this only if the new
    job is given up too. Its sequences are then reported failed (not cached,
    so the next run queries them again) without a retry or a split, which
    would only add searches to NCBI's queue, and it says nothing about NCBI
    being down."""


class BlastCpuLimit(ValueError):
    """A search NCBI stopped at its CPU budget. Resending it the same fails the
    same way, so it is split into smaller searches instead of retried, and it
    says nothing about NCBI being down."""


def read_xml2_archive(data: bytes) -> list:
    """Parse the zip archive NCBI returns for an XML2 BLAST job.

    The archive holds one XML2 file per query (``<RID>_<n>.xml``) and an index
    file (``<RID>.xml``) that only lists them, so the query files are parsed in
    query order and the index is skipped.

    A query can carry a ``<message>`` element, which the Bio.Blast XML2 parser
    cannot read (no handler for it, up to Biopython 1.88): "No hits found", or
    a per-query BLAST warning such as "Could not calculate ungapped
    Karlin-Altschul parameters" on a low-complexity sequence. Messages are
    removed before parsing, so the query comes out with whatever hits it has;
    any message other than "No hits found" is logged as a warning. The one
    exception is NCBI's "CPU usage limit was exceeded": the search stopped
    early, so the archive raises ``BlastCpuLimit`` and the batch is split.

    Returns:
        Bio.Blast Record objects, one per query.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        numbered = {}
        for name in archive.namelist():
            match = re.search(r"_(\d+)\.xml$", name)
            if match:
                numbered[int(match.group(1))] = name
        records = []
        for n in sorted(numbered):
            xml = archive.read(numbered[n])
            for message in _XML2_MESSAGE.findall(xml):
                if _is_cpu_limit(message.decode(errors="replace")):
                    raise BlastCpuLimit(
                        f"NCBI stopped the search: {message.decode(errors='replace')}"
                    )
                if message.strip() != _NO_HITS_MESSAGE:
                    title = _XML2_QUERY_TITLE.search(xml)
                    query = title.group(1).decode(errors="replace") if title else n
                    logger.warning(
                        f"BLAST message for {query}: {message.decode(errors='replace')}"
                    )
            xml = _XML2_MESSAGE.sub(b"", xml)
            records.extend(Blast.parse(io.BytesIO(xml)))
        return records


def _read_job_result(data: bytes, multi_fasta: str) -> list:
    """Parse a job's XML2 archive. Raises if it does not hold one record per
    query, so an incomplete archive is retried instead of passing for queries
    without hits."""
    records = read_xml2_archive(data)
    n_queries = sum(line.startswith(">") for line in multi_fasta.splitlines())
    if len(records) != n_queries:
        raise ValueError(f"BLAST result holds {len(records)} of {n_queries} queries")
    return records


def _run_qblast_once(
    multi_fasta: str,
    evalue: float,
    hitlist_size: int,
    entrez_query: str | None,
) -> list:
    """Submit one BLASTn job to NCBI through Biopython's qblast and parse its
    XML2 result (one attempt). The fallback client (BLAST_CLIENT=qblast). It
    polls on its own, so a slow job neither makes way nor is given up."""
    with Blast.qblast(
        program="blastn",
        database=BLAST_DATABASE,
        sequence=multi_fasta,
        expect=evalue,
        hitlist_size=hitlist_size,
        entrez_query=entrez_query,
        format_type="XML2",
    ) as stream:
        return _read_job_result(stream.read(), multi_fasta)


# Same job, resubmitted as a whole after each delay in BLAST_RETRY_DELAYS: a
# failed submission, poll or truncated response is retried.
_run_qblast = retry_call(
    delays=BLAST_RETRY_DELAYS,
    no_retry=(BlastCpuLimit,),
    describe=lambda: _job_label.get(),
)(_run_qblast_once)


### BLAST URL API CLIENT ###

# TRIDENT's own client for NCBI's Common URL API, the one qblast uses: submit a
# job (CMD=Put, which returns its RID), then poll it for its XML2 result
# (CMD=Get). NCBI answers a poll with the result's zip archive once the job is
# done, and before that with a page holding the job's status (Status=WAITING,
# FAILED or UNKNOWN, as NCBI's sample client web_blast.pl reads it). Unlike
# qblast, which takes any such page for a running job, it reads that status,
# so a failed or expired job raises instead of being polled forever; and it
# spaces the requests of all threads as NCBI asks. It sends the same search
# parameters as qblast, so both give the same hits; qblast stays available as
# a fallback (BLAST_CLIENT=qblast) in case NCBI changes the API.
# https://blast.ncbi.nlm.nih.gov/doc/blast-help/urlapi.html
# https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html
NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# NCBI's guidelines: at most one request every 10 s, and each job polled at
# most once a minute. The first poll comes after NCBI's estimate of the job's
# time (RTOE, or BLAST_FIRST_POLL without one), as in NCBI's sample client:
# small jobs are often done within it. Most jobs take 1 to 3 min, but one can
# wait in NCBI's queue for hours: still waiting after BLAST_SLOW_JOB, it is
# polled every BLAST_SLOW_POLL_INTERVAL and makes way for the next batches (see
# _BlastRun), and after BLAST_GIVE_UP (the default, set per run) it is given up
# (see BlastGaveUp). A job NCBI has lost ends up UNKNOWN once its RID expires,
# and raises then.
BLAST_REQUEST_INTERVAL = 10.0
BLAST_FIRST_POLL = 30.0
BLAST_POLL_INTERVAL = 60.0
BLAST_SLOW_JOB = 300.0
BLAST_SLOW_POLL_INTERVAL = 120.0
BLAST_GIVE_UP = 600.0
# Failed polls in a row (request error, or neither a result nor a status)
# tolerated before the job is given up, so a transient error does not cost a
# resubmission of a job that may be almost done.
BLAST_POLL_FAILURES = 3
# (connect, read) timeouts of one request; a large result can take a while.
BLAST_TIMEOUT = (30.0, 300.0)

_RID = re.compile(r"^\s*RID = (\S+)", re.MULTILINE)
_RTOE = re.compile(r"^\s*RTOE = (\d+)", re.MULTILINE)
_STATUS = re.compile(r"QBlastInfoBegin\s+Status=(\w+)")
_NCBI_MESSAGE = re.compile(r"Message ID#[^<\n]*")

_request_lock = threading.Lock()
_next_request = 0.0


def _blast_request(**params) -> requests.Response:
    """POST one URL API request, with TRIDENT's tool name and the contact
    email, at least BLAST_REQUEST_INTERVAL after the previous request of any
    thread. Raises on an HTTP error, or ``StepStopped`` once the step is
    stopped."""
    global _next_request
    # Wait (outside the lock) until no request of any thread was sent in the
    # last BLAST_REQUEST_INTERVAL. Nothing is reserved ahead, so a stopped
    # request leaves no slot that would delay the next run (or other users).
    while True:
        stop.check_stop()
        with _request_lock:
            now = time.monotonic()
            if now >= _next_request:
                _next_request = now + BLAST_REQUEST_INTERVAL
                break
            wait = _next_request - now
        stop.wait(wait)
    params["tool"] = "trident"
    if Blast.email:
        params["email"] = Blast.email
    response = requests.post(
        NCBI_BLAST_URL,
        data=params,
        # Blast.email rather than the config: jobs run in threads that cannot
        # read the email set in Settings (session state).
        headers={"User-Agent": config.user_agent(Blast.email)},
        timeout=BLAST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def _submit_blast_job(
    multi_fasta: str, evalue: float, hitlist_size: int, entrez_query: str | None
) -> tuple[str, int | None]:
    """Submit a BLASTn search of BLAST_DATABASE; return its RID and RTOE
    (seconds)."""
    params = {
        "CMD": "Put",
        "PROGRAM": "blastn",
        "DATABASE": BLAST_DATABASE,
        "QUERY": multi_fasta,
        "EXPECT": evalue,
        "HITLIST_SIZE": hitlist_size,
    }
    if entrez_query:
        params["ENTREZ_QUERY"] = entrez_query
    page = _blast_request(**params).text
    match = _RID.search(page)
    if not match:
        message = _NCBI_MESSAGE.search(page)
        raise ValueError(
            f"NCBI refused the BLAST job: {message.group(0).strip()}"
            if message
            else "NCBI returned no RID for the BLAST job"
        )
    rtoe = _RTOE.search(page)
    return match.group(1), int(rtoe.group(1)) if rtoe else None


def _poll_blast_job(rid: str, rtoe: int | None, job: _Job) -> bytes:
    """Poll a job until it is done and return its XML2 result (zip archive).

    The first poll comes after RTOE (or BLAST_FIRST_POLL). A failed poll
    (request error, or a reply that is neither the archive nor a status page)
    is tried again, up to BLAST_POLL_FAILURES in a row. A job still waiting
    after ``job.slow_after`` is set slow, and after ``job.give_up`` it
    raises ``BlastGaveUp`` (both counted from the first poll, so a long NCBI
    estimate does not count). Raises if NCBI reports the job failed or unknown
    (expired RID).
    """
    stop.wait(rtoe or BLAST_FIRST_POLL)
    start = time.monotonic()
    failures = 0
    status_link = f"{NCBI_BLAST_URL}?CMD=Get&RID={rid}&FORMAT_OBJECT=SearchInfo"
    while True:
        try:
            # ALIGNMENTS and DESCRIPTIONS as qblast sends them (its defaults).
            reply = _blast_request(
                CMD="Get", RID=rid, FORMAT_TYPE="XML2", ALIGNMENTS=500, DESCRIPTIONS=500
            )
            if reply.content.startswith(b"PK"):
                return reply.content
            match = _STATUS.search(reply.text)
            if not match or match.group(1) == "READY":
                raise ValueError("neither a result nor a job status in NCBI's reply")
        except (requests.RequestException, ValueError) as e:
            failures += 1
            if failures >= BLAST_POLL_FAILURES:
                raise
            logger.debug(f"Poll of BLAST job {rid} failed ({e}); polling again")
            stop.wait(BLAST_SLOW_POLL_INTERVAL if job.slow else BLAST_POLL_INTERVAL)
            continue
        failures = 0
        if match.group(1) != "WAITING":
            ended = f"BLAST job {rid} ended with status {match.group(1)}"
            if _is_cpu_limit(reply.text):
                raise BlastCpuLimit(f"{ended}: CPU usage limit was exceeded")
            raise ValueError(ended)
        waited = time.monotonic() - start
        if waited > job.give_up:
            raise BlastGaveUp(
                f"RID {rid} still waiting at NCBI after {duration(waited)} "
                f"(status: {status_link})"
            )
        if waited > job.slow_after and not job.slow:
            job.set_slow()
            logger.info(
                f"{_job_label.get()} (RID {rid}) still waiting at NCBI after "
                f"{duration(waited)}; now polled every "
                f"{duration(BLAST_SLOW_POLL_INTERVAL)} (status: {status_link})"
            )
        stop.wait(BLAST_SLOW_POLL_INTERVAL if job.slow else BLAST_POLL_INTERVAL)


def _run_blast_job_once(
    multi_fasta: str,
    evalue: float,
    hitlist_size: int,
    entrez_query: str | None,
) -> list:
    """Submit one BLASTn job to NCBI and parse its XML2 result (one attempt)."""
    job = _Job()
    rid, rtoe = _submit_blast_job(multi_fasta, evalue, hitlist_size, entrez_query)
    # RTOE is not shown: NCBI gives the same 30 s for jobs of a minute or an hour.
    logger.info(f"{_job_label.get()} submitted: RID {rid}")
    return _read_job_result(_poll_blast_job(rid, rtoe, job), multi_fasta)


# Same job, resubmitted as a whole after each delay in BLAST_RETRY_DELAYS, as
# with qblast: a failed submission or poll, or a bad result, is retried (a job
# given up or too heavy for NCBI is not).
_run_blast_job = retry_call(
    delays=BLAST_RETRY_DELAYS,
    no_retry=(BlastCpuLimit, BlastGaveUp),
    describe=lambda: _job_label.get(),
)(_run_blast_job_once)


def _blast_runner(retry: bool) -> Callable:
    """The job function of the BLAST client: TRIDENT's URL API client, or
    Biopython's qblast with BLAST_CLIENT=qblast."""
    if config.get("BLAST_CLIENT") == "qblast":
        return _run_qblast if retry else _run_qblast_once
    return _run_blast_job if retry else _run_blast_job_once


class BlastCircuit:
    """Stops retrying BLAST jobs once NCBI looks down, for the rest of a run.

    Retries cost about 90 s per failed job, so during an NCBI outage a run of
    many jobs would block for hours before reporting anything. After
    ``threshold`` jobs in a row have failed all their retries, the circuit is
    open: later jobs get a single attempt and failed batches are not split.
    A success closes it again. Thread-safe; create one per run and pass it to
    every ``submit_blast_batch`` / ``blast_batch`` / ``run_blast_batches`` call
    of that run.
    """

    def __init__(self, threshold: int = BLAST_CIRCUIT_THRESHOLD):
        self.threshold = threshold
        self._failures = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._failures >= self.threshold

    def record(self, success: bool) -> None:
        with self._lock:
            if success:
                self._failures = 0
            else:
                self._failures += 1
                if self._failures == self.threshold:
                    logger.warning(
                        f"{self.threshold} BLAST jobs in a row failed after "
                        "retries; NCBI may be down. Remaining jobs of this run "
                        "get a single attempt."
                    )


@require_ncbi_auth
def submit_blast_batch(
    batch: list[SeqRecord],
    entrez_query: str | None = None,
    evalue: float = 1e-10,
    hitlist_size: int = 500,
    batch_id: int | str | None = None,
    retry: bool = True,
    circuit: BlastCircuit | None = None,
) -> pd.DataFrame | None:
    """Submit a BLASTn query for a single batch of sequences.

    Builds a multi-FASTA string from the batch, submits it to NCBI BLAST
    through the configured client (``_blast_runner``), parses the XML2
    response, and returns a raw results DataFrame.
    With ``retry``, a failed job is resubmitted after each delay in
    ``BLAST_RETRY_DELAYS``; if it still fails, returns None so the caller can
    handle partial results. A job given up while waiting at NCBI is
    resubmitted once, at once (see ``BlastGaveUp``).

    Args:
        batch: List of SeqRecord objects to query.
        entrez_query: Optional filter in Entrez search syntax (BLAST's
            ENTREZ_QUERY parameter, applied by NCBI to the searched database),
            e.g. '"Gadus morhua"[orgn]'.
        evalue: E-value threshold for reporting hits.
        hitlist_size: Maximum number of hits to return per query sequence.
        batch_id: Identifier used for logging only.
        retry: Retry a failed job (default). False makes a single attempt,
            used when splitting a batch that already failed its retries.
        circuit: Optional run-wide ``BlastCircuit``. While it is open, the job
            gets a single attempt; its outcome is recorded when retries apply.

    Returns:
        DataFrame of raw BLAST hits, or None if the query failed.

    Raises:
        BlastCpuLimit: NCBI stopped the search at its CPU budget; the caller
            splits the batch (it is not recorded on ``circuit``).
        BlastGaveUp: the job, resubmitted once, was given up again while
            waiting at NCBI; the caller reports its sequences failed (not
            recorded either).
    """
    if circuit is not None and circuit.is_open:
        retry = False
    label = f"Batch {batch_id}" if batch_id is not None else "BLAST job"
    token = _job_label.set(label)
    start = time.monotonic()
    try:
        stop.check_stop()  # also for the qblast client, which has no stop
        logger.debug(f"{label}: {len(batch)} sequences")

        # BLAST is sent the bases (see clean_sequence); the hits keep the
        # sequence as given.
        multi_fasta = records_to_fasta_string(
            [blast_record(r.id, clean_sequence(str(r.seq))) for r in batch]
        )
        runner = _blast_runner(retry)
        try:
            records = runner(multi_fasta, evalue, hitlist_size, entrez_query)
        except BlastGaveUp as e:
            logger.warning(f"{label} given up: {e}; resubmitting it")
            try:
                records = _blast_runner(retry=False)(
                    multi_fasta, evalue, hitlist_size, entrez_query
                )
            except (BlastCpuLimit, BlastGaveUp, stop.StepStopped):
                raise
            except Exception as resubmit_error:
                # Failed as a given-up job: no retry, no split, and nothing
                # said about NCBI being down.
                raise BlastGaveUp(
                    f"its resubmission failed ({resubmit_error})"
                ) from resubmit_error
        df = parse_blast_records(records)

        logger.info(
            f"{label} done after {duration(time.monotonic() - start)}: {len(df)} hits"
        )
        result = df

    except BlastCpuLimit as e:
        logger.warning(f"{label} too heavy for NCBI: {e}")
        raise
    except BlastGaveUp as e:
        logger.warning(f"{label} given up again: {e}. Run the step again to retry it.")
        raise
    except Exception as e:
        logger.error(f"{label} failed: {e}")
        result = None
    finally:
        _job_label.reset(token)

    # A failure counts once retries were tried; any success closes the circuit.
    # A batch skipped by a stop says nothing about NCBI.
    if (
        circuit is not None
        and (retry or result is not None)
        and not stop.stop_requested()
    ):
        circuit.record(result is not None)
    return result


def _split_batch(
    batch: list[SeqRecord],
    batch_id: int | str | None,
    circuit: BlastCircuit | None = None,
    too_heavy: bool = False,
    **blast_kwargs,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Split a batch that failed into halves, each with a single attempt, down
    to single sequences: a half that succeeds is kept, a half that fails is
    split again.

    Two causes, told apart by NCBI's reply:

    - The batch failed all its retries (*too_heavy* False): likely one of its
      sequences. If both halves fail too, the problem is likely on NCBI's side
      (overload, outage), so splitting stops and the whole batch is reported
      failed. This bounds the extra jobs: about 2 per halving level for one
      bad sequence, and 2 in total when NCBI is down. Known limit: two bad
      sequences in different halves also make both halves fail; a lower batch
      size narrows it down on the next run.
    - NCBI stopped the search at its CPU budget (*too_heavy* True, see
      ``BlastCpuLimit``): half the work can still be too much, so both halves
      failing that way is expected and each is halved again.

    A half that succeeds shows NCBI is up, so it is recorded as a success on
    ``circuit``: a batch failed by a bad sequence or by its size must not count
    toward the "NCBI is down" threshold. Splitting stops once the step is
    stopped, or once the circuit is open unless the batch was too heavy (a
    CPU-limit reply shows NCBI is answering). The halves of a too-heavy batch
    keep their retries, since nothing says they would fail. Once a part is
    given up while waiting at NCBI, nothing more is sent: more searches would
    only wait in the same queue, so every part not done yet is failed.

    Returns:
        Tuple of (result DataFrames of the parts that succeeded, ids of the
        sequences still failed).
    """
    done = []  # (result DataFrame, part) of the parts that succeeded
    try:
        _split_into(batch, batch_id, circuit, too_heavy, blast_kwargs, done)
    except BlastGaveUp:
        pass  # nothing more is sent: the parts not done fail below
    ok = {rec.id for _, part in done for rec in part}
    return [df for df, _ in done], [rec.id for rec in batch if rec.id not in ok]


def _split_into(batch, batch_id, circuit, too_heavy, blast_kwargs, done):
    """``_split_batch`` on *batch*: adds (result, part) to *done* for each part
    that succeeds. A give-up raises ``BlastGaveUp`` up to ``_split_batch``."""
    circuit_open = circuit is not None and circuit.is_open and not too_heavy
    if len(batch) == 1 or stop.stop_requested() or circuit_open:
        return

    mid = len(batch) // 2
    halves = [batch[:mid], batch[mid:]]
    reason = "too heavy for NCBI" if too_heavy else "failed after retries"
    logger.warning(
        f"Batch {batch_id} {reason}; splitting it into "
        f"{len(halves[0])} + {len(halves[1])} sequences"
    )
    to_split = []  # (half, whether it was too heavy)
    for half in halves:
        try:
            res = submit_blast_batch(
                half,
                batch_id=batch_id,
                retry=too_heavy,
                circuit=circuit,
                **blast_kwargs,
            )
        except BlastCpuLimit:
            to_split.append((half, True))
        else:
            if res is not None:
                done.append((res, half))
            else:
                to_split.append((half, False))
    if not too_heavy and len(to_split) == 2 and not any(h for _, h in to_split):
        return  # both halves failed: likely NCBI's side

    for half, half_heavy in to_split:
        _split_into(half, batch_id, circuit, half_heavy, blast_kwargs, done)


def blast_batch(
    batch: list[SeqRecord],
    batch_id: int | str,
    circuit: BlastCircuit,
    progress_handler: dict | object | None = None,
    **blast_kwargs,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Run one batch as a worker of a parallel run (``run_blast_pool``): submit
    it with retries, split it if it still fails or is too heavy for NCBI (see
    ``_split_batch``), then report its sequences to *progress_handler*. A batch
    given up twice while waiting at NCBI (see ``BlastGaveUp``) fails as it is.

    *blast_kwargs* are those of ``submit_blast_batch`` (evalue, hitlist_size,
    entrez_query), so callers running batches with different queries in one
    pool give each its own.

    Returns:
        Tuple of (result DataFrames, ids of the sequences whose query errored).
    """
    # Read before submitting: a batch submitted while the circuit is open got a
    # single attempt, so it is not split even if another thread closes the
    # circuit in the meantime.
    retried = not circuit.is_open
    try:
        result = submit_blast_batch(
            batch, batch_id=batch_id, circuit=circuit, **blast_kwargs
        )
    except BlastCpuLimit:
        outcome = _split_batch(batch, batch_id, circuit, too_heavy=True, **blast_kwargs)
    except BlastGaveUp:
        outcome = [], [rec.id for rec in batch]
    else:
        if result is not None:
            outcome = [result], []
        elif not retried:
            outcome = [], [rec.id for rec in batch]
        else:  # stops at once if the step was stopped or the circuit opened
            outcome = _split_batch(batch, batch_id, circuit, **blast_kwargs)
    notify_progress(progress_handler, len(batch))
    return outcome


def run_blast_batches(
    batches: list[list[SeqRecord]],
    n_threads: int = 10,
    evalue: float = 1e-10,
    hitlist_size: int = 500,
    entrez_query: str | None = None,
    progress_handler: dict | object | None = None,
    circuit: BlastCircuit | None = None,
    give_up: float | None = None,
    on_batch: Callable[[int, list[pd.DataFrame], list[str]], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Execute multiple NCBI BLAST batches in parallel.

    Runs up to n_threads batches concurrently, and a few more while jobs are
    slow at NCBI (see ``run_blast_pool``). Progress is reported per batch via
    progress_handler. Each worker runs in a copy of the caller's context, so
    log context set by the caller (e.g. the UI log sink) also applies to the
    workers' logs.

    Args:
        batches: List of sequence batches, each a list of SeqRecord objects.
        n_threads: Batches run at once at normal pace. Capped at len(batches).
        evalue: E-value significance threshold for BLAST hits.
        hitlist_size: Maximum number of hits per query sequence.
        entrez_query: Optional filter in Entrez search syntax, passed to every
            job (see submit_blast_batch).
        progress_handler: Optional progress tracker. Accepts:
            - dict with a 'current' key (incremented by batch size) for
              Streamlit or notebook use.
            - Any object with an .update(n) method (e.g. tqdm) for notebooks.
            - None for silent execution.
        circuit: Optional ``BlastCircuit`` shared by the whole run. A new one
            is created for this call when none is given.
        give_up: Seconds a job may wait at NCBI before it is resubmitted
            once, then counted as failed (default BLAST_GIVE_UP, see
            ``BlastGaveUp``).
        on_batch: Called in the worker thread when a batch is done, with its
            index in *batches*, its result DataFrames and the ids of its
            sequences whose query errored. The results then go to on_batch
            only: the returned DataFrame is empty.

    Returns:
        Tuple of (concatenated DataFrame of successful batch results,
        failed_ids). ``failed_ids`` lists the ids of sequences whose query
        errored (distinct from sequences that simply returned no hits), so the
        caller can retry them instead of caching them as empty. A batch that
        fails its retries, or that NCBI stops at its CPU limit, is split by
        ``_split_batch`` (unless the circuit is open), so only the sequences
        that still fail are listed. A batch given up twice while waiting at
        NCBI (see ``BlastGaveUp``) lists all its sequences.
    """
    if not batches:
        return pd.DataFrame(columns=BLAST_RAW_COLUMNS), []

    if circuit is None:
        circuit = BlastCircuit()
    blast_kwargs = {
        "evalue": evalue,
        "hitlist_size": hitlist_size,
        "entrez_query": entrez_query,
    }

    def worker(batch, i) -> tuple[list[pd.DataFrame], list[str]]:
        label = f"{i + 1}/{len(batches)}"
        dfs, failed_ids = blast_batch(
            batch, label, circuit, progress_handler, **blast_kwargs
        )
        if on_batch is None:
            return dfs, failed_ids
        on_batch(i, dfs, failed_ids)
        return [], failed_ids

    outcomes = run_blast_pool(
        worker, [(batch, i) for i, batch in enumerate(batches)], n_threads, give_up
    )

    successful = [df for dfs, _ in outcomes for df in dfs]
    # Ids of sequences whose query still errored after retries and splitting,
    # so the caller can retry them rather than cache them as empty (a failed
    # query is not a "no hits" result).
    failed_ids = [rec_id for _, ids in outcomes for rec_id in ids]
    if not successful:
        return pd.DataFrame(columns=BLAST_RAW_COLUMNS), failed_ids
    return pd.concat(successful, ignore_index=True), failed_ids


### RESULT PROCESSING ###

# Output columns of process_blast_results, also present when there are no hits.
BLAST_HIT_COLUMNS = [
    "seq_id",
    "dna_sequence",
    "genus",
    "specificEpithet",
    "scientificName",
    "ncbi_taxid",
    "query_cover",
    "identity_percentage",
    "align_length",
    "identities",
    "gaps",
    "query_start",
    "query_end",
    "hit_def",
    "hit_url",
]


def process_blast_results(
    df: pd.DataFrame,
    fasta_sequences: list[SeqRecord] | None = None,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Compute derived metrics and standardize taxonomy for raw BLAST results.

    Adds identity_percentage, query_cover, scientificName, genus,
    specificEpithet, hit_url, and dna_sequence columns. scientificName is the
    binomial of the hit's NCBI taxonomy name (sciname); the record title
    (hit_def) is kept as is, for display only.

    Args:
        df: Raw BLAST DataFrame from parse_blast_records().
        fasta_sequences: Optional list of SeqRecord objects used to attach
            the original DNA sequence to each hit row.
        round_digits: Decimal places for identity_percentage and query_cover.

    Returns:
        Processed DataFrame with standardized columns, sorted by descending
        identity within each seq_id.
    """
    # No hits (e.g. every batch failed): nothing to process.
    if df.empty:
        return pd.DataFrame(columns=BLAST_HIT_COLUMNS)

    df["identity_percentage"] = ((df["identities"] / df["align_length"]) * 100).round(
        round_digits
    )
    df["query_cover"] = (
        ((df["query_end"] - df["query_start"] + 1) / df["query_length"]) * 100
    ).round(round_digits)

    # A hit without a taxonomy name gets no name (NaN), not an empty one.
    df["scientificName"] = df["sciname"].map(to_binomial).replace("", pd.NA)

    # First token of the name. Vectorised and tolerant of a missing name: genus
    # becomes NaN rather than raising IndexError.
    df["genus"] = df["scientificName"].str.split().str[0]
    df["specificEpithet"] = df.apply(
        lambda row: extract_specific_epithet(row["scientificName"], row["genus"]),
        axis=1,
    )

    df["hit_url"] = df["hit_id"].apply(
        lambda x: (
            f"https://www.ncbi.nlm.nih.gov/nucleotide/{x.split('|')[-2]}"
            if "|" in x
            else None
        )
    )

    df = (
        df.groupby("seq_id", sort=False)
        .apply(
            lambda x: x.sort_values(by="identity_percentage", ascending=False),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    if fasta_sequences:
        seq_dict = {str(sequence.id): str(sequence.seq) for sequence in fasta_sequences}
        df["dna_sequence"] = df["seq_id"].map(seq_dict)
    else:
        df["dna_sequence"] = None

    return df[BLAST_HIT_COLUMNS]


### FILTERING ###


def compute_barcoding_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Add an identity_drop column: how much identity falls to the next-ranked hit.

    Within each seq_id group (assumed sorted by descending identity),
    identity_drop is strictly positive when the next hit has lower identity. The last
    row of each group gets NaN (no next hit to compare against).

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.

    Returns:
        Copy of df with an additional identity_drop column.
    """
    df = df.copy()
    df["identity_drop"] = df.groupby("seq_id")["identity_percentage"].diff(-1)
    return df


def filter_by_barcoding_gap(
    df: pd.DataFrame,
    gap_size: float = NCBI_GAP_SIZE,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split sequences into those with a detectable barcoding gap and those without.

    For each seq_id, searches for the first drop ≥ gap_size in identity
    percentage among hits with identity ≥ gap_min_top. Sequences where such a
    gap is found are kept only above the gap; the rest are returned unfiltered
    for downstream similarity filtering.

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.
        gap_size: Minimum identity drop (percentage points) to qualify as a gap.
        gap_min_top: The top of the gap must be at or above this identity value.

    Returns:
        Tuple of:
        - gap_filtered_df: Rows for sequences where a gap was found (above-gap hits only).
        - no_gap_df: All rows for sequences where no gap was found.
        - query_filter_method: Dict mapping each seq_id to 'barcoding_gap' or 'similarity'.
    """
    queries_with_gaps = []
    queries_without_gaps = []
    query_filter_method = {}

    df = df.copy()
    df = compute_barcoding_gap(df)

    for seq_id, group in df.groupby("seq_id"):
        group = group.reset_index(drop=True)
        gap_rows = group[
            (group["identity_drop"] >= gap_size)
            & (group["identity_percentage"] >= gap_min_top)
        ]

        if not gap_rows.empty:
            first_gap_pos = gap_rows.index[0]
            queries_with_gaps.append(group.iloc[: first_gap_pos + 1])
            query_filter_method[seq_id] = "barcoding_gap"
        else:
            queries_without_gaps.append(group)
            query_filter_method[seq_id] = "similarity"

    gap_filtered_df = (
        pd.concat(queries_with_gaps).reset_index(drop=True)
        if queries_with_gaps
        else df.iloc[:0]
    )
    no_gap_df = (
        pd.concat(queries_without_gaps).reset_index(drop=True)
        if queries_without_gaps
        else df.iloc[:0]
    )

    return gap_filtered_df, no_gap_df, query_filter_method


def filter_by_similarity(
    df: pd.DataFrame, gap_size: float = NCBI_GAP_SIZE
) -> pd.DataFrame:
    """Keep only hits within gap_size percentage points of the top hit per sequence.

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.
        gap_size: Maximum allowed drop from the best hit's identity percentage.

    Returns:
        Filtered DataFrame containing only hits within the similarity range.
    """
    max_per_query = df.groupby("seq_id")["identity_percentage"].transform("max")
    filtered_df = df[df["identity_percentage"] >= (max_per_query - gap_size)]
    return filtered_df


def filter_blast_results(
    df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    gap_size: float = NCBI_GAP_SIZE,
    method: str = NCBI_METHOD,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> pd.DataFrame:
    """Filter BLAST results by query cover, then by barcoding gap or
    similarity range.

    First applies a query_cover threshold, then filters by either the barcoding
    gap method or the similarity range method. When method='barcoding_gap',
    sequences without a detectable gap fall back to the similarity range.

    Args:
        df: Processed BLAST DataFrame from process_blast_results().
        query_cover: Minimum query cover (%) of a kept hit.
        gap_size: Gap size threshold (percentage points) used by both methods.
        method: Either 'barcoding_gap' or 'similarity'.
        gap_min_top: Minimum identity for barcoding gap search. Only used when
            method='barcoding_gap'.

    Returns:
        Filtered DataFrame sorted by descending identity, then ascending species name.

    Raises:
        ValueError: If method is not 'barcoding_gap' or 'similarity'.
    """
    original_rows = len(df)
    df = df[df["query_cover"] >= query_cover]
    logger.debug(
        f"Query cover filter (≥{query_cover}%): {original_rows} → {len(df)} rows"
    )

    if method == "barcoding_gap":
        df_gap, df_nogap, _ = filter_by_barcoding_gap(
            df, gap_size=gap_size, gap_min_top=gap_min_top
        )
        logger.debug(
            f"Gap analysis: {len(df_gap)} rows with gaps, {len(df_nogap)} rows need similarity method"
        )

        df_gap["filter_method"] = "barcoding_gap"

        if not df_nogap.empty:
            df_nogap = filter_by_similarity(df_nogap, gap_size=gap_size)
            df_nogap["filter_method"] = "similarity"

        dfs_to_concat = [d for d in [df_gap, df_nogap] if not d.empty]
        df = (
            pd.concat(dfs_to_concat).reset_index(drop=True)
            if dfs_to_concat
            else df_gap.iloc[:0]
        )

        logger.debug(
            f"Final result: {len(df)} rows after barcoding gap + similarity filtering"
        )

    elif method == "similarity":
        df = filter_by_similarity(df, gap_size=gap_size)
        df["filter_method"] = "similarity"
        logger.debug(f"Similarity range: {len(df)} rows kept")

    else:
        raise ValueError(
            f"Unknown filtering method: {method!r}. Expected 'barcoding_gap' or 'similarity'."
        )

    df = df.sort_values(
        by=["identity_percentage", "scientificName"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


### GAP ANALYSIS ###


class FilterCut(NamedTuple):
    """Where the MOL filter cut one sequence's hits: a barcoding gap or a range."""

    top: float
    bottom: float
    label: str
    is_gap: bool


def mol_filter_cut(
    df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    gap_size: float = NCBI_GAP_SIZE,
    method: str = NCBI_METHOD,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> FilterCut | None:
    """Locate the cut the MOL filter makes in one sequence's hits, for display.

    Runs filter_blast_results on the hits with the MOL settings, so the Overview
    and the plots show the gap or range the filter used, not one found by
    another rule.

    Returns:
        For a barcoding gap, its top is the lowest kept hit and its bottom the
        next hit below that passed query cover; for the similarity range, the
        best hit that passed query cover and *gap_size* below it. None when no
        hit passes query cover.
    """
    covered = df.loc[df["query_cover"] >= query_cover, "identity_percentage"]
    if covered.empty:
        return None
    kept = filter_blast_results(
        df,
        query_cover=query_cover,
        gap_size=gap_size,
        method=method,
        gap_min_top=gap_min_top,
    )
    if (kept["filter_method"] == "barcoding_gap").any():
        top = kept["identity_percentage"].min()
        bottom = covered[covered < top].max()
        return FilterCut(top, bottom, f"Gap<br>{top - bottom:.2f}%", True)
    top = covered.max()
    return FilterCut(top, top - gap_size, f"Range<br>{gap_size}%", False)


def species_across_gap(
    df: pd.DataFrame, cut: FilterCut | None, query_cover: float = NCBI_QUERY_COVER
) -> dict[str, dict]:
    """List the species with hits on both sides of the MOL filter's barcoding gap.

    Such a species is kept above the gap but also matches below it, which
    makes the boundary ambiguous. Hits below query cover are left out, as the
    filter left them out.

    Args:
        df: One sequence's BLAST hits.
        cut: The filter's cut, from mol_filter_cut; nothing is listed unless
            it is a barcoding gap.
        query_cover: The MOL filter's minimum query cover.

    Returns:
        Dict mapping species name to {'before_gap': [...], 'after_gap': [...]}
        identity lists, highest first.
    """
    if cut is None or not cut.is_gap:
        return {}
    covered = df[df["query_cover"] >= query_cover]
    details = {}
    for species, hits in covered.groupby("scientificName"):
        identities = hits["identity_percentage"].sort_values(ascending=False)
        before = identities[identities >= cut.top].tolist()
        after = identities[identities < cut.top].tolist()
        if before and after:
            details[species] = {"before_gap": before, "after_gap": after}
    return details


def format_species_gap_details(species_details: dict[str, dict]) -> str:
    """Format the output of species_across_gap as a readable string.

    Args:
        species_details: Dict mapping species name to
            {'before_gap': [...], 'after_gap': [...]} identity lists, as
            returned by species_across_gap().

    Returns:
        Markdown-formatted string listing each problematic species with its
        identity percentages on each side of the gap.
    """
    lines = []
    for species, identities in species_details.items():
        before_str = ", ".join([f"{x:.2f}%" for x in identities["before_gap"]])
        after_str = ", ".join([f"{x:.2f}%" for x in identities["after_gap"]])
        lines.append(
            f"**{species}**\n  - Above the gap: {before_str}\n  - Below the gap: {after_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


FILTERED_OUT_COLOR = "darkgrey"


def _species_colors(filtered_species):
    """Map each species kept by the MOL filter to a palette color."""
    import plotly.express as px

    if not filtered_species:
        return {}
    n = len(filtered_species)
    if n <= 10:
        palette = px.colors.qualitative.Plotly
    elif n <= 24:
        palette = px.colors.qualitative.Light24
    else:
        palette = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
    return {
        sp: palette[i % len(palette)] for i, sp in enumerate(sorted(filtered_species))
    }


def _name_unnamed_hits(df):
    """Copy of the hits where a hit without a species name gets one to show."""
    return df.assign(scientificName=df["scientificName"].fillna("No species name"))


def _species_groups(df, filtered_species):
    """Split hits by species, kept species first, with their color and legend.

    Yields:
        (species, rows, kept, color, legend): *legend* holds the name,
        legendgroup and showlegend of the species' trace; every filtered-out
        species shares one "Filtered out" legend entry.
    """
    colors = _species_colors(filtered_species)
    groups = dict(tuple(df.groupby("scientificName", sort=False)))
    order = sorted(sp for sp in groups if sp in colors) + sorted(
        sp for sp in groups if sp not in colors
    )
    filtered_out_shown = False
    for species in order:
        kept = species in colors
        legend = dict(
            name=species if kept else "Filtered out",
            legendgroup=species if kept else "filtered_out",
            showlegend=kept or not filtered_out_shown,
        )
        yield (
            species,
            groups[species],
            kept,
            colors.get(species, FILTERED_OUT_COLOR),
            legend,
        )
        filtered_out_shown = filtered_out_shown or not kept


def _apply_plot_layout(fig, title, **extra):
    """Apply common layout settings to a plot figure."""
    layout = {
        "hovermode": "closest",
        "showlegend": True,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        "height": 500,
        **extra,
    }
    if title:
        layout["title"] = title
    fig.update_layout(**layout)


def _add_gap_bracket(fig, cut, x, cap, label_offset):
    """Draw the MOL filter's cut (from mol_filter_cut) as a red bracket at *x*.

    *cap* is the half-width of the bracket ends and *label_offset* the distance
    of the label to the left of the bracket, both in x-axis units. Nothing is
    drawn when *cut* is None.
    """
    if cut is None:
        return
    top, bottom, label, _ = cut
    style = dict(color="red", width=2)
    fig.add_shape(type="line", x0=x, y0=top, x1=x, y1=bottom, line=style)
    for y in (top, bottom):
        fig.add_shape(type="line", x0=x - cap, y0=y, x1=x + cap, y1=y, line=style)
    fig.add_annotation(
        x=x - label_offset,
        y=(top + bottom) / 2,
        text=label,
        showarrow=False,
        font=dict(color="red", size=10),
        xanchor="right",
        align="center",
    )


def plot_identity_percentage(
    df,
    cut=None,
    filtered_species=None,
    title="Identity Percentage Distribution",
    marker_size=10,
):
    """Plot one point per hit at its identity, highlighting filtered species.

    The hits of one identity level are lined up from x = 1, so the last point
    of a level sits at its number of hits. Shows the barcoding gap or the
    similarity range the MOL filter used.

    Args:
        df: DataFrame with 'scientificName' and 'identity_percentage' columns.
        cut: The MOL filter's cut, from mol_filter_cut (None: no bracket).
        filtered_species: Set/list of species names to highlight.
        title: Plot title (empty string to hide).
        marker_size: Size of scatter markers for filtered species (filtered-out
            species are drawn smaller).

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    df_plot = _name_unnamed_hits(df)
    df_plot["x_position"] = df_plot.groupby("identity_percentage").cumcount() + 1

    for species, sp_data, kept, color, legend in _species_groups(
        df_plot, filtered_species
    ):
        fig.add_trace(
            go.Scatter(
                x=sp_data["x_position"],
                y=sp_data["identity_percentage"],
                mode="markers",
                marker=dict(
                    size=marker_size if kept else max(marker_size - 4, 4),
                    color=color,
                    opacity=0.9 if kept else 0.6,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=f"<b>{species}</b><br>Identity: %{{y:.2f}}%<extra></extra>",
                **legend,
            )
        )

    _add_gap_bracket(fig, cut, x=0, cap=0.4, label_offset=0.7)

    _apply_plot_layout(fig, title, xaxis_title="Hits", yaxis_title="Identity (%)")
    return fig


HOVER_SPECIES_MAX = 10


def _hover_species_list(names):
    """List species with their hit counts, most hits first, for a bar hover.

    Capped at HOVER_SPECIES_MAX lines: a low-identity bin can hold over a
    hundred species, more than a hover box can show.
    """
    counts = names.value_counts()
    lines = [f"{name} ({n})" for name, n in counts.head(HOVER_SPECIES_MAX).items()]
    if len(counts) > HOVER_SPECIES_MAX:
        lines.append(f"+{len(counts) - HOVER_SPECIES_MAX} more species")
    return "<br>".join(lines)


def _bin_positions(levels, bin_width):
    """Map each identity level to its bin, in the order of *levels*.

    A bin starts at a level and takes the following levels less than
    *bin_width* away from it; every level maps to that first level.
    """
    positions, start = {}, None
    for level in levels:
        # The tolerance keeps a level exactly one bin width away out of the bin
        if start is None or abs(start - level) >= bin_width - 1e-9:
            start = level
        positions[level] = start
    return positions


def plot_identity_bars(
    df,
    cut=None,
    filtered_species=None,
    title="Hits per Identity Bin",
    bin_width=0.2,
):
    """Plot the number of hits per identity bin, stacked by species.

    Bins are built from the hits, on a true identity scale, and each bar sits
    at a real identity: the one of its bin nearest the barcoding gap, so the
    two bars around the gap sit at its ends (with a similarity range, the best
    one of the bin). A small bin only merges levels a fraction of a mismatch
    apart (98.87, 98.86 and 98.85 differ by alignment length); the hover lists
    the exact identities of each segment.

    Args:
        df: DataFrame with 'scientificName' and 'identity_percentage' columns.
        cut: The MOL filter's cut, from mol_filter_cut (None: no bracket).
        filtered_species: Set/list of species names to highlight.
        title: Plot title (empty string to hide).
        bin_width: Bin height in identity percent.

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    df = _name_unnamed_hits(df)
    levels = df["identity_percentage"].unique()
    if cut is not None and cut.is_gap:
        # Bins grow away from the gap: upwards above it, downwards below it.
        # Hits inside it (below query cover) less than a bin from one end join
        # that end's bar, the others get their own bins: no two bars overlap
        def near(level, end):
            return abs(level - end) < bin_width - 1e-9

        above = sorted(level for level in levels if level >= cut.top)
        below = sorted((level for level in levels if level <= cut.bottom), reverse=True)
        inside = [level for level in levels if cut.bottom < level < cut.top]
        positions = (
            _bin_positions(above, bin_width)
            | _bin_positions(below, bin_width)
            | _bin_positions(
                sorted(
                    (
                        level
                        for level in inside
                        if not near(level, cut.top) and not near(level, cut.bottom)
                    ),
                    reverse=True,
                ),
                bin_width,
            )
            | {level: cut.top for level in inside if near(level, cut.top)}
            | {
                level: cut.bottom
                for level in inside
                if near(level, cut.bottom) and not near(level, cut.top)
            }
        )
    else:
        positions = _bin_positions(sorted(levels, reverse=True), bin_width)
    # One grey segment per bin for every filtered-out species, named in its
    # hover: one outlined segment each would only add stripes
    kept = df["scientificName"].isin(filtered_species or [])
    binned = df.assign(
        bin_y=df["identity_percentage"].map(positions),
        species=df["scientificName"],
        scientificName=df["scientificName"].where(kept, "Filtered out"),
    )
    counts = (
        binned.groupby(["bin_y", "scientificName"])
        .agg(
            hits=("identity_percentage", "size"),
            identities=(
                "identity_percentage",
                lambda ids: ", ".join(
                    dict.fromkeys(f"{i:.2f}" for i in sorted(ids, reverse=True))
                ),
            ),
            species=("species", _hover_species_list),
        )
        .reset_index()
    )

    for species, sp_counts, sp_kept, color, legend in _species_groups(
        counts, filtered_species
    ):
        species_line = "" if sp_kept else "<br>%{customdata[1]}"
        fig.add_trace(
            go.Bar(
                x=sp_counts["hits"],
                y=sp_counts["bin_y"],
                width=bin_width * 0.9,
                orientation="h",
                marker=dict(color=color, line=dict(width=1, color="black")),
                customdata=sp_counts[["identities", "species"]].values,
                hovertemplate=(
                    f"<b>{species}</b><br>Identity: %{{customdata[0]}}%<br>"
                    f"Hits: %{{x}}{species_line}<extra></extra>"
                ),
                **legend,
            )
        )

    widest = binned.groupby("bin_y").size().max() if not df.empty else 1
    _add_gap_bracket(
        fig,
        cut,
        x=-0.02 * widest,
        cap=0.01 * widest,
        label_offset=0.02 * widest,
    )

    _apply_plot_layout(
        fig,
        title,
        barmode="stack",
        xaxis_title="Number of hits",
        yaxis_title="Identity (%)",
    )
    return fig
