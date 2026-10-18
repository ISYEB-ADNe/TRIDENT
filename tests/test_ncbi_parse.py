"""Tests for reading NCBI BLAST XML2 results into hit rows (no network).

The fixtures are real XML2 archives as NCBI returns them: one query
(MUSA_Vert16S_00065) cut down to three hits, one of them titled "PREDICTED:",
and one query without hits.
"""

import io
import zipfile
from pathlib import Path

import pytest

from trident.clients.ncbi import (
    BLAST_HIT_COLUMNS,
    BlastCpuLimit,
    parse_blast_records,
    process_blast_results,
    read_xml2_archive,
)

FIXTURES = Path(__file__).parent / "fixtures"
ONE_QUERY = FIXTURES / "blast_xml2_one_query.zip"
NO_HITS = FIXTURES / "blast_xml2_no_hits.zip"


def _hits(archive: Path):
    return process_blast_results(
        parse_blast_records(read_xml2_archive(archive.read_bytes()))
    )


def test_hit_names_come_from_the_ncbi_taxonomy_not_the_title():
    """A hit titled "PREDICTED: Orcinus orca ..." is named Orcinus orca, with
    its taxid, so its genus can be expanded."""
    hits = _hits(ONE_QUERY)
    assert set(hits["seq_id"]) == {"MUSA_Vert16S_00065"}
    predicted = hits[hits["hit_def"].str.startswith("PREDICTED:")].iloc[0]
    assert predicted["scientificName"] == "Orcinus orca"
    assert predicted["genus"] == "Orcinus"
    assert predicted["ncbi_taxid"] == 9733


def test_query_without_hits_gives_no_rows():
    records = read_xml2_archive(NO_HITS.read_bytes())
    assert [r.query.description for r in records] == ["MUSA_Vert16S_00111"]
    assert parse_blast_records(records).empty


def _no_hits_archive_with_message(message: bytes) -> bytes:
    """The no-hits archive with its "No hits found" replaced by *message*."""
    with zipfile.ZipFile(NO_HITS) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["RID_1.xml"] = files["RID_1.xml"].replace(b"No hits found", message)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buf.getvalue()


def test_a_blast_warning_does_not_lose_the_query():
    """A per-query BLAST warning (here on a query without hits) is dropped
    before parsing: the query still comes out, it does not fail the job."""
    data = _no_hits_archive_with_message(
        b"Could not calculate ungapped Karlin-Altschul parameters"
    )
    records = read_xml2_archive(data)
    assert [r.query.description for r in records] == ["MUSA_Vert16S_00111"]


def test_a_search_stopped_by_the_cpu_limit_fails():
    """A search NCBI stopped at its CPU limit raises, so its batch is split
    rather than its queries read as having no hits."""
    data = _no_hits_archive_with_message(b"CPU usage limit was exceeded")
    with pytest.raises(BlastCpuLimit):
        read_xml2_archive(data)


def test_no_hits_keep_the_columns_of_hits():
    no_hits = process_blast_results(parse_blast_records([]))
    assert list(no_hits.columns) == list(_hits(ONE_QUERY).columns) == BLAST_HIT_COLUMNS
