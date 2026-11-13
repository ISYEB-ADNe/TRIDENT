"""Tests for reading NCBI BLAST XML2 results into hit rows (no network).

The fixtures are real XML2 archives as NCBI returns them: one query
(MUSA_Vert16S_00065) cut down to three hits, one of them titled "PREDICTED:",
and one query without hits.
"""

import io
import re
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


def _edited_archive(path: Path, edit) -> bytes:
    """The archive at *path* with its result file passed through *edit*."""
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["RID_1.xml"] = edit(files["RID_1.xml"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buf.getvalue()


def _no_hits_archive_with_message(message: bytes) -> bytes:
    """The no-hits archive with its "No hits found" replaced by *message*."""
    return _edited_archive(NO_HITS, lambda xml: xml.replace(b"No hits found", message))


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


def _second_alignment(xml: bytes) -> bytes:
    """*xml* with a second, shorter alignment (identical over 60 bases) added
    to the first hit: a lower score but a higher identity than the first."""
    start = xml.index(b"<Hsp>")
    hsp = xml[start : xml.index(b"</Hsp>", start) + len(b"</Hsp>")]
    piece = re.sub(rb"<num>1</num>", b"<num>2</num>", hsp)
    for tag, value in [
        (b"bit-score", b"111.3"),
        (b"score", b"120"),
        (b"identity", b"60"),
        (b"query-to", b"60"),
        (b"hit-to", b"2253"),
        (b"align-len", b"60"),
    ]:
        piece = re.sub(
            rb"<%s>[^<]*</%s>" % (tag, tag), b"<%s>%s</%s>" % (tag, value, tag), piece
        )
    for tag in (b"qseq", b"hseq", b"midline"):
        piece, n = re.subn(
            rb"<%s>([^<]{60})[^<]*</%s>" % (tag, tag),
            rb"<%s>\1</%s>" % (tag, tag),
            piece,
        )
        assert n == 1, tag
    return xml.replace(hsp, hsp + piece, 1)


def test_a_record_aligned_in_several_pieces_counts_once_by_its_best_alignment():
    """A GenBank record the query aligns to twice gives one row: the
    alignment with the highest score, with its own identity and cover."""
    records = read_xml2_archive(_edited_archive(ONE_QUERY, _second_alignment))
    assert sum(len(hit) for hit in records[0]) == 4  # the fixture's 3 + 1
    hits = process_blast_results(parse_blast_records(records))
    assert len(hits) == len(_hits(ONE_QUERY)) == hits["hit_url"].nunique()
    first = hits[hits["hit_def"] == records[0][0].target.description].iloc[0]
    assert first["identity_percentage"] == round(218 / 238 * 100, 2)
    assert first["query_cover"] == 100.0
