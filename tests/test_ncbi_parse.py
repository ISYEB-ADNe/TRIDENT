"""Tests for parse_blast_records — flattening BLAST records to rows (no network)."""

from types import SimpleNamespace

import pandas as pd

from trident.clients.ncbi import parse_blast_records, process_blast_results


def _hsp(**over):
    base = dict(
        score=100,
        bits=99.0,
        expect=1e-50,
        align_length=200,
        identities=198,
        gaps=0,
        query_start=1,
        query_end=200,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _alignment(hit_def, hsps):
    return SimpleNamespace(
        hit_def=hit_def,
        hit_id="gi|1",
        accession="MT123456",
        hsps=hsps,
    )


def _record(query, alignments):
    return SimpleNamespace(query=query, query_length=200, alignments=alignments)


def test_flattens_one_row_per_hsp():
    rec = _record("s1", [_alignment("Gadus morhua", [_hsp(), _hsp(identities=150)])])
    df = parse_blast_records([rec])
    assert len(df) == 2  # two HSPs -> two rows
    assert set(df["seq_id"]) == {"s1"}
    assert df["hit_def"].iloc[0] == "Gadus morhua"
    assert df["identities"].tolist() == [198, 150]


def test_record_with_no_alignments_yields_no_rows():
    rec = _record("s1", [])
    df = parse_blast_records([rec])
    assert df.empty


def test_multiple_records():
    recs = [
        _record("s1", [_alignment("A", [_hsp()])]),
        _record("s2", [_alignment("B", [_hsp()])]),
    ]
    df = parse_blast_records(recs)
    assert set(df["seq_id"]) == {"s1", "s2"}
    assert "query_length" in df.columns


def _raw_row(hit_def, **over):
    base = dict(
        seq_id="s1",
        query_length=200,
        hit_def=hit_def,
        hit_id="gi|1|x",
        hit_accession="MT123456",
        score=100,
        bits=99.0,
        evalue=1e-50,
        align_length=200,
        identities=198,
        gaps=0,
        query_start=1,
        query_end=200,
    )
    base.update(over)
    return base


def test_process_blast_results_tolerates_empty_hit_def():
    # Two rows with an empty hit_def: the (empty) name is not unique, so no
    # Entrez re-resolution is attempted (no network), and genus extraction must
    # not raise on the empty name.
    df = pd.DataFrame([_raw_row(""), _raw_row("", query_start=5)])
    out = process_blast_results(df)
    assert out["scientificName"].tolist() == ["", ""]
    assert out["genus"].isna().all()
