"""Tests for BOLD record cleaning / marker filtering (network-free)."""

import pandas as pd

from trident.clients import bold
from trident.clients.bold import _clean_records, query_species


def _rec(**over):
    base = {
        "marker_code": "COI-5P",
        "inst": "Some Museum",
        "processid": "P1",
        "nuc": "ACGT",
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Mammalia",
        "order": "Artiodactyla",
        "family": "Balaenopteridae",
        "genus": "Balaenoptera",
        "species": "Balaenoptera physalus",
        "taxid": 2715,
    }
    base.update(over)
    return base


def test_keep_only_coi5p_drops_other_markers():
    records = [
        _rec(marker_code="COI-5P", processid="A"),
        _rec(marker_code="COI-3P", processid="B"),
        _rec(marker_code="16S", processid="C"),
    ]
    df = _clean_records(records, keep_only_COI5P=True)
    assert df["seq_id"].tolist() == ["A"]


def test_keep_all_markers_when_flag_off():
    records = [
        _rec(marker_code="COI-5P", processid="A"),
        _rec(marker_code="16S", processid="B"),
    ]
    df = _clean_records(records, keep_only_COI5P=False)
    assert set(df["seq_id"]) == {"A", "B"}


def test_ncbi_mined_excluded_by_default():
    records = [
        _rec(processid="A", inst="Some Museum"),
        _rec(processid="B", inst="Mined from GenBank, NCBI"),
    ]
    assert _clean_records(records)["seq_id"].tolist() == ["A"]
    both = _clean_records(records, keep_ncbi=True)
    assert set(both["seq_id"]) == {"A", "B"}


def test_field_mapping():
    df = _clean_records([_rec(processid="P9", taxid=42)])
    row = df.iloc[0]
    assert row["scientificName"] == "Balaenoptera physalus"
    assert row["specificEpithet"] == "physalus"
    assert row["dna_sequence"] == "ACGT"
    assert row["taxonID"] == "42"
    assert row["taxonID_db"] == "BOLD"
    assert row["seq_url"].endswith("/record/P9")


def test_empty_records_returns_empty_frame():
    assert _clean_records([]).empty
    assert isinstance(_clean_records([]), pd.DataFrame)


# --- query_species failure contract (None = error, [] = genuine empty) ---
# failure_sink in the pipeline relies on this distinction.


def _stub_pipeline(
    monkeypatch, *, query_id="qid", download="", preprocess_raises=False
):
    if preprocess_raises:

        def _pre(*a, **k):
            raise RuntimeError("403 Forbidden")
    else:

        def _pre(*a, **k):
            return {"successful_terms": [{"matched": "Gadus morhua"}]}

    monkeypatch.setattr(bold, "_preprocess_query", _pre)
    monkeypatch.setattr(bold, "_get_query_id", lambda *a, **k: query_id)
    monkeypatch.setattr(bold, "_download_results", lambda *a, **k: download)


def test_query_species_success_returns_records(monkeypatch):
    _stub_pipeline(monkeypatch, download='{"processid": "P1"}\n')
    assert query_species("Gadus morhua") == [{"processid": "P1"}]


def test_query_species_error_returns_none(monkeypatch):
    # An exception anywhere in the pipeline -> None (a failure, retried later).
    _stub_pipeline(monkeypatch, preprocess_raises=True)
    assert query_species("Gadus morhua") is None


def test_query_species_no_query_id_returns_none(monkeypatch):
    _stub_pipeline(monkeypatch, query_id=None)
    assert query_species("Gadus morhua") is None


def test_query_species_empty_download_returns_empty_list(monkeypatch):
    # A successful query with no records -> [] (a genuine empty, gets cached).
    _stub_pipeline(monkeypatch, download="")
    assert query_species("Gadus morhua") == []
