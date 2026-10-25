"""Tests for the BOLD client: record filters, the query's failure contract, and
the reason each queried species gets no sequence (network-free)."""

import json
from pathlib import Path

import pandas as pd

from trident.clients import bold
from trident.clients.bold import (
    _clean_records,
    get_records_from_species_list,
    outcomes_by_species,
    query_species,
    with_sequences,
)
from trident.pipelines.extra_pipeline import prepare_bold_other_names
from trident.pipelines.names import (
    bold_name_rows,
    build_name_table,
    gbif_name_rows,
    replace_source_rows,
)


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
    df, _ = _clean_records(records, keep_only_COI5P=True)
    assert df["seq_id"].tolist() == ["A"]


def test_keep_all_markers_when_flag_off():
    records = [
        _rec(marker_code="COI-5P", processid="A"),
        _rec(marker_code="16S", processid="B"),
    ]
    df, _ = _clean_records(records, keep_only_COI5P=False)
    assert set(df["seq_id"]) == {"A", "B"}


def test_ncbi_mined_excluded_by_default():
    records = [
        _rec(processid="A", inst="Some Museum"),
        _rec(processid="B", inst="Mined from GenBank, NCBI"),
    ]
    assert _clean_records(records)[0]["seq_id"].tolist() == ["A"]
    both, _ = _clean_records(records, keep_ncbi=True)
    assert set(both["seq_id"]) == {"A", "B"}


def test_field_mapping():
    df, _ = _clean_records([_rec(processid="P9", taxid=42)])
    row = df.iloc[0]
    assert row["scientificName"] == "Balaenoptera physalus"
    assert row["specificEpithet"] == "physalus"
    assert row["dna_sequence"] == "ACGT"
    assert row["taxonID"] == "42"
    assert row["taxonID_db"] == "BOLD"
    assert row["seq_url"].endswith("/record/P9")


def test_empty_records_returns_empty_frame():
    assert _clean_records([])[0].empty
    assert isinstance(_clean_records([])[0], pd.DataFrame)


def test_no_records_keep_the_columns_of_records():
    assert list(_clean_records([])[0].columns) == list(
        _clean_records([_rec()])[0].columns
    )


# --- query_species failure contract (None = error, else a genuine answer) ---
# failure_sink in the pipeline relies on this distinction.


def _stub_pipeline(
    monkeypatch, *, query_id="qid", download="", preprocess_raises=False
):
    if preprocess_raises:

        def _pre(*a, **k):
            raise RuntimeError("403 Forbidden")
    else:

        def _pre(*a, **k):
            return {"successful_terms": [{"matched": "tax:species:Gadus morhua"}]}

    monkeypatch.setattr(bold, "_preprocess_query", _pre)
    monkeypatch.setattr(bold, "_get_query_id", lambda *a, **k: query_id)
    monkeypatch.setattr(bold, "_download_results", lambda *a, **k: download)


def test_query_species_error_returns_none(monkeypatch):
    # An exception anywhere in the pipeline -> None (a failure, retried later).
    _stub_pipeline(monkeypatch, preprocess_raises=True)
    assert query_species("Gadus morhua") is None


def test_query_species_preprocessor_without_a_match_returns_none(monkeypatch):
    # Neither a taxon nor BOLD's record-ID fallback: a failure, not "not in BOLD".
    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(bold, "_preprocess_query", lambda *a, **k: {})
    assert query_species("Gadus morhua") is None


def test_query_species_queries_a_rejected_name_as_written(monkeypatch):
    # A name in failed_terms is neither "not in BOLD" nor a failure.
    _stub_pipeline(monkeypatch, download='{"processid": "P1"}\n')
    monkeypatch.setattr(
        bold,
        "_preprocess_query",
        lambda *a, **k: {"successful_terms": [], "failed_terms": ["Gadus morhua"]},
    )
    assert query_species("Gadus morhua") == (True, [{"processid": "P1"}])


def test_query_species_no_query_id_returns_none(monkeypatch):
    _stub_pipeline(monkeypatch, query_id=None)
    assert query_species("Gadus morhua") is None


def test_query_species_empty_download_is_in_bold_without_records(monkeypatch):
    # A successful query with no records: a genuine empty, cached as such.
    _stub_pipeline(monkeypatch, download="")
    assert query_species("Gadus morhua") == (True, [])


# --- Search outcomes, on BOLD's real answers for three species ---

FIXTURE = json.loads((Path(__file__).parent / "fixtures/bold_records.json").read_text())


def _stub_bold(monkeypatch):
    """BOLD answering from the fixture; a query for a name it does not know
    fails the test."""

    def _pre(query, **k):
        return FIXTURE[query.removeprefix("tax:species:")]["preprocessor"]

    def _query_id(term, **k):
        assert term.startswith("tax:"), f"queried a name BOLD does not know: {term}"
        return term.removeprefix("tax:species:")

    def _download(name, **k):
        return "\n".join(json.dumps(r) for r in FIXTURE[name]["records"])

    monkeypatch.setattr(bold, "_preprocess_query", _pre)
    monkeypatch.setattr(bold, "_get_query_id", _query_id)
    monkeypatch.setattr(bold, "_download_results", _download)


def test_each_queried_species_gets_its_sequences_or_why_it_has_none(monkeypatch):
    # Larus pipixcan is a name BOLD does not know: its preprocessor answers
    # with a search of record IDs, which _stub_bold refuses to query.
    _stub_bold(monkeypatch)
    species = ["Leucophaeus pipixcan", "Chaetodon leucopleura", "Larus pipixcan"]
    df, failed = get_records_from_species_list(species, rate_limit_delay=0)

    assert failed == []
    assert outcomes_by_species(df).to_dict() == {
        "Leucophaeus pipixcan": bold.SEQUENCES,
        "Chaetodon leucopleura": bold.GENBANK_ONLY,
        "Larus pipixcan": bold.NOT_IN_BOLD,
    }
    sequences = df[df["bold_outcome"] == bold.SEQUENCES]
    assert len(with_sequences(df)) == len(sequences) == 8
    assert set(sequences["scientificName"]) == {"Leucophaeus pipixcan"}


def test_a_species_bold_files_under_another_name_is_found_under_it(monkeypatch):
    # WoRMS accepts Larus pipixcan, which BOLD does not know; GBIF (and BOLD)
    # call it Leucophaeus pipixcan. The other GBIF names stand for other
    # species, so they are never searched: Chroicocephalus saundersi is
    # another WoRMS species, Larus genei is the NCBI name of Chroicocephalus
    # genei (a made-up lump, to test the rule), and GBIF lumps Larus
    # novaehollandiae and Larus scopulinus as Chroicocephalus novaehollandiae.
    _stub_bold(monkeypatch)
    gbif = {
        "Larus pipixcan": "Leucophaeus pipixcan",
        "Larus saundersi": "Chroicocephalus saundersi",
        "Chroicocephalus saundersi": "Chroicocephalus saundersi",
        "Larus ridibundus": "Larus genei",
        "Larus novaehollandiae": "Chroicocephalus novaehollandiae",
        "Larus scopulinus": "Chroicocephalus novaehollandiae",
    }
    ncbi = build_name_table(
        pd.DataFrame(
            {
                "scientificName": ["Larus genei"],
                "acceptedName": ["Chroicocephalus genei"],
                "acceptedNameUsageID": [137149],
                "matchedBy": ["name"],
            }
        )
    )
    gbif_rows = gbif_name_rows(
        pd.DataFrame(
            {
                "scientificName": list(gbif),
                "gbif_scientificName": list(gbif.values()),
                "taxonID": ["6Q4RB", "5YGFL", "5YGFL", "9VMHX", "5YGFR", "5YGFR"],
            }
        )
    )
    name_table = replace_source_rows(ncbi, gbif_rows, "gbif")
    species = [
        "Larus pipixcan",
        "Larus saundersi",
        "Larus ridibundus",
        "Larus novaehollandiae",
        "Larus scopulinus",
    ]
    other_names = prepare_bold_other_names(species, name_table)
    assert other_names == {"Larus pipixcan": [("Leucophaeus pipixcan", "gbif")]}

    df, failed = get_records_from_species_list(
        ["Larus pipixcan"], other_names, rate_limit_delay=0
    )
    assert failed == []
    assert outcomes_by_species(df).to_dict() == {"Larus pipixcan": bold.SEQUENCES}
    assert len(with_sequences(df)) == 8
    assert bold_name_rows(df)[["species", "name", "matched_by"]].values.tolist() == [
        ["Larus pipixcan", "Leucophaeus pipixcan", "gbif"]
    ]
