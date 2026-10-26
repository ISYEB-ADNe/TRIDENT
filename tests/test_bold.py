"""Tests for the BOLD client: record filters, the query's failure contract, and
the reason each queried species gets no sequence. BOLD is faked at its HTTP
boundary with its real answers (tests/fixtures/bold_records.json)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

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


def test_records_kept_are_coi5p_and_not_mined_from_genbank_unless_asked():
    records = [
        _rec(processid="A"),
        _rec(processid="B", marker_code="COI-3P"),
        _rec(processid="C", marker_code="16S"),
        _rec(processid="D", inst="Mined from GenBank, NCBI"),
    ]
    assert _clean_records(records)[0]["seq_id"].tolist() == ["A"]
    assert _clean_records(records, keep_only_COI5P=False)[0]["seq_id"].tolist() == [
        "A",
        "B",
        "C",
    ]
    assert _clean_records(records, keep_ncbi=True)[0]["seq_id"].tolist() == ["A", "D"]


def test_a_record_becomes_a_row_with_its_taxon_and_link():
    """No record gives an empty frame with the same columns."""
    df, _ = _clean_records([_rec(processid="P9", taxid=42)])
    row = df.iloc[0]
    assert row["scientificName"] == "Balaenoptera physalus"
    assert row["specificEpithet"] == "physalus"
    assert row["dna_sequence"] == "ACGT"
    assert (row["taxonID"], row["taxonID_db"]) == ("42", "BOLD")
    assert row["seq_url"].endswith("/record/P9")
    empty = _clean_records([])[0]
    assert empty.empty and list(empty.columns) == list(df.columns)


# --- BOLD at its HTTP boundary, on its real answers for three species ---

FIXTURE = json.loads((Path(__file__).parent / "fixtures/bold_records.json").read_text())


class _Answer:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def json(self):
        return self._body


class _BoldSession:
    """BOLD's API answering from the fixture: the preprocessor, a query ID
    (``{"query_id", "extent_limit"}`` as BOLD gives it) and the records as
    JSONL, the records trimmed to the fields the client reads. *overrides*
    replaces an endpoint's answer (preprocessor, query, download)."""

    def __init__(self, **overrides):
        self.overrides = overrides

    def get(self, url, params=None, timeout=None):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint in self.overrides:
            return self.overrides[endpoint]
        if endpoint == "preprocessor":
            name = params["query"].removeprefix("tax:species:")
            return _Answer(body=FIXTURE[name]["preprocessor"])
        if endpoint == "query":
            term = params["query"]
            assert term.startswith("tax:"), f"queried a name BOLD does not know: {term}"
            return _Answer(body={"query_id": term, "extent_limit": None})
        query_id = url.rsplit("/", 2)[-2]
        records = FIXTURE[query_id.removeprefix("tax:species:")]["records"]
        return _Answer(text="\n".join(json.dumps(r) for r in records))

    def close(self):
        pass


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"preprocessor": _Answer(status=403)}, None),
        ({"preprocessor": _Answer(body={})}, None),
        ({"query": _Answer(body={"extent_limit": None})}, None),
        ({"download": _Answer(text="")}, (True, [])),
    ],
    ids=["refused", "no match", "no query id", "no record"],
)
def test_query_fails_only_when_bold_gave_no_usable_answer(overrides, expected):
    """None is a failure (retried); a query with no record is a genuine empty."""
    session = _BoldSession(**overrides)
    assert query_species("Leucophaeus pipixcan", session=session) == expected


def test_a_name_the_preprocessor_rejects_is_queried_as_written():
    rejected = {"successful_terms": [], "failed_terms": ["Leucophaeus pipixcan"]}
    session = _BoldSession(preprocessor=_Answer(body=rejected))
    in_bold, records = query_species("Leucophaeus pipixcan", session=session)
    assert in_bold and len(records) == 8


def test_each_queried_species_gets_its_sequences_or_why_it_has_none(monkeypatch):
    # Larus pipixcan is a name BOLD does not know: its preprocessor answers
    # with a search of record IDs, which _BoldSession refuses to query.
    monkeypatch.setattr(bold, "create_session", lambda **k: _BoldSession())
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
    monkeypatch.setattr(bold, "create_session", lambda **k: _BoldSession())
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
