"""Tests for the GBIF client and GBIF search. No network: GBIF answers come
from fixtures or stand-in sessions."""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from trident.clients import gbif
from trident.clients.gbif import (
    area_params,
    counts_for_area,
    filter_taxon_matches,
    get_extent_column_name,
    parse_coordinate,
    search_taxon_occurrences,
    _GBIF_TAXON_COLS,
    _match_single_taxon_name,
    match_taxon_names,
)
from trident.core.utils import ensure_columns
from trident.pipelines import geo_pipeline


# --- parse_coordinate ---


def test_parse_numeric():
    assert parse_coordinate(45.5, "lat") == 45.5
    assert parse_coordinate(-120, "lon") == -120.0


def test_parse_decimal_string():
    assert parse_coordinate("45.5", "lat") == 45.5


def test_parse_dms_north():
    # 45°30'15"N = 45 + 30/60 + 15/3600
    assert parse_coordinate("45°30'15\"N", "lat") == pytest.approx(45.504166, abs=1e-5)


def test_parse_dms_south_is_negative():
    assert parse_coordinate("45°30'0\"S", "lat") == pytest.approx(-45.5, abs=1e-6)


def test_parse_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_coordinate(200, "lon")
    with pytest.raises(ValueError):
        parse_coordinate(95, "lat")


def test_parse_wrong_direction_for_type_raises():
    # An east/west direction is invalid for a latitude.
    with pytest.raises(ValueError):
        parse_coordinate("45°30'15\"E", "lat")


def test_parse_empty_and_garbage_raise():
    with pytest.raises(ValueError):
        parse_coordinate("", "lat")
    with pytest.raises(ValueError):
        parse_coordinate("not a coordinate", "lat")


# --- get_extent_column_name ---


def test_extent_numeric():
    assert get_extent_column_name(100) == "100 km"
    assert get_extent_column_name(100.0) == "100 km"
    assert get_extent_column_name("250") == "250 km"


def test_extent_string():
    assert get_extent_column_name("global") == "global"
    assert get_extent_column_name("GLOBAL") == "global"


# --- match failure contract (matchType == "ERROR" -> routed to failure_sink) ---


def test_match_error_result_on_request_failure():
    class BoomSession:
        def get(self, *a, **k):
            raise RuntimeError("403 Forbidden")

    res = _match_single_taxon_name("Gadus morhua", session=BoomSession())
    assert res["matchType"] == "ERROR"
    assert res["query"] == "Gadus morhua"
    assert res["needs_review"] is True


# --- occurrence fetch failure contract: errored keys reported, not counted ---


class _BoomSession:
    """Session whose every request raises, simulating an occurrence outage."""

    def get(self, *a, **k):
        raise RuntimeError("network down")

    def close(self):
        pass


def test_failed_occurrence_batch_reports_failed_keys(monkeypatch):
    # Every batch errors -> no count rows emitted; the keys come back as failed
    # so the caller retries them instead of caching them as absent.
    monkeypatch.setattr(gbif, "create_session", lambda *a, **k: _BoomSession())

    df, failed = search_taxon_occurrences(["111", "222"])

    assert df.empty
    assert set(failed) == {"111", "222"}


class _JsonSession:
    """Session answering each request with answer(params) as its JSON body."""

    def __init__(self, answer):
        self._answer = answer

    def get(self, url, params=None, timeout=None):
        body = self._answer(params)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return body

        return R()

    def close(self):
        pass


def _FacetSession(counts):
    """Session returning fixed facet counts; keys absent from it zero-fill."""
    facets = [{"name": n, "count": c} for n, c in counts.items()]
    return _JsonSession(lambda params: {"facets": [{"counts": facets}]})


def test_search_returns_counts_and_zero_fills_absent(monkeypatch):
    # "111" has occurrences; "222" is genuinely absent -> count 0 (not failed).
    monkeypatch.setattr(
        gbif, "create_session", lambda *a, **k: _FacetSession({"111": 42})
    )
    df, failed = search_taxon_occurrences(["111", "222"])

    assert failed == []
    assert dict(zip(df["taxonID"], df["count"])) == {"111": 42, "222": 0}


def test_search_counts_rare_species_cut_from_a_full_facet_list(monkeypatch):
    # Like GBIF, the facet lists each species' ancestors too, most records
    # first, cut at facetLimit: in a batch of 4 the species fall off the end.
    def answer(params):
        counts = []
        for key in params["taxonKey"]:
            counts += [{"name": f"{key}-a{i}", "count": 1000} for i in range(15)]
            counts.append({"name": key, "count": int(key)})
        counts.sort(key=lambda c: -c["count"])
        return {"facets": [{"counts": counts[: params["facetLimit"]]}]}

    monkeypatch.setattr(gbif, "create_session", lambda *a, **k: _JsonSession(answer))
    keys = ["1", "2", "3", "4"]
    df, failed = search_taxon_occurrences(keys, batch_size=4)

    assert failed == []
    assert dict(zip(df["taxonID"], df["count"])) == {k: int(k) for k in keys}


def test_search_counts_a_species_only_from_its_own_batch(monkeypatch):
    # Like GBIF: asking for Larus glaucoides thayeri also lists L. glaucoides,
    # with only the subspecies' records (counts seen live, worldwide).
    thayeri, glaucoides = "B765P", "3SBQC"

    def answer(params):
        if params["taxonKey"] == [thayeri]:
            counts = {thayeri: 4760, glaucoides: 4760}
        else:
            counts = {thayeri: 4760, glaucoides: 520609}
        return {
            "facets": [{"counts": [{"name": n, "count": c} for n, c in counts.items()]}]
        }

    monkeypatch.setattr(gbif, "create_session", lambda *a, **k: _JsonSession(answer))
    df, _ = search_taxon_occurrences([thayeri, glaucoides], batch_size=1)

    assert dict(zip(df["taxonID"], df["count"])) == {thayeri: 4760, glaucoides: 520609}


def test_run_gbif_search_routes_occurrence_errors_to_failure_sink(monkeypatch):
    # Matches succeed (two good taxon keys), but the occurrence fetch errors.
    def fake_match(names, **k):
        return pd.DataFrame(
            [
                {
                    "query": n,
                    "canonicalName": n,
                    "confidence": 99,
                    "matchType": "EXACT",
                    "needs_review": False,
                    "taxonID": str(1000 + i),
                    "rank": "SPECIES",
                    "genus": n.split()[0],
                    "kingdom": "Animalia",
                    "status": "ACCEPTED",
                    "matched_by": "WoRMS ID",
                }
                for i, n in enumerate(names)
            ]
        )

    monkeypatch.setattr(geo_pipeline, "match_taxon_names", fake_match)
    monkeypatch.setattr(gbif, "create_session", lambda *a, **k: _BoomSession())

    sink: list = []
    df, _ = geo_pipeline.run_gbif_search(
        ["Gadus morhua", "Thunnus thynnus"],
        latitude=43.0,
        longitude=7.0,
        extents=[500],
        failure_sink=sink,
        db_path=None,
    )

    # Errored species are excluded from the result and queued for retry in
    # that area.
    assert set(sink) == {("Gadus morhua", 500), ("Thunnus thynnus", 500)}
    assert df.empty  # both taxa errored -> dropped from the result


# --- no usable matches: filter must not crash, returns empty columned frame ---


def test_filter_taxon_matches_low_confidence_returns_empty():
    # canonicalName present but every row below the confidence threshold.
    df = pd.DataFrame(
        [
            {
                "query": "Foo bar",
                "canonicalName": "Foo bar",
                "confidence": 40,
                "matchType": "FUZZY",
                "needs_review": True,
                "taxonID": "1",
                "rank": "SPECIES",
                "genus": "Foo",
                "kingdom": "Animalia",
                "status": "ACCEPTED",
            }
        ]
    )
    out = filter_taxon_matches(df)
    assert out.empty
    assert list(out.columns) == _GBIF_TAXON_COLS


def test_filter_taxon_matches_all_none_no_canonicalname():
    # Every match is NONE, so the canonicalName column never exists.
    df = pd.DataFrame(
        [
            {
                "query": "Foo bar",
                "confidence": 0,
                "matchType": "NONE",
                "needs_review": True,
                "taxonID": None,
            }
        ]
    )
    out = filter_taxon_matches(df)
    assert out.empty
    assert list(out.columns) == _GBIF_TAXON_COLS


def test_counts_for_area_empty_keys_returns_empty():
    out, failed = counts_for_area([], extent="global")
    assert out.empty
    assert failed == []
    assert list(out.columns) == ["taxonID", "occurrences", "gbif_extent"]


def test_run_gbif_search_no_matches_returns_empty(monkeypatch):
    # No species clears the match filter; the step returns empty, no crash.
    def m_none(names, **k):
        return pd.DataFrame(
            [
                {
                    "query": n,
                    "confidence": 0,
                    "matchType": "NONE",
                    "needs_review": True,
                    "taxonID": None,
                }
                for n in names
            ]
        )

    monkeypatch.setattr(geo_pipeline, "match_taxon_names", m_none)
    df, _ = geo_pipeline.run_gbif_search(
        ["Foo bar", "Baz qux"],
        latitude=43.0,
        longitude=7.0,
        extents=[500],
        db_path=None,
    )
    assert df.empty
    # The columns read by the GBIF merge and filter are present.
    assert {"scientificName", "occurrences", "gbif_extent"} <= set(df.columns)


def test_merge_drops_species_without_gbif_data(monkeypatch):
    """Species without GBIF data have no extent row, so the merge leaves them
    out (as the cache layer would); the empty result keeps its columns."""
    no_match = gbif.filter_taxon_matches(pd.DataFrame(columns=gbif.GBIF_MATCH_COLUMNS))
    gbif_search_df = ensure_columns(no_match, ["occurrences", "gbif_extent"])
    tax_df = pd.DataFrame(
        {"seq_id": ["s1"], "scientificName": ["Gadus morhua"], "in_mol": [True]}
    )

    merged = inspect.unwrap(geo_pipeline.run_gbif_merge)(gbif_search_df, tax_df)

    assert merged.empty
    assert {"seq_id", "scientificName", "in_mol", "geo_extent"} <= set(merged.columns)


def test_merge_keeps_whole_counts_when_a_species_lacks_gbif_data():
    """A TAX species GBIF does not hold leaves the merge, and the counts of
    the others stay whole numbers (a decimal column would be saved as such)."""
    gbif_search_df = pd.DataFrame(
        {
            "scientificName": ["Gadus morhua"],
            "taxonURL": ["https://www.gbif.org/taxon/X"],
            "occurrences": [12],
            "gbif_extent": ["500"],
        }
    )
    tax_df = pd.DataFrame(
        {"seq_id": ["s1", "s1"], "scientificName": ["Gadus morhua", "Gadus ogac"]}
    )

    merged = inspect.unwrap(geo_pipeline.run_gbif_merge)(gbif_search_df, tax_df)

    assert merged["scientificName"].tolist() == ["Gadus morhua"]
    assert pd.api.types.is_integer_dtype(merged["geo_occurrences"])


def test_filter_finds_the_extent_however_it_is_written():
    """The filter's extent matches the saved one whether given as 250, 250.0
    or "250" (a notebook may pass a float)."""
    merged = pd.DataFrame(
        {
            "seq_id": ["s1", "s1"],
            "scientificName": ["Gadus morhua", "Gadus morhua"],
            "geo_occurrences": [12, 40],
            "geo_extent": ["250", "global"],
        }
    ).reindex(columns=[*geo_pipeline.GEO_LIST_COLUMNS, "geo_extent"])
    kept = [
        inspect.unwrap(geo_pipeline.run_gbif_filter)(merged, extent, 3)
        for extent in (250, 250.0, "250")
    ]
    assert [df["geo_occurrences"].tolist() for df in kept] == [[12]] * 3


def test_geo_list_takes_gbif_counts_not_gbif_names():
    """The GEO merge adds GBIF's counts and taxon page to the TAX rows; GBIF's
    own name of the species stays out of the GEO list."""
    gbif_search_df = pd.DataFrame(
        {
            "scientificName": ["Larus saundersi"],
            "gbif_scientificName": ["Chroicocephalus saundersi"],
            "gbif_matched_by": ["WoRMS ID"],
            "genus": ["Chroicocephalus"],
            "taxonID": ["5YGFL"],
            "taxonURL": ["https://www.gbif.org/taxon/5YGFL"],
            "occurrences": [12],
            "gbif_extent": [500],
        }
    )
    tax_df = pd.DataFrame(
        {
            "seq_id": ["s1"],
            "scientificName": ["Larus saundersi"],
            "genus": ["Larus"],
            "taxonID": ["343919"],
        }
    )

    merged = inspect.unwrap(geo_pipeline.run_gbif_merge)(gbif_search_df, tax_df)

    row = merged.iloc[0]
    assert (row["genus"], row["taxonID"], row["geo_occurrences"]) == (
        "Larus",
        "343919",
        12,
    )
    assert row["geo_url"] == "https://www.gbif.org/taxon/5YGFL"
    assert not {"gbif_scientificName", "gbif_matched_by"} & set(merged.columns)


def test_match_no_names_keeps_match_columns():
    assert list(gbif.match_taxon_names([]).columns) == gbif.GBIF_MATCH_COLUMNS


# --- v2 match (COL XR) on real GBIF answers ---

# Real v2 species/match answers (Oct 2026), queried with the WoRMS ID of each name
V2_MATCHES = json.loads(
    (Path(__file__).parent / "fixtures" / "gbif_v2_match.json").read_text()
)


def test_name_matching_reuses_one_session_per_thread(monkeypatch):
    """Each matching thread reuses one session for all its names (no new
    connection per name), and every session is closed once all are matched."""
    answer = next(iter(V2_MATCHES.values()))["body"]
    sessions = []

    class _CountedSession(_JsonSession):
        closed = False

        def close(self):
            self.closed = True

    def new_session(*a, **k):
        sessions.append(_CountedSession(lambda params: answer))
        return sessions[-1]

    monkeypatch.setattr(gbif, "create_session", new_session)
    names = [f"Species {i}" for i in range(60)]

    matches = match_taxon_names(names, n_jobs=3)

    assert len(matches) == 60
    assert 1 <= len(sessions) <= 3
    assert all(session.closed for session in sessions)


def test_species_counted_under_gbif_species_of_worms_id_or_name(monkeypatch):
    """Each name keeps its row under the GBIF species of its WoRMS ID, or of its
    name or a variant of it; names sharing a species are both kept. A synonym
    in GBIF of another species (a separate species in WoRMS) counts under its
    own GBIF key; one that only changes the genus, under the accepted key. A
    subspecies in GBIF counts under its species when it is the nominate one,
    else under its own key. Each name also keeps GBIF's species for it: the
    accepted species of a synonym, the species of a subspecies. A genus-only
    or fuzzy match is left out."""
    answers = {name: match["body"] for name, match in V2_MATCHES.items()}
    # A misspelt name GBIF matches fuzzily to Spicara flexuosum, without an ID
    spicara = answers["Spicara flexuosum"]
    answers["Spicara flexuosa"] = spicara | {
        "diagnostics": {"matchType": "FUZZY", "confidence": 96}
    }
    # A name GBIF holds as a synonym of a species with the same epithet in
    # another genus (Lestrolepis japonica)
    answers["Arctozenus japonicus"] = answers["Lestrolepis philippina"]
    monkeypatch.setattr(
        gbif,
        "create_session",
        lambda *a, **k: _JsonSession(lambda params: answers[params["scientificName"]]),
    )
    worms_ids = {
        name: match["params"]["scientificNameID"].rsplit(":", 1)[1]
        for name, match in V2_MATCHES.items()
    }

    matches = match_taxon_names(list(answers), worms_ids=worms_ids)
    kept = filter_taxon_matches(matches).set_index("scientificName")

    assert set(kept.index) == {
        "Larus saundersi",  # WoRMS ID, GBIF's name: Chroicocephalus saundersi
        "Chroicocephalus saundersi",
        "Spicara flexuosum",
        "Monocentris neozelanica",  # name variant: neozelanicus
        "Arctozenus japonicus",  # synonym, genus change only: L. japonica
        "Lestrolepis philippina",  # synonym of L. japonica, separate in WoRMS
        "Larus atricilla",  # nominate subspecies of Leucophaeus atricilla
        "Larus novaehollandiae",  # nominate subspecies of C. novaehollandiae
        "Larus scopulinus",  # subspecies of Chroicocephalus novaehollandiae
        "Larus thayeri",  # subspecies of Larus glaucoides
    }
    larus = kept.loc["Larus saundersi"]
    assert larus["gbif_scientificName"] == "Chroicocephalus saundersi"
    assert larus["gbif_matched_by"] == "WoRMS ID"
    assert larus["taxonID"] == kept.loc["Chroicocephalus saundersi", "taxonID"]
    assert larus["taxonURL"] == f"https://www.gbif.org/taxon/{larus['taxonID']}"
    assert larus["aphia_id"] == int(worms_ids["Larus saundersi"])
    lestrolepis = V2_MATCHES["Lestrolepis philippina"]["body"]
    own, japonica = kept.loc["Lestrolepis philippina"], kept.loc["Arctozenus japonicus"]
    assert own["taxonID"] == lestrolepis["usage"]["key"]
    assert own["gbif_scientificName"] == "Lestrolepis philippina"
    assert "synonym in GBIF of Lestrolepis japonica" in own["gbif_note"]
    assert japonica["taxonID"] == lestrolepis["acceptedUsage"]["key"]
    assert own["gbif_speciesID"] == japonica["gbif_speciesID"] == japonica["taxonID"]
    names = kept["gbif_scientificName"].to_dict()
    assert names["Larus atricilla"] == "Leucophaeus atricilla atricilla"
    assert names["Larus novaehollandiae"] == (
        "Chroicocephalus novaehollandiae novaehollandiae"
    )
    assert names["Larus scopulinus"] == "Chroicocephalus novaehollandiae scopulinus"
    assert names["Larus thayeri"] == "Larus glaucoides thayeri"
    scopulinus = V2_MATCHES["Larus scopulinus"]["body"]["classification"]
    assert kept.loc["Larus novaehollandiae", "taxonID"] == scopulinus[-2]["key"]
    assert kept.loc["Larus scopulinus", "taxonID"] == scopulinus[-1]["key"]
    assert kept.loc["Larus scopulinus", "gbif_speciesID"] == scopulinus[-2]["key"]


def test_names_sharing_a_gbif_species_both_get_its_counts(monkeypatch):
    def fake_match(names, **k):
        return pd.DataFrame(
            [
                {
                    "query": n,
                    "canonicalName": "Chroicocephalus saundersi",
                    "confidence": 100,
                    "matchType": "EXACT",
                    "needs_review": False,
                    "taxonID": "5YGFL",
                    "rank": "SPECIES",
                    "genus": "Chroicocephalus",
                    "matched_by": "WoRMS ID",
                }
                for n in names
            ]
        )

    monkeypatch.setattr(geo_pipeline, "match_taxon_names", fake_match)
    monkeypatch.setattr(
        gbif, "create_session", lambda *a, **k: _FacetSession({"5YGFL": 12})
    )
    df, _ = geo_pipeline.run_gbif_search(
        ["Larus saundersi", "Chroicocephalus saundersi"],
        latitude=24.5,
        longitude=54.4,
        extents=[500],
        db_path=None,
    )

    assert dict(zip(df["scientificName"], df["occurrences"])) == {
        "Larus saundersi": 12,
        "Chroicocephalus saundersi": 12,
    }


# --- search areas ---


def test_area_params_circle_box_and_global():
    assert area_params(24.5, 54.4, 500, "circle") == {"geoDistance": "24.5,54.4,500km"}
    box = area_params(24.5, 54.4, 500, "box")["geometry"]
    assert box.startswith("POLYGON((") and box.count(",") == 4
    assert area_params(24.5, 54.4, "global", "circle") == {}
    with pytest.raises(ValueError):
        area_params(24.5, 54.4, 500, "square")
