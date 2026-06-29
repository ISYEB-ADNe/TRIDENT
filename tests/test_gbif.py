"""Tests for pure GBIF helpers (coordinate parsing, extent naming). No network."""

import pandas as pd
import pytest

from trident.clients import gbif
from trident.clients.gbif import (
    counts_for_area,
    filter_taxon_matches,
    get_extent_column_name,
    parse_coordinate,
    search_taxon_occurrences,
    _GBIF_TAXON_COLS,
    _match_single_taxon_name,
)
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


class _FacetSession:
    """Session returning fixed facet counts; keys absent from it zero-fill."""

    def __init__(self, counts):
        self._counts = counts

    def get(self, *a, **k):
        counts = self._counts

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "facets": [
                        {"counts": [{"name": n, "count": c} for n, c in counts.items()]}
                    ]
                }

        return R()

    def close(self):
        pass


def test_search_returns_counts_and_zero_fills_absent(monkeypatch):
    # "111" has occurrences; "222" is genuinely absent -> count 0 (not failed).
    monkeypatch.setattr(
        gbif, "create_session", lambda *a, **k: _FacetSession({"111": 42})
    )
    df, failed = search_taxon_occurrences(["111", "222"])

    assert failed == []
    assert dict(zip(df["taxonID"], df["count"])) == {"111": 42, "222": 0}


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

    # Errored species are excluded from the result and queued for retry.
    assert set(sink) == {"Gadus morhua", "Thunnus thynnus"}
    assert df.empty  # both taxa errored -> dropped from the result


def test_failure_sink_uses_input_names_not_canonical(monkeypatch):
    # GBIF canonicalName differs in case from the input species string. The
    # failure_sink must carry the *input* string (the cache-item key), or the
    # cache cannot match it for retry.
    def fake_match(names, **k):
        return pd.DataFrame(
            [
                {
                    "query": n,
                    "canonicalName": n.lower().capitalize(),  # e.g. "Gadus morhua"
                    "confidence": 99,
                    "matchType": "EXACT",
                    "needs_review": False,
                    "taxonID": str(1000 + i),
                    "rank": "SPECIES",
                    "genus": n.split()[0].lower().capitalize(),
                    "kingdom": "Animalia",
                    "status": "ACCEPTED",
                }
                for i, n in enumerate(names)
            ]
        )

    monkeypatch.setattr(geo_pipeline, "match_taxon_names", fake_match)
    monkeypatch.setattr(gbif, "create_session", lambda *a, **k: _BoomSession())

    sink: list = []
    geo_pipeline.run_gbif_search(
        ["GADUS MORHUA"],
        latitude=43.0,
        longitude=7.0,
        extents=[500],
        failure_sink=sink,
        db_path=None,
    )
    assert sink == ["GADUS MORHUA"]  # input string, not "Gadus morhua"


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
    # No species clears the backbone filter; the step returns empty, no crash.
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
