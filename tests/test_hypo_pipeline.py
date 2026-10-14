"""Tests for the HYPO pipeline's network-free logic.

prepare_hypo_input (build per-seq BLAST inputs + Entrez query), run_hypo_merge
(map proxy hits back to target species), run_hypo_filter (threshold), and
finalize_hypo_results (attach the per-species NCBI check). The two network steps
(run_hypo_search, run_hypo_check) are only tested for their failure handling,
with BLAST stubbed out.

The cached steps are wrapped by @save_to_db (+ @preserve_sequence_order); we
unwrap to the pure logic so no database is involved.
"""

import inspect

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import trident.pipelines.hypo_pipeline as hypo_pipeline
from trident.core.database import view_cached_runs
from trident.pipelines.hypo_pipeline import (
    prepare_hypo_input,
    run_hypo_search,
    run_hypo_merge,
    run_hypo_filter,
    finalize_hypo_results,
)

_merge = inspect.unwrap(run_hypo_merge)
_filter = inspect.unwrap(run_hypo_filter)
_finalize = inspect.unwrap(finalize_hypo_results)


# --- prepare_hypo_input ---


def test_prepare_builds_seqrecords_and_entrez_from_mol_species():
    extra_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "dna_sequence_extra": ["ACGT", "TTGG"],
            "seq_id_extra": ["P1", "P2"],
        }
    )
    # Only the in_mol species constrains the BLAST (Entrez organism query).
    geo_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "scientificName": ["Gadus morhua", "Other species"],
            "in_mol": [True, False],
        }
    )
    out = prepare_hypo_input(extra_df, geo_df)

    assert list(out.keys()) == ["t1"]
    assert [r.id for r in out["t1"]["sequences"]] == ["P1", "P2"]
    q = out["t1"]["entrez_query"]
    assert '"Gadus morhua"[orgn]' in q
    assert "Other species" not in q  # not in_mol -> excluded


def test_prepare_empty_entrez_when_no_mol_species():
    extra_df = pd.DataFrame(
        {"seq_id": ["t1"], "dna_sequence_extra": ["ACGT"], "seq_id_extra": ["P1"]}
    )
    geo_df = pd.DataFrame(
        {"seq_id": ["t1"], "scientificName": ["X"], "in_mol": [False]}
    )
    out = prepare_hypo_input(extra_df, geo_df)
    assert out["t1"]["entrez_query"] == ""


# --- run_hypo_merge: proxy hit -> target species ---


def test_merge_maps_proxy_to_target_and_renames_hit():
    # EXTRA: target seq t1 / species "Gadus morhua" carried by proxy P1.
    extra_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "seq_id_extra": ["P1"],
            "dna_sequence_extra": ["ACGT"],
        }
    )
    # Proxy P1 BLASTed to a (different) hit species.
    hypo_search_df = pd.DataFrame(
        {
            "seq_id": ["P1"],
            "scientificName": ["Gadus macrocephalus"],
            "query_cover": [80.0],
            "identity_percentage": [92.0],
            "hit_def": ["def"],
            "hit_url": ["url"],
        }
    )
    out = _merge(hypo_search_df, extra_df)
    row = out.iloc[0]
    assert row["seq_id"] == "t1"  # target seq preserved
    assert row["scientificName"] == "Gadus morhua"  # target species preserved
    assert row["scientificName_hit"] == "Gadus macrocephalus"  # BLAST hit renamed
    assert row["identity_percentage"] == 92.0
    assert not any(c.endswith("_blast") for c in out.columns)  # dupes dropped


# --- run_hypo_filter: query_cover + identity thresholds ---


def test_filter_keeps_only_rows_passing_both_thresholds():
    df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t1"],
            "scientificName": ["A", "B", "C"],
            "query_cover": [60.0, 40.0, 60.0],  # B fails query_cover
            "identity_percentage": [96.0, 99.0, 90.0],  # C fails identity
        }
    )
    out = _filter(df, query_cover=50.0, identity=95.0)
    assert out["scientificName"].tolist() == ["A"]


# --- finalize_hypo_results: best proxy + attached NCBI check ---


def test_finalize_keeps_best_proxy_and_attaches_check():
    # Two proxy hits for (t1, Gadus morhua): best identity 95 wins.
    hypo_filter_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "scientificName": ["Gadus morhua", "Gadus morhua"],
            "identity_percentage": [95.0, 92.0],
            "query_cover": [100.0, 100.0],
        }
    )
    # The per-species NCBI check found the marker at 90% identity.
    hypo_check_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "identity_percentage": [90.0],
            "query_cover": [100.0],
            "hit_url": ["nurl"],
            "hit_found": [True],
        }
    )
    out = _finalize(hypo_filter_df, hypo_check_df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["identity_percentage"] == 95.0  # best proxy
    assert row["ncbi_top_identity_percentage"] == 90.0  # from the check


def test_finalize_check_absent_leaves_ncbi_stats_na():
    hypo_filter_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "identity_percentage": [95.0],
            "query_cover": [100.0],
        }
    )
    # Check ran but found nothing (hit_found False) -> excluded from check_hits.
    hypo_check_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "identity_percentage": [pd.NA],
            "query_cover": [pd.NA],
            "hit_url": [pd.NA],
            "hit_found": [False],
        }
    )
    out = _finalize(hypo_filter_df, hypo_check_df)
    assert pd.isna(out.iloc[0]["ncbi_top_identity_percentage"])


def test_finalize_tolerates_columnless_check_results():
    """Every check query failed: the cache returns a frame with no columns.
    Finalize must keep the HYPO species, with no check stats, not crash."""
    hypo_filter_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "identity_percentage": [95.0],
            "query_cover": [100.0],
        }
    )
    out = _finalize(hypo_filter_df, pd.DataFrame())
    assert list(out["scientificName"]) == ["Gadus morhua"]
    assert pd.isna(out.iloc[0]["ncbi_top_identity_percentage"])


# --- run_hypo_search: failure handling ---


def test_hypo_search_failed_proxy_blast_is_not_cached(tmp_path, monkeypatch):
    """A proxy record whose BLAST errored must be retried, not cached as "no
    hits". The cache key is the proxy record id, so that is what gets reported."""

    def blast(batches, **kwargs):
        # P1 errors; P2 is searched and genuinely returns no hits.
        return pd.DataFrame(), ["P1"]

    monkeypatch.setattr(hypo_pipeline, "run_blast_batches", blast)
    sequences = {
        "t1": {
            "sequences": [
                SeqRecord(Seq("ACGT"), id="P1"),
                SeqRecord(Seq("TTGG"), id="P2"),
            ],
            "entrez_query": '"Gadus morhua"[orgn]',
        }
    }
    db_path = str(tmp_path / "hypo.db")

    df, _ = run_hypo_search(sequences, db_path=db_path)

    assert df.attrs["failed_items"] == ["P1"]
    assert set(view_cached_runs(db_path, "hypo_search")["seq_id"]) == {"P2"}
