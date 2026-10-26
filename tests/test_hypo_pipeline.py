"""Tests for the HYPO pipeline's network-free logic.

hypo_queries and prepare_hypo_input (Entrez query of the MOL list, per-seq BLAST inputs), run_hypo_merge
(map proxy hits back to target species), run_hypo_filter (threshold), and
finalize_hypo_results (attach the per-species NCBI check). The two network steps
(run_hypo_search, run_hypo_check) are tested for their failure handling and
for running their BLAST jobs in parallel, with BLAST stubbed out.

The cached steps are wrapped by @save_to_db (+ @preserve_sequence_order); we
unwrap to the pure logic so no database is involved.
"""

import inspect
import threading

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import trident.pipelines.hypo_pipeline as hypo_pipeline
from trident.core.database import view_cached_runs
from trident.pipelines.names import build_name_table, gbif_name_rows
from trident.pipelines.hypo_pipeline import (
    count_hypo_searches,
    hypo_check_queries,
    hypo_queries,
    prepare_hypo_input,
    run_hypo_check,
    run_hypo_search,
    run_hypo_merge,
    run_hypo_filter,
    finalize_hypo_results,
)

_merge = inspect.unwrap(run_hypo_merge)
_filter = inspect.unwrap(run_hypo_filter)
_finalize = inspect.unwrap(finalize_hypo_results)


# --- prepare_hypo_input ---


def test_prepare_constrains_the_search_to_the_mol_list():
    """The proxies of a sequence are BLASTed against its MOL species that WoRMS
    kept in the TAX list, whether or not they passed GEO, by their accepted
    name and the other NCBI names its MOL hits gave them. A sequence without such
    species is not searched: unconstrained, a proxy would match its own
    species."""
    extra_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t2"],
            "proxy_sequence": ["ACGT", "TTGG", "AAAA"],
            "proxy_id": ["P1", "P2", "P3"],
        }
    )
    tax_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t1", "t2"],
            "scientificName": [
                "Omobranchus ferox",
                "Omobranchus fasciolatus",
                "Caranx ruber",
                "Caranx ignobilis",
            ],
            "in_mol": [True, False, True, False],
        }
    )
    mol_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t1", "t2"],
            "scientificName": [
                "Omobranchus ferox",
                "omobranchus  ferox",
                "Carangoides ruber",
                "Caranx sp.",
            ],
        }
    )
    name_table = build_name_table(
        pd.DataFrame(
            {
                "scientificName": ["Carangoides ruber"],
                "acceptedName": ["Caranx ruber"],
                "acceptedNameUsageID": [218434],
                "matchedBy": ["name"],
            }
        )
    )
    queries_df = hypo_queries(tax_df, mol_df, name_table)
    out = prepare_hypo_input(extra_df, queries_df)

    assert list(out.keys()) == ["t1"]
    assert [r.id for r in out["t1"]["sequences"]] == ["P1", "P2"]
    assert out["t1"]["entrez_query"] == (
        '("Carangoides ruber"[orgn] OR "Caranx ruber"[orgn] OR '
        '"Omobranchus ferox"[orgn])'
    )
    assert count_hypo_searches(extra_df, queries_df) == (2, 1)


# --- run_hypo_merge: proxy hit -> target species ---


def test_merge_maps_proxy_to_target_under_its_own_query():
    """Each target gets the hits of its proxy found under its own Entrez query:
    for proxy P1, shared by t1 and t2, t1 does not get the hit found under t2's
    query. A hit NCBI returned outside the MOL list (here the target species
    itself) is dropped."""
    extra_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t2"],
            "scientificName": ["Gadus morhua", "Gadus chalcogrammus"],
            "proxy_id": ["P1", "P1"],
            "proxy_sequence": ["ACGT", "ACGT"],
        }
    ).reindex(columns=hypo_pipeline.EXTRA_COLUMNS)
    queries_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t2"],
            "entrez_query": ['("Gadus macrocephalus"[orgn])', '("Gadus ogac"[orgn])'],
            "mol_species": ["Gadus macrocephalus", "Gadus ogac"],
        }
    )
    hypo_search_df = pd.DataFrame(
        {
            "seq_id": ["P1", "P1", "P1"],
            "scientificName": ["Gadus macrocephalus", "Gadus ogac", "Gadus morhua"],
            "query_cover": [80.0, 70.0, 99.0],
            "identity_percentage": [92.0, 90.0, 100.0],
            "hit_def": ["def", "def", "def"],
            "hit_url": ["url", "url", "url"],
            "entrez_query": [
                '("Gadus macrocephalus"[orgn])',
                '("Gadus ogac"[orgn])',
                '("Gadus macrocephalus"[orgn])',
            ],
        }
    )
    out = _merge(hypo_search_df, extra_df, queries_df)
    assert list(zip(out["seq_id"], out["proxy_hit_species"])) == [
        ("t1", "Gadus macrocephalus"),
        ("t2", "Gadus ogac"),
    ]
    row = out.iloc[0]
    assert row["scientificName"] == "Gadus morhua"  # target species preserved
    assert row["proxy_identity"] == 92.0
    assert not any(c.endswith("_blast") for c in out.columns)  # dupes dropped
    assert "entrez_query" not in out.columns


# --- run_hypo_filter: query_cover + identity thresholds ---


def test_filter_keeps_only_rows_passing_both_thresholds():
    df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1", "t1"],
            "scientificName": ["A", "B", "C"],
            "proxy_query_cover": [60.0, 40.0, 60.0],  # B fails query_cover
            "proxy_identity": [96.0, 99.0, 90.0],  # C fails identity
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
            "proxy_identity": [95.0, 92.0],
            "proxy_query_cover": [100.0, 100.0],
        }
    ).reindex(columns=hypo_pipeline.HYPO_HIT_COLUMNS)
    # The per-species NCBI check found the marker at 90% over the whole
    # sequence, and at 100% over a short part of it.
    hypo_check_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "scientificName": ["Gadus morhua", "Gadus morhua"],
            "identity_percentage": [100.0, 90.0],
            "query_cover": [40.0, 100.0],
            "hit_url": ["purl", "nurl"],
            "hit_found": [True, True],
        }
    )
    out = _finalize(hypo_filter_df, hypo_check_df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["proxy_identity"] == 95.0  # best proxy
    # The best check hit is the one covering the MOL query cover.
    assert row["hypo_check_identity"] == 90.0


def test_finalize_check_absent_leaves_ncbi_stats_na():
    hypo_filter_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "proxy_identity": [95.0],
            "proxy_query_cover": [100.0],
        }
    ).reindex(columns=hypo_pipeline.HYPO_HIT_COLUMNS)
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
    assert pd.isna(out.iloc[0]["hypo_check_identity"])


def test_check_hits_stay_numeric_after_a_first_check_without_hit(tmp_path, monkeypatch):
    """The first check results saved have no hit; hits saved later must still
    read back as numbers (the database must not type the columns as text)."""
    hit = pd.DataFrame(
        {
            "scientificName": ["Gadus morhua"],
            "query_cover": [100.0],
            "identity_percentage": [99.5],
            "hit_def": ["Gadus morhua"],
            "hit_url": ["https://www.ncbi.nlm.nih.gov/nucleotide/X1"],
        }
    )

    def blast(batch, entrez_query, **kwargs):
        found = "Gadus" in entrez_query
        return pd.DataFrame({"seq_id": [batch[0].id]}) if found else pd.DataFrame()

    monkeypatch.setattr(hypo_pipeline, "submit_blast_batch", blast)
    monkeypatch.setattr(hypo_pipeline, "process_blast_results", lambda df, **k: hit)
    db_path = str(tmp_path / "check.db")

    def check(species):
        filter_df = pd.DataFrame(
            {"seq_id": ["t1"], "scientificName": [species], "dna_sequence": ["ACGT"]}
        )
        return run_hypo_check(filter_df, db_path=db_path)[0]

    check("Salmo salar")  # saved first, no hit
    check("Gadus morhua")  # saved second, with a hit
    cached = check("Gadus morhua")  # read back from the database

    assert pd.api.types.is_numeric_dtype(cached["identity_percentage"])


def test_finalize_after_a_check_that_found_nothing(monkeypatch):
    """When the check finds none of the species, their rows have no hit stats
    but every column is present, so finalize keeps the species without check
    stats."""
    monkeypatch.setattr(
        hypo_pipeline, "submit_blast_batch", lambda *a, **k: pd.DataFrame()
    )
    hypo_filter_df = pd.DataFrame(
        {
            "seq_id": ["t1"],
            "scientificName": ["Gadus morhua"],
            "dna_sequence": ["ACGT"],
            "proxy_identity": [95.0],
            "proxy_query_cover": [100.0],
        }
    ).reindex(columns=hypo_pipeline.HYPO_HIT_COLUMNS)

    hypo_check_df = inspect.unwrap(run_hypo_check)(hypo_filter_df)
    out = _finalize(hypo_filter_df, hypo_check_df)

    assert set(hypo_pipeline.HYPO_CHECK_COLUMNS) <= set(hypo_check_df.columns)
    assert list(out["scientificName"]) == ["Gadus morhua"]
    assert pd.isna(out.iloc[0]["hypo_check_identity"])


# --- run_hypo_search: failure handling ---


def test_hypo_search_failed_proxy_blast_is_not_cached(tmp_path, monkeypatch):
    """A proxy record whose BLAST errored is retried, not cached as "no hits".
    The cache key is (proxy record id, MOTU query): a proxy shared by two MOTUs
    that errors under one query stays cached under the other."""
    query_a, query_b = '"Gadus morhua"[orgn]', '"Gadus chalcogrammus"[orgn]'

    def blast(
        batch, batch_id, circuit, progress_handler=None, entrez_query=None, **kwargs
    ):
        # P1 errors under query A only; every other search returns no hits.
        failed = [r.id for r in batch if r.id == "P1" and entrez_query == query_a]
        return [], failed

    monkeypatch.setattr(hypo_pipeline, "blast_batch", blast)
    sequences = {
        "t1": {
            "sequences": [
                SeqRecord(Seq("ACGT"), id="P1"),
                SeqRecord(Seq("TTGG"), id="P2"),
            ],
            "entrez_query": query_a,
        },
        "t2": {
            "sequences": [SeqRecord(Seq("ACGT"), id="P1")],
            "entrez_query": query_b,
        },
    }
    db_path = str(tmp_path / "hypo.db")

    df, _ = run_hypo_search(sequences, db_path=db_path)

    assert df.attrs["failed_items"] == [("P1", query_a)]
    cached = view_cached_runs(db_path, "hypo_search")
    assert sorted(zip(cached["seq_id"], cached["entrez_query"])) == sorted(
        [("P1", query_b), ("P2", query_a)]
    )


# --- run_hypo_search: BLAST jobs ---


def test_hypo_search_runs_motus_in_parallel(tmp_path, monkeypatch):
    """A slow BLAST job holds only its own thread: the next MOTU's job runs
    meanwhile, instead of waiting for it."""
    other_done = threading.Event()

    def blast(
        batch, batch_id, circuit, progress_handler=None, entrez_query=None, **kwargs
    ):
        if batch[0].id == "P1":  # t1's job is slow until t2's job has run
            assert other_done.wait(timeout=5), "t2 waited for t1's job"
        else:
            other_done.set()
        return [], []

    monkeypatch.setattr(hypo_pipeline, "blast_batch", blast)
    sequences = {
        "t1": {"sequences": [SeqRecord(Seq("ACGT"), id="P1")], "entrez_query": "a"},
        "t2": {"sequences": [SeqRecord(Seq("TTGG"), id="P2")], "entrez_query": "b"},
    }

    run_hypo_search(sequences, num_threads=2, db_path=str(tmp_path / "hypo.db"))

    assert other_done.is_set()


def test_hypo_search_batches_motus_sharing_a_query(tmp_path, monkeypatch):
    """MOTUs with the same Entrez query fill one batch, and a proxy they share
    is sent once; a different query gets its own batch."""
    sent = []

    def blast(
        batch, batch_id, circuit, progress_handler=None, entrez_query=None, **kwargs
    ):
        sent.append((entrez_query, [r.id for r in batch]))
        return [], []

    monkeypatch.setattr(hypo_pipeline, "blast_batch", blast)
    p1, p2, p3 = (
        SeqRecord(Seq(s), id=i)
        for s, i in [("ACGT", "P1"), ("TTGG", "P2"), ("GGCC", "P3")]
    )
    sequences = {
        "t1": {"sequences": [p1, p2], "entrez_query": "a"},
        "t2": {"sequences": [p2], "entrez_query": "a"},
        "t3": {"sequences": [p3], "entrez_query": "b"},
    }

    run_hypo_search(sequences, batch_size=10, db_path=str(tmp_path / "hypo.db"))

    assert sorted(sent) == [("a", ["P1", "P2"]), ("b", ["P3"])]


# --- run_hypo_check ---


def test_hypo_check_searches_every_name_of_the_species(tmp_path, monkeypatch):
    """A species is checked under its own name and the names other sources
    give it: NCBI finds nothing for Phalacrocorax pygmaeus, which it files as
    Microcarbo pygmaeus, GBIF's name. A check made before the species had
    another name is made again."""
    queries = []

    def blast(batch, entrez_query=None, **kwargs):
        queries.append(entrez_query)
        return pd.DataFrame()

    monkeypatch.setattr(hypo_pipeline, "submit_blast_batch", blast)
    species = "Phalacrocorax pygmaeus"
    filter_df = pd.DataFrame(
        {"seq_id": ["t1"], "scientificName": [species], "dna_sequence": ["ACGT"]}
    )
    name_table = gbif_name_rows(
        pd.DataFrame(
            {
                "scientificName": [species],
                "gbif_scientificName": ["Microcarbo pygmaeus"],
                "taxonID": ["9LDJ8"],
                "gbif_matched_by": ["WoRMS ID"],
                "aphia_id": [148763],
            }
        )
    )
    db_path = str(tmp_path / "hypo.db")

    run_hypo_check(filter_df, db_path=db_path)
    check_queries = hypo_check_queries([species], name_table)
    run_hypo_check(filter_df, check_queries=check_queries, db_path=db_path)

    assert queries == [
        '("Phalacrocorax pygmaeus"[orgn])',
        '("Phalacrocorax pygmaeus"[orgn] OR "Microcarbo pygmaeus"[orgn])',
    ]


def test_hypo_check_failed_species_does_not_evict_its_siblings(tmp_path, monkeypatch):
    """A check whose BLAST errored is retried; the other species checked for
    the same sequence stay cached."""

    def blast(batch, entrez_query=None, **kwargs):
        return None if "Gadus ogac" in entrez_query else pd.DataFrame()

    monkeypatch.setattr(hypo_pipeline, "submit_blast_batch", blast)
    filter_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t1"],
            "scientificName": ["Gadus morhua", "Gadus ogac"],
            "dna_sequence": ["ACGT", "ACGT"],
        }
    )
    db_path = str(tmp_path / "hypo.db")

    df, _ = run_hypo_check(filter_df, db_path=db_path)

    assert df.attrs["failed_items"] == [("t1", "Gadus ogac")]
    cached = view_cached_runs(db_path, "hypo_check")
    assert list(zip(cached["seq_id"], cached["scientificName"])) == [
        ("t1", "Gadus morhua")
    ]


def test_hypo_check_runs_checks_in_parallel(tmp_path, monkeypatch):
    """A slow check holds only its own thread: the next species is checked
    meanwhile."""
    other_done = threading.Event()

    def blast(batch, entrez_query=None, **kwargs):
        if "Gadus morhua" in entrez_query:  # slow until the other check ran
            assert other_done.wait(timeout=5), "the checks ran one at a time"
        else:
            other_done.set()
        return pd.DataFrame()

    monkeypatch.setattr(hypo_pipeline, "submit_blast_batch", blast)
    filter_df = pd.DataFrame(
        {
            "seq_id": ["t1", "t2"],
            "scientificName": ["Gadus morhua", "Gadus ogac"],
            "dna_sequence": ["ACGT", "TTGG"],
        }
    )

    run_hypo_check(filter_df, num_threads=2, db_path=str(tmp_path / "hypo.db"))

    assert other_done.is_set()
