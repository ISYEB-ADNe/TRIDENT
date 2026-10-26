"""NCBI BLAST pipeline — produces the MOL (Molecular) results list.

Step order:
    1. prepare_ncbi_input(sequences_df)         → list[SeqRecord]
    2. run_ncbi_search(sequences, ...)           → (ncbi_search_df, params)
    3. run_ncbi_filter(ncbi_search_df, ...)       → (ncbi_filter_df, params)
    4. finalize_mol_results(ncbi_filter_df, ...)  → (mol_df, params)

Exploration:
    build_mol_summary(mol_df)
    build_mol_sequence_report(mol_df, seq_id, ncbi_search_df, ...)
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from Bio.SeqRecord import SeqRecord
from loguru import logger

from trident.clients.fasta import dataframe_to_records
from trident.clients.ncbi import (
    BLAST_GIVE_UP,
    BLAST_HIT_COLUMNS,
    create_sequence_batches,
    filter_blast_results,
    mol_filter_cut,
    process_blast_results,
    run_blast_batches,
    species_across_gap,
)
from trident.core.constants import (
    NCBI_EV_EXPONENT,
    NCBI_MAX_HITS,
    NCBI_QUERY_COVER,
    NCBI_GAP_SIZE,
    NCBI_METHOD,
    NCBI_GAP_MIN_TOP,
    NCBI_LOW_IDENTITY_THRESHOLD,
)
from trident.core.database import save_to_db, PartialCache, FullCache
from trident.core.utils import (
    StepLog,
    preserve_sequence_order,
    pluralize,
    species_for_sequences,
)


### DATA PREPARATION ###


def prepare_ncbi_input(sequences_df: pd.DataFrame) -> list[SeqRecord]:
    """Extract SeqRecord objects from a sequences DataFrame.

    Args:
        sequences_df: DataFrame produced by run_fasta_workflow().

    Returns:
        List of SeqRecord objects ready for BLAST submission.
    """
    return dataframe_to_records(sequences_df)


### NCBI SEARCH ###


@save_to_db(
    table_name="ncbi_search",
    cache=PartialCache(
        items_kwarg="sequences",
        item_key="seq_id",
        extract=lambda seq: seq.id,
        params={
            "ncbi_ev_exponent": "ev_exponent",
            "ncbi_max_hits": "max_hits",
        },
    ),
    log_as=("NCBI BLAST", "sequence"),
)
def run_ncbi_search(
    sequences: list[SeqRecord],
    batch_size: int = 50,
    num_threads: int = 3,
    ev_exponent: int = NCBI_EV_EXPONENT,
    max_hits: int = NCBI_MAX_HITS,
    give_up_minutes: float = BLAST_GIVE_UP / 60,
    progress_handler: Any | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Run BLASTn against the NCBI nt database and return processed hits.

    Splits sequences into batches and submits them in parallel. Results are
    enriched with identity metrics and standardized taxonomy before returning.

    Args:
        sequences: List of SeqRecord objects to query.
        batch_size: Number of sequences per BLAST request.
        num_threads: Batches run at once at normal pace, plus
            BLAST_SLOW_THREADS while jobs are slow (see ``run_blast_pool``).
        ev_exponent: Negative exponent for the E-value threshold
            (e.g. 20 → evalue=1e-20).
        max_hits: Maximum BLAST hits returned per query sequence.
        give_up_minutes: Minutes a BLAST job may wait at NCBI before it is
            resubmitted once, then counted as failed (see ``BlastGaveUp``).
        progress_handler: Optional progress tracker (dict, tqdm, or None).
        failure_sink: If provided, seq_ids whose BLAST batch errored are
            appended here so the cache layer retries them instead of caching
            them as empty.
        result_sink: Given by save_to_db, which saves the sequences of each
            batch as soon as its BLAST is done.

    Returns:
        DataFrame of processed BLAST hits with taxonomy and alignment metrics.
    """
    log = StepLog("NCBI BLAST")
    batches = create_sequence_batches(sequences, batch_size)
    logger.debug(
        f"NCBI BLAST: {len(batches)} {pluralize(len(batches), 'batch')} "
        f"of up to {batch_size} sequences"
    )
    hits_of: dict[int, pd.DataFrame] = {}  # processed hits of each batch

    def batch_done(i, dfs, failed):
        batch = batches[i]
        found = [df for df in dfs if not df.empty]
        raw = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
        hits_of[i] = process_blast_results(raw, fasta_sequences=batch)
        if result_sink is not None:
            result_sink(hits_of[i], [rec.id for rec in batch if rec.id not in failed])

    _, failed_ids = run_blast_batches(
        batches,
        n_threads=num_threads,
        evalue=10.0 ** (-ev_exponent),
        hitlist_size=max_hits,
        progress_handler=progress_handler,
        give_up=give_up_minutes * 60,
        on_batch=batch_done,
    )
    if failed_ids and failure_sink is not None:
        failure_sink.extend(failed_ids)

    found = [hits_of[i] for i in sorted(hits_of) if not hits_of[i].empty]
    ncbi_df = (
        pd.concat(found, ignore_index=True)
        if found
        else pd.DataFrame(columns=BLAST_HIT_COLUMNS)
    )

    log.done(
        f"{len(ncbi_df)} {pluralize(len(ncbi_df), 'hit')}",
        failed_ids,
        "sequence",
    )
    return ncbi_df


### NCBI FILTERING ###


@save_to_db(
    table_name="ncbi_filter",
    cache=FullCache(
        local={
            "ncbi_query_cover": "query_cover",
            "ncbi_gap_size": "gap_size",
            "ncbi_method": "method",
            "ncbi_gap_min_top": "gap_min_top",
        },
        inherit_from=["search_params"],
        fingerprint_on=["ncbi_search_df"],
    ),
    log_reuse_as="MOL filter",
)
@preserve_sequence_order("seq_id", "ncbi_search_df")
def run_ncbi_filter(
    ncbi_search_df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    gap_size: float = NCBI_GAP_SIZE,
    method: str = NCBI_METHOD,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
    search_params: dict | None = None,
) -> pd.DataFrame:
    """Filter NCBI search results by query coverage and identity gap method.

    Args:
        ncbi_search_df: DataFrame from run_ncbi_search().
        query_cover: Minimum query coverage percentage to retain a hit.
        gap_size: Minimum identity drop defining a barcoding gap, or maximum
            drop from the best hit when using similarity filtering.
        method: Either 'barcoding_gap' or 'similarity'.
        gap_min_top: Minimum identity for barcoding gap search (the top of the
            gap must be ≥ this value). Only used when method='barcoding_gap'.
        search_params: Parameter dict from run_ncbi_search, used for caching only.

    Returns:
        Filtered DataFrame.
    """
    if search_params is None:
        search_params = {}

    if method == "barcoding_gap":
        rule = (
            f"hits above a barcoding gap of at least {gap_size}% (gap top ≥ "
            f"{gap_min_top}%), else hits within {gap_size}% of the best"
        )
    else:
        rule = f"hits within {gap_size}% of the best"
    logger.info(f"MOL filter: query cover ≥ {query_cover}%, {rule}")

    filtered_df = filter_blast_results(
        ncbi_search_df,
        query_cover=query_cover,
        gap_size=gap_size,
        method=method,
        gap_min_top=gap_min_top,
    )

    # identity_drop serves the barcoding gap method only (absent with the
    # similarity method): the MOL list does not keep it.
    filtered_df = filtered_df.drop(columns="identity_drop", errors="ignore")

    n_seqs = filtered_df["seq_id"].nunique()
    logger.info(
        f"MOL filter: {len(filtered_df)} {pluralize(len(filtered_df), 'hit')} kept "
        f"for {n_seqs} {pluralize(n_seqs, 'sequence')}"
    )
    return filtered_df


### FINALIZE ###


@save_to_db(
    table_name="results_mol",
    cache=FullCache(
        local={
            "ncbi_low_identity_threshold": "threshold",
            "ncbi_enforce_low_identity": "enforce_threshold",
        },
        inherit_from=["filter_params"],
    ),
    log_reuse_as="MOL list",
    bool_columns=("low_identity_warning",),
)
@preserve_sequence_order("seq_id", "ncbi_filter_df")
def finalize_mol_results(
    ncbi_filter_df: pd.DataFrame,
    threshold: float = NCBI_LOW_IDENTITY_THRESHOLD,
    enforce_threshold: bool = False,
    filter_params: dict | None = None,
) -> pd.DataFrame:
    """Finalize the MOL list for export or downstream analysis.

    Adds a low_identity_warning flag for hits below the identity threshold.
    Optionally removes those hits entirely.

    Args:
        ncbi_filter_df: Filtered DataFrame from run_ncbi_filter().
        threshold: Identity percentage below which low_identity_warning is set.
        enforce_threshold: If True, remove hits below threshold instead of just warning.
        filter_params: Parameter dict from run_ncbi_filter, used for caching only.

    Returns:
        Finalized MOL DataFrame with a low_identity_warning column (low hits
        removed when enforce_threshold is True).
    """
    if filter_params is None:
        filter_params = {}

    mol_df = ncbi_filter_df.copy()
    mol_df["low_identity_warning"] = mol_df["identity_percentage"] < threshold

    if enforce_threshold:
        n_before = len(mol_df)
        mol_df = mol_df[~mol_df["low_identity_warning"]].reset_index(drop=True)
        n_removed = n_before - len(mol_df)
        if n_removed:
            logger.info(
                f"MOL list: {n_removed} {pluralize(n_removed, 'hit')} below "
                f"{threshold}% identity removed"
            )

    logger.success(
        f"MOL list: {species_for_sequences(mol_df)} "
        f"({len(mol_df)} {pluralize(len(mol_df), 'hit')})"
    )
    return mol_df


### SUMMARY / EXPLORATION ###


@preserve_sequence_order("seq_id", "sequences_df")
def build_ncbi_search_overview(
    ncbi_search_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    max_hits: int,
) -> pd.DataFrame:
    """Build a per-sequence overview of raw BLAST search results (before filtering).

    Flags sequences where max_hits was reached and the identity range is narrow,
    suggesting the search may need more hits to capture lower-identity matches.

    Args:
        ncbi_search_df: Raw BLAST DataFrame from run_ncbi_search().
        sequences_df: Original sequences DataFrame (for ordering and full seq_id list).
        max_hits: The max_hits parameter used in the search.

    Returns:
        DataFrame with one row per seq_id: hit count, identity range, and a
        saturation warning flag.
    """
    all_seq_ids = sequences_df["seq_id"].unique()
    identities = ncbi_search_df.groupby("seq_id")["identity_percentage"]
    stats = identities.agg(["size", "min", "max"])

    rows = []
    for seq_id in all_seq_ids:
        if seq_id not in stats.index:
            rows.append(
                {
                    "seq_id": seq_id,
                    "n_blast_hits": 0,
                    "min_identity": None,
                    "max_identity": None,
                    "identity_range": None,
                    "max_hits_reached": False,
                }
            )
            continue

        n_hits = stats.at[seq_id, "size"]
        min_id, max_id = stats.at[seq_id, "min"], stats.at[seq_id, "max"]
        identity_range = max_id - min_id

        rows.append(
            {
                "seq_id": seq_id,
                "n_blast_hits": n_hits,
                "min_identity": round(min_id, 2),
                "max_identity": round(max_id, 2),
                "identity_range": round(identity_range, 2),
                "max_hits_reached": n_hits >= max_hits,
            }
        )

    return pd.DataFrame(rows)


@preserve_sequence_order("seq_id", "sequences_df")
def build_mol_summary(
    mol_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    ncbi_filter_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-sequence summary table for the MOL list.

    Args:
        mol_df: Finalized MOL DataFrame from finalize_mol_results().
        sequences_df: Original sequences DataFrame (used for ordering and full seq_id list).
        ncbi_filter_df: Pre-finalization filter DataFrame, used to recover
            filter_method for sequences whose hits were all removed.

    Returns:
        DataFrame with one row per seq_id containing hit counts, species
        counts, top identity, filter method, and low_identity_warning: a
        species kept for the sequence has its best hit below the MOL list's
        low identity threshold.
    """
    present_ids = set(mol_df["seq_id"].unique())
    all_seq_ids = sequences_df["seq_id"].unique()

    # Build filter_method lookup from ncbi_filter_df for sequences removed during finalize
    filter_method_lookup: dict[str, str] = {}
    if ncbi_filter_df is not None:
        for sid in ncbi_filter_df["seq_id"].unique():
            rows = ncbi_filter_df[ncbi_filter_df["seq_id"] == sid]
            filter_method_lookup[sid] = rows["filter_method"].iloc[0]

    summary_data = []
    for seq_id in mol_df["seq_id"].unique():
        query_data = mol_df[mol_df["seq_id"] == seq_id]
        summary_data.append(
            {
                "seq_id": seq_id,
                "filter_method": query_data["filter_method"].iloc[0],
                "n_hits": len(query_data),
                "n_species": query_data["scientificName"].nunique(),
                "top_identity": round(query_data["identity_percentage"].max(), 2),
                # A species whose best hit is low: all its hits are low.
                "low_identity_warning": query_data["low_identity_warning"]
                .groupby(query_data["scientificName"], dropna=False)
                .all()
                .any(),
            }
        )

    # Add empty sequences (no hits after filtering)
    for seq_id in all_seq_ids:
        if seq_id not in present_ids:
            summary_data.append(
                {
                    "seq_id": seq_id,
                    "filter_method": filter_method_lookup.get(seq_id),
                    "n_hits": 0,
                    "n_species": 0,
                    "top_identity": None,
                    "low_identity_warning": False,
                }
            )

    return pd.DataFrame(summary_data)


def build_mol_sequence_report(
    mol_df: pd.DataFrame,
    seq_id: str,
    ncbi_search_df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    gap_size: float = NCBI_GAP_SIZE,
    method: str = NCBI_METHOD,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> dict:
    """Build a detailed per-sequence analysis report from the MOL list.

    The gap is located on all the sequence's BLAST hits with the MOL filter
    settings: the MOL list only holds the hits above it.

    Args:
        mol_df: Finalized MOL DataFrame from finalize_mol_results().
        seq_id: Sequence identifier to report on.
        ncbi_search_df: All BLAST hits, from run_ncbi_search().
        query_cover, gap_size, method, gap_min_top: The MOL filter settings
            (see run_ncbi_filter).

    Returns:
        Dict with keys:
        - 'seq_id': The sequence identifier.
        - 'sequence_data': DataFrame of hits for this sequence (with an n_hits column: hits of
          its species).
        - 'search_hits': All BLAST hits of this sequence.
        - 'cut': The filter's gap or range, from mol_filter_cut(); its
          is_gap says which method the filter used for this sequence.
        - 'species_across_gap': Output of species_across_gap(): species with
          hits on both sides of the gap (empty without a gap).
        - 'summary_stats': Dict with total_hits, unique_species, top_identity,
          and top_species.

    Raises:
        ValueError: If seq_id is not found in mol_df.
    """
    sequence_data = mol_df[mol_df["seq_id"] == seq_id].copy()

    if sequence_data.empty:
        raise ValueError(f"Sequence {seq_id!r} not found in MOL results")

    sequence_data["n_hits"] = sequence_data.groupby("scientificName")[
        "scientificName"
    ].transform("count")

    search_hits = ncbi_search_df[ncbi_search_df["seq_id"] == seq_id]
    cut = mol_filter_cut(
        search_hits,
        query_cover=query_cover,
        gap_size=gap_size,
        method=method,
        gap_min_top=gap_min_top,
    )

    return {
        "seq_id": seq_id,
        "sequence_data": sequence_data,
        "search_hits": search_hits,
        "cut": cut,
        "species_across_gap": species_across_gap(search_hits, cut, query_cover),
        "summary_stats": {
            "total_hits": len(sequence_data),
            "unique_species": sequence_data["scientificName"].nunique(),
            "top_identity": sequence_data["identity_percentage"].max(),
            "top_species": sequence_data.loc[
                sequence_data["identity_percentage"].idxmax(), "scientificName"
            ],
        },
    }
