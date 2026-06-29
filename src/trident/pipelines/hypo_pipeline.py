"""
HYPO pipeline — hypothetical species validation via proxy sequences.

Produces species that have no target marker in GenBank but are validated
via CO1 proxy (BOLD sequences BLASTed against NCBI) and confirmed as
geographically plausible.

Steps:
    1. prepare_hypo_input       — build per-seq_id BLAST inputs from EXTRA + GEO
    2. run_hypo_search          — NCBI BLAST on proxy sequences          [CustomCache]
    3. run_hypo_merge           — map proxy IDs back to targets (EXTRA)   [FullCache]
    4. run_hypo_filter          — threshold on identity / query cover     [FullCache]
    5. run_hypo_check           — per-species NCBI BLAST verification     [CustomCache]
    6. finalize_hypo_results    — label, merge check results, final df    [FullCache]
"""

import pandas as pd

from loguru import logger

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from trident.core.constants import (
    HYPO_EV_EXPONENT,
    HYPO_MAX_HITS,
    HYPO_IDENTITY_CUTOFF,
    HYPO_NTOP,
    HYPO_QUERY_COVER,
    HYPO_IDENTITY,
    HYPO_CHECK_EV_EXPONENT,
)
from trident.core.database import save_to_db, FullCache, CustomCache
from trident.core.utils import (
    preserve_sequence_order,
    notify_progress,
    group_species_by_flag,
    ensure_columns,
    top_hit_per_group,
)
from trident.clients.ncbi import (
    create_sequence_batches,
    run_blast_batches,
    process_blast_results,
    submit_blast_batch,
)


### DATA PREPARATION ###


def prepare_hypo_input(extra_df, geo_df):
    """Build per-seq_id BLAST inputs from EXTRA proxy sequences and GEO validation status.

    For each seq_id, creates SeqRecords from EXTRA sequences and an Entrez
    organism query from MOL-validated species to constrain the BLAST search.

    Args:
        extra_df: EXTRA DataFrame with proxy sequences (seq_id_extra, dna_sequence_extra).
        geo_df: GEO DataFrame with 'in_mol' flag per species.

    Returns:
        dict: {seq_id: {'sequences': [SeqRecord, ...], 'entrez_query': str}, ...}
    """
    mol_species_by_seq = group_species_by_flag(geo_df, "in_mol")

    input_data = {}
    for seq_id, group in extra_df.groupby("seq_id", observed=False):
        seq_records = [
            SeqRecord(Seq(seq.replace("-", "N")), id=seq_id_extra)
            for seq, seq_id_extra in zip(
                group["dna_sequence_extra"], group["seq_id_extra"]
            )
        ]

        # Build Entrez query from MOL-validated species for this seq_id
        species_list = mol_species_by_seq.get(seq_id, [])
        if species_list:
            formatted = " OR ".join(f'"{sp}"[orgn]' for sp in species_list)
            query_string = f"({formatted})"
        else:
            query_string = ""

        input_data[seq_id] = {"sequences": seq_records, "entrez_query": query_string}

    n_seq = sum(len(v["sequences"]) for v in input_data.values())
    logger.info(f"Prepared HYPO input: {n_seq} sequences to BLAST")
    return input_data


### SEARCH ###


def _hypo_search_params(
    sequences_dict,
    ev_exponent: int,
    max_hits: int,
    identity_cutoff: float,
    ntop: int,
    **kwargs,
):
    """Prepare parameters dict for HYPO search step."""
    params = {
        "hypo_ev_exponent": ev_exponent,
        "hypo_max_hits": max_hits,
        "hypo_identity_cutoff": identity_cutoff,
        "hypo_ntop": ntop,
    }
    inputs = [
        {"seq_id": s.id, "entrez_query": v["entrez_query"], **params}
        for v in sequences_dict.values()
        for s in v["sequences"]
    ]
    return inputs, params


def _hypo_search_rebuild(input_selected, original_inputs):
    """Rebuild inputs for HYPO search step."""
    selected_lookup = {
        (item["seq_id"], item["entrez_query"]) for item in input_selected
    }

    new_sequences_dict = {}
    for sample_id, data in original_inputs["sequences_dict"].items():
        matched_records = [
            rec
            for rec in data["sequences"]
            if (rec.id, data["entrez_query"]) in selected_lookup
        ]
        if matched_records:
            new_sequences_dict[sample_id] = {
                "sequences": matched_records,
                "entrez_query": data["entrez_query"],
            }

    new_inputs = original_inputs.copy()
    new_inputs["sequences_dict"] = new_sequences_dict
    return new_inputs


@save_to_db(
    "hypo_search",
    cache=CustomCache(
        prepare_fn=_hypo_search_params,
        rebuild_fn=_hypo_search_rebuild,
        match_map_dict={"seq_id": "seq_id", "entrez_query": "entrez_query"},
    ),
)
def run_hypo_search(
    sequences_dict,
    batch_size: int = 10,
    num_threads: int = 3,
    ev_exponent: int = HYPO_EV_EXPONENT,
    max_hits: int = HYPO_MAX_HITS,
    identity_cutoff: float = HYPO_IDENTITY_CUTOFF,
    ntop: int = HYPO_NTOP,
    progress_handler: dict | None = None,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """NCBI BLAST search on proxy sequences grouped by seq_id.

    Args:
        sequences_dict: {seq_id: {'sequences': [SeqRecord], 'entrez_query': str}}.
        batch_size: Sequences per NCBI batch.
        num_threads: Parallel threads.
        ev_exponent: Negative exponent for E-value threshold (e.g. 10 → 1e-10).
        max_hits: Maximum hits per query.
        identity_cutoff: Minimum identity percentage to retain.
        ntop: If > 0, keep only top N hits per query by identity.
        progress_handler: Optional progress tracker.
        failure_sink: If provided, seq_ids whose proxy BLAST errored are appended
            here so the cache layer retries them instead of caching them as empty.

    Returns:
        DataFrame with NCBI BLAST results for proxy sequences.
    """
    seq_word = "sequence" if len(sequences_dict) == 1 else "sequences"
    logger.info(f"Starting HYPO search: {len(sequences_dict)} {seq_word}")

    all_results = []
    for seq_id, seq_dict in sequences_dict.items():
        logger.info(f"Processing {seq_id} ({len(seq_dict['sequences'])} seqs)")
        batches = create_sequence_batches(seq_dict["sequences"], batch_size)
        results, failed_ids = run_blast_batches(
            batches,
            n_threads=num_threads,
            evalue=10 ** (-ev_exponent),
            hitlist_size=max_hits,
            entrez_query=seq_dict["entrez_query"],
            progress_handler=progress_handler,
        )
        # A proxy BLAST error fails this seq_id; report it so the search is
        # retried rather than cached as empty.
        if failed_ids and failure_sink is not None:
            failure_sink.append(seq_id)
        if not results.empty:
            results = process_blast_results(
                results, fasta_sequences=seq_dict["sequences"]
            )
            results["entrez_query"] = seq_dict["entrez_query"]
            all_results.append(results)

    if all_results:
        search_df = pd.concat(all_results, ignore_index=True)
        if identity_cutoff > 0:
            search_df = search_df[search_df["identity_percentage"] >= identity_cutoff]
        if ntop > 0:
            search_df = top_hit_per_group(
                search_df,
                keys=["seq_id", "scientificName"],
                sort=["identity_percentage", "query_cover"],
                n=ntop,
            )
    else:
        search_df = pd.DataFrame()

    logger.success(f"HYPO search completed: {len(search_df)} hits")
    return search_df


### MERGE ###


@save_to_db(
    table_name="hypo_merge",
    cache=FullCache(
        inherit_from=["hypo_search_params", "extra_params"],
        fingerprint_on=["hypo_search_df"],
    ),
)
@preserve_sequence_order("seq_id", "extra_df")
def run_hypo_merge(
    hypo_search_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    hypo_search_params: dict | None = None,
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """Map proxy sequence IDs back to target sequences via EXTRA.

    Joins BLAST search results (keyed by seq_id_extra) with the EXTRA
    DataFrame to recover the original seq_id, scientificName, and
    target dna_sequence.

    Args:
        hypo_search_df: Search results from run_hypo_search.
        extra_df: EXTRA DataFrame with proxy-to-target mapping.
        hypo_search_params: Params from run_hypo_search (for caching).
        extra_params: Params from finalize_extra_results (for caching).

    Returns:
        Merged DataFrame with BLAST stats and EXTRA metadata.
    """
    if hypo_search_params is None:
        hypo_search_params = {}
    if extra_params is None:
        extra_params = {}

    search_cols = [
        "seq_id",
        "scientificName",
        "query_cover",
        "identity_percentage",
        "hit_def",
        "hit_url",
    ]
    merge_df = extra_df.merge(
        hypo_search_df[search_cols],
        left_on="seq_id_extra",
        right_on="seq_id",
        suffixes=("", "_blast"),
    )

    # seq_id_blast is the duplicate from search (= seq_id_extra), drop it
    # scientificName_blast is the BLAST hit species (not the target species)
    merge_df = merge_df.rename(columns={"scientificName_blast": "scientificName_hit"})
    cols_to_drop = [col for col in merge_df.columns if col.endswith("_blast")]
    merge_df = merge_df.drop(columns=cols_to_drop)

    return merge_df


### FILTER ###


@save_to_db(
    table_name="hypo_filter",
    cache=FullCache(
        local={
            "hypo_query_cover": "query_cover",
            "hypo_identity": "identity",
        },
        inherit_from=["hypo_merge_params"],
    ),
)
@preserve_sequence_order("seq_id", "hypo_merge_df")
def run_hypo_filter(
    hypo_merge_df: pd.DataFrame,
    query_cover: float = HYPO_QUERY_COVER,
    identity: float = HYPO_IDENTITY,
    hypo_merge_params: dict | None = None,
) -> pd.DataFrame:
    """Filter HYPO merge results by query coverage and identity thresholds.

    Args:
        hypo_merge_df: Merged results from run_hypo_merge.
        query_cover: Minimum query coverage percentage.
        identity: Minimum identity percentage.
        hypo_merge_params: Params from run_hypo_merge (for caching).

    Returns:
        Filtered DataFrame keeping rows meeting both the query_cover and
        identity thresholds.
    """
    if hypo_merge_params is None:
        hypo_merge_params = {}

    return hypo_merge_df[
        (hypo_merge_df["query_cover"] >= query_cover)
        & (hypo_merge_df["identity_percentage"] >= identity)
    ].copy()


### FILTER SUMMARY ###


@preserve_sequence_order("seq_id", "hypo_filter_df")
def build_hypo_filter_summary(hypo_filter_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize filtered HYPO BLAST results: best hit and total count per species.

    For each (seq_id, scientificName) pair, returns the top hit by identity
    and query cover, plus the number of BLAST hits that passed filtering.

    Args:
        hypo_filter_df: Filtered results from run_hypo_filter.

    Returns:
        Summary DataFrame with columns: seq_id, scientificName, hit_count,
        top_identity_percentage, top_query_cover.
    """
    out_cols = [
        "seq_id",
        "scientificName",
        "hit_count",
        "identity_percentage",
        "query_cover",
    ]

    # Empty filter results come back from the cache with no columns at all.
    if hypo_filter_df.empty:
        return pd.DataFrame(columns=out_cols)

    best_hits = (
        hypo_filter_df.sort_values(
            ["identity_percentage", "query_cover"], ascending=[False, False]
        )
        .groupby(["seq_id", "scientificName"], observed=False, as_index=False)
        .first()
    )

    hit_counts = (
        hypo_filter_df.groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="hit_count")
    )

    summary = best_hits.merge(hit_counts, on=["seq_id", "scientificName"])

    return summary[out_cols].reset_index(drop=True)


### CHECK ###


def _hypo_check_params(
    hypo_filter_df: pd.DataFrame,
    ev_exponent: int,
    **kwargs,
):
    """Prepare parameters dict for HYPO check step."""
    sequences_df = hypo_filter_df[["seq_id", "scientificName"]].drop_duplicates()
    params = {
        "hypo_check_ev_exponent": ev_exponent,
    }
    inputs = [
        {
            "seq_id": row["seq_id"],
            "scientificName": row["scientificName"],
            **params,
        }
        for _, row in sequences_df.iterrows()
    ]
    return inputs, params


def _hypo_check_rebuild(input_selected, original_inputs):
    """Rebuild inputs for HYPO check step."""
    selected_lookup = {
        (entry["seq_id"], entry["scientificName"]) for entry in input_selected
    }

    filter_df = original_inputs["hypo_filter_df"]
    mask = pd.MultiIndex.from_frame(filter_df[["seq_id", "scientificName"]]).isin(
        selected_lookup
    )
    original_inputs = original_inputs.copy()
    original_inputs["hypo_filter_df"] = filter_df[mask]
    return original_inputs


@save_to_db(
    table_name="hypo_check",
    cache=CustomCache(
        prepare_fn=_hypo_check_params,
        rebuild_fn=_hypo_check_rebuild,
        match_map_dict={"seq_id": "seq_id", "scientificName": "scientificName"},
    ),
)
@preserve_sequence_order("seq_id", "hypo_filter_df")
def run_hypo_check(
    hypo_filter_df: pd.DataFrame,
    ev_exponent: int = HYPO_CHECK_EV_EXPONENT,
    progress_handler: dict | None = None,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """Per-species NCBI BLAST verification of HYPO candidates.

    BLASTs each target sequence against NCBI filtered to that species
    to confirm the species has the target marker in GenBank.

    Args:
        hypo_filter_df: Filtered DataFrame from run_hypo_filter.
        ev_exponent: Negative exponent for E-value threshold.
        progress_handler: Optional progress tracker.
        failure_sink: If provided, seq_ids whose BLAST errored are appended
            here so the cache layer retries them instead of caching them as a
            (false) no-hit.

    Returns:
        DataFrame with NCBI check results per species.
    """
    out_cols = [
        "seq_id",
        "scientificName",
        "dna_sequence",
        "scientificName_NCBI",
        "query_cover",
        "identity_percentage",
        "hit_found",
        "hit_def",
        "hit_url",
    ]

    # Empty filter results come back from the cache with no columns at all.
    if hypo_filter_df.empty:
        return pd.DataFrame(columns=out_cols)

    sequences_df = (
        hypo_filter_df[["seq_id", "scientificName", "dna_sequence"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(
        f"Starting HYPO check for {len(sequences_df)} sequences "
        f"(evalue=1e-{ev_exponent})"
    )

    all_results = []
    for index, row in sequences_df.iterrows():
        seq_id, sci_name, seq_dna = (
            row["seq_id"],
            row["scientificName"],
            row["dna_sequence"],
        )

        sequence = SeqRecord(Seq(seq_dna), id=seq_id)
        blast_res = submit_blast_batch(
            [sequence],
            entrez_query=f'"{sci_name}"[orgn]',
            evalue=10 ** (-ev_exponent),
            batch_id=index,
            hitlist_size=10,
        )

        if blast_res is None:
            # BLAST errored — report so this seq_id is retried, not recorded
            # as a (false) no-hit.
            if failure_sink is not None:
                failure_sink.append(seq_id)
            notify_progress(progress_handler)
            continue

        if blast_res.empty:
            result = pd.DataFrame(
                {
                    "seq_id": [seq_id],
                    "scientificName": [sci_name],
                    "dna_sequence": [seq_dna],
                    "hit_found": [False],
                }
            )
        else:
            ncbi_data = process_blast_results(blast_res, fasta_sequences=[sequence])
            result = pd.DataFrame(
                {
                    "seq_id": seq_id,
                    "scientificName": sci_name,
                    "dna_sequence": seq_dna,
                    "scientificName_NCBI": ncbi_data["scientificName"],
                    "query_cover": ncbi_data["query_cover"],
                    "identity_percentage": ncbi_data["identity_percentage"],
                    "hit_found": True,
                    "hit_def": ncbi_data["hit_def"],
                    "hit_url": ncbi_data["hit_url"],
                }
            )

        all_results.append(result)
        notify_progress(progress_handler)

    # Every sequence may have errored (all reported to failure_sink).
    if not all_results:
        return pd.DataFrame(columns=out_cols)

    check_df = pd.concat(all_results, ignore_index=True)

    n_with_hits = check_df.loc[check_df["hit_found"], "scientificName"].nunique()
    logger.info(
        f"HYPO check complete: {n_with_hits}/{len(sequences_df)} species with hits"
    )
    return check_df


### FINALIZE HYPO RESULTS (final) ###


@save_to_db(
    table_name="results_hypo",
    cache=FullCache(
        inherit_from=["hypo_check_params", "hypo_filter_params"],
        fingerprint_on=["hypo_check_df"],
    ),
)
@preserve_sequence_order("seq_id", "hypo_filter_df")
def finalize_hypo_results(
    hypo_filter_df: pd.DataFrame,
    hypo_check_df: pd.DataFrame,
    hypo_filter_params: dict | None = None,
    hypo_check_params: dict | None = None,
) -> pd.DataFrame:
    """Build final HYPO species list by merging check results into filtered data.

    Produces one row per (seq_id, scientificName) with the best NCBI check
    hit stats. Only species not already in MOL — the final output can be
    concatenated with geo_df to form the complete species list.

    Args:
        hypo_filter_df: Filtered results from run_hypo_filter (has taxonomy from GEO).
        hypo_check_df: Check results from run_hypo_check.
        hypo_filter_params: Params from run_hypo_filter (for caching).
        hypo_check_params: Params from run_hypo_check (for caching).

    Returns:
        HYPO species DataFrame with check stats (one row per seq_id × species).
    """
    if hypo_filter_params is None:
        hypo_filter_params = {}
    if hypo_check_params is None:
        hypo_check_params = {}

    # Best check hit per (seq_id, scientificName)
    check_hits = top_hit_per_group(
        hypo_check_df[hypo_check_df["hit_found"].astype(bool)],
        keys=["seq_id", "scientificName"],
        sort=["identity_percentage", "query_cover"],
        columns=[
            "seq_id",
            "scientificName",
            "identity_percentage",
            "query_cover",
            "hit_url",
        ],
    ).rename(
        columns={
            "identity_percentage": "ncbi_top_identity_percentage",
            "query_cover": "ncbi_top_query_cover",
            "hit_url": "ncbi_top_hit_url",
        }
    )

    # Keep best proxy hit per (seq_id, scientificName)
    hypo_df = top_hit_per_group(
        hypo_filter_df,
        keys=["seq_id", "scientificName"],
        sort=["identity_percentage", "query_cover"],
    ).reset_index(drop=True)

    # Merge check stats
    hypo_df = hypo_df.merge(check_hits, on=["seq_id", "scientificName"], how="left")

    logger.success(
        f"Finalized HYPO list: {len(hypo_df)} rows, "
        f"{hypo_df['scientificName'].nunique()} species"
    )
    return hypo_df


### CHECK SUMMARY ###


@preserve_sequence_order("seq_id", "hypo_check_df")
def build_hypo_check_summary(hypo_check_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize check results: best hit and count per (seq_id, scientificName).

    Args:
        hypo_check_df: Check results from run_hypo_check.

    Returns:
        Summary DataFrame with columns: seq_id, scientificName, hit_count,
        identity_percentage, query_cover, hit_url.
    """
    out_cols = [
        "seq_id",
        "scientificName",
        "hit_count",
        "identity_percentage",
        "query_cover",
        "hit_url",
    ]

    # Empty check results come back from the cache with no columns at all.
    if hypo_check_df.empty:
        return pd.DataFrame(columns=out_cols)

    # Ensure expected columns exist (may be absent when all results are no-hit)
    ensure_columns(hypo_check_df, ("identity_percentage", "query_cover", "hit_url"))

    best_hits = (
        hypo_check_df.sort_values(
            ["identity_percentage", "query_cover"], ascending=[False, False]
        )
        .groupby(["seq_id", "scientificName"], as_index=False, observed=False)
        .first()
    )

    hit_counts = (
        hypo_check_df[hypo_check_df["hit_found"].astype(bool)]
        .groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="hit_count")
    )

    summary = best_hits.merge(hit_counts, on=["seq_id", "scientificName"], how="left")
    summary["hit_count"] = summary["hit_count"].fillna(0).astype(int)

    return summary[out_cols]
