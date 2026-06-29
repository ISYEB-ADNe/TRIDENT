"""BOLD pipeline — produces the EXTRA (new barcode sequences) results list.

Step order:
    1. prepare_bold_input(geo_df, ncbi_search_df)       → list[str]
    2. run_bold_search(species_list, ...)               → (bold_search_df, params)
    3. run_bold_filter(bold_search_df, ...)             → (bold_filter_df, params)
    4. run_bold_merge(bold_filter_df, geo_df, ...)      → (bold_merge_df, params)
    5. finalize_extra_results(bold_merge_df, ...)       → (extra_df, params)

Exploration:
    build_extra_summary(extra_df)
"""

import pandas as pd

from loguru import logger

from trident.core.utils import (
    reorder_taxonomy_columns,
    preserve_sequence_order,
    notify_progress,
    normalize_name,
)
from trident.core.constants import BOLD_SIMILARITY
from trident.core.database import save_to_db, PartialCache, FullCache, CustomCache
from trident.core.sequence_selection import select_longest_sequences

from trident.clients.bold import get_records_from_species_list


### DATA PREPARATION ###


def prepare_bold_input(
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None = None,
) -> tuple[list[str], int]:
    """Extract species needing BOLD querying.

    Returns species that have at least one (seq_id, scientificName) pair
    in ``geo_df`` not already covered by NCBI BLAST search results.  A species
    is only excluded when every one of its seq_id pairs was found in
    ``ncbi_search_df``.  Falls back to the ``in_mol`` column when
    ``ncbi_search_df`` is not provided.

    Args:
        geo_df: GEO DataFrame with 'seq_id', 'scientificName', and
            'in_mol' columns.
        ncbi_search_df: Optional NCBI BLAST search results DataFrame. When
            provided, (seq_id, scientificName) pairs found here are
            excluded from BOLD querying.

    Returns:
        Tuple of (species_list, n_excluded): unique species names to query
        in BOLD, and how many were excluded.
    """
    # All (seq_id, species) pairs in GEO
    geo_pairs = geo_df[["seq_id", "scientificName"]].drop_duplicates()

    if ncbi_search_df is not None and not ncbi_search_df.empty:
        # Remove pairs already covered by NCBI BLAST search
        ncbi_pairs = ncbi_search_df[["seq_id", "scientificName"]].drop_duplicates()
        candidates = geo_pairs.merge(
            ncbi_pairs, on=["seq_id", "scientificName"], how="left", indicator=True
        )
        candidates = candidates[candidates["_merge"] == "left_only"].drop(
            columns="_merge"
        )
    else:
        # Fallback: exclude species in final MOL list
        candidates = geo_df[~geo_df["in_mol"].astype(bool)][
            ["seq_id", "scientificName"]
        ].drop_duplicates()

    all_species = geo_pairs["scientificName"].unique()
    species_list = candidates["scientificName"].unique().tolist()
    n_excluded = len(all_species) - len(species_list)
    logger.info(
        f"Prepared BOLD input: {len(species_list)} species to query, "
        f"{n_excluded} excluded (already covered by NCBI BLAST)"
    )
    return species_list, n_excluded


### BOLD SEARCH ###


@save_to_db(
    table_name="bold_search",
    cache=PartialCache(
        items_kwarg="species_list",
        item_key="scientificName",
        params={
            "bold_keep_only_COI5P": "keep_only_COI5P",
            "bold_keep_ncbi": "keep_ncbi",
        },
    ),
)
def run_bold_search(
    species_list: list[str],
    keep_only_COI5P: bool = True,
    keep_ncbi: bool = False,
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """Retrieve barcode records from BOLD for a list of species.

    Args:
        species_list: Species names to query.
        keep_only_COI5P: If True, only retain COI-5P marker records.
        keep_ncbi: If True, retain records also found in NCBI.
        progress_handler: Optional progress tracker (dict or tqdm-like).
        user_agent: User-Agent header for HTTP requests.
        failure_sink: If provided, species whose query failed (e.g. network or
            403 errors) are appended here so the cache layer can skip them and
            retry on the next run instead of caching them as empty.

    Returns:
        DataFrame with BOLD records.
    """
    logger.info(f"Starting BOLD search: {len(species_list)} unique species to query")

    bold_search_df, failed_species = get_records_from_species_list(
        species_list,
        keep_only_COI5P=keep_only_COI5P,
        keep_ncbi=keep_ncbi,
        progress_handler=progress_handler,
        user_agent=user_agent,
    )

    # Remove records with missing sequences (rare cases)
    if not bold_search_df.empty:
        bold_search_df = bold_search_df.dropna(subset=["dna_sequence"])

    n_failed = len(failed_species)
    if n_failed:
        logger.warning(f"{n_failed} species failed in BOLD search: {failed_species}")
        if failure_sink is not None:
            failure_sink.extend(failed_species)

    logger.info(f"Completed BOLD search: {len(bold_search_df)} records retrieved")

    return bold_search_df


### BOLD FILTER ###


def _bold_filter_params(
    bold_search_df: pd.DataFrame,
    longest_n: int | None,
    similarity: float,
    bold_search_params: dict | None = None,
    **kwargs,
):
    """Extract per-species cache entries with filter parameters."""
    params = {
        **(bold_search_params or {}),
        "bold_longest_n": longest_n,
        "bold_similarity": similarity,
    }
    inputs = [
        {"scientificName": species, **params}
        for species in bold_search_df["scientificName"].unique()
    ]
    return inputs, params


def _bold_filter_rebuild(input_selected, original_inputs):
    """Filter the DataFrame to only uncached species."""
    species_to_keep = {item["scientificName"] for item in input_selected}
    return original_inputs.copy() | {
        "bold_search_df": original_inputs["bold_search_df"][
            original_inputs["bold_search_df"]["scientificName"].isin(species_to_keep)
        ]
    }


@save_to_db(
    table_name="bold_filter",
    cache=CustomCache(
        prepare_fn=_bold_filter_params,
        rebuild_fn=_bold_filter_rebuild,
        match_map_dict={"scientificName": "scientificName"},
    ),
)
def run_bold_filter(
    bold_search_df: pd.DataFrame,
    longest_n: int | None = None,
    similarity: float = BOLD_SIMILARITY,
    bold_search_params: dict | None = None,
    progress_handler: dict | None = None,
) -> pd.DataFrame:
    """Filter redundant BOLD sequences, keeping the longest per species.

    Args:
        bold_search_df: DataFrame with 'scientificName' and 'dna_sequence' columns.
        longest_n: Max sequences to keep per species (None = keep all non-redundant).
        similarity: Percent identity threshold above which sequences are redundant.
        bold_search_params: Parameter dict from run_bold_search, propagated into
            this step's cache key (caching only).
        progress_handler: Optional progress tracker (dict or tqdm-like).

    Returns:
        Filtered DataFrame with representative sequences per species.
    """

    msg = "Filtering BOLD sequences using 'longest' method"
    if longest_n is not None:
        msg += f" (keeping up to {longest_n} longest per species)"
    else:
        msg += " (keeping all non-redundant per species)"
    logger.info(msg)

    filtered_dfs = []
    total_original = len(bold_search_df)

    for species, species_df in bold_search_df.groupby("scientificName", sort=False):
        try:
            if len(species_df) < 2:
                filtered_dfs.append(species_df)
                continue

            seqs = species_df["dna_sequence"].astype(str).str.strip().tolist()
            rep_indices, _ = select_longest_sequences(
                seqs,
                n_longest=longest_n,
                identity_threshold=similarity,
            )

            global_indices = species_df.index[rep_indices]
            filtered_dfs.append(bold_search_df.loc[global_indices])
        except Exception as e:
            logger.opt(exception=True).warning(
                f"Filtering failed for {species}, keeping all: {e}"
            )
            filtered_dfs.append(species_df)
        finally:
            notify_progress(progress_handler)

    if not filtered_dfs:
        return bold_search_df.iloc[0:0].copy()
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    reduction = (1 - len(result_df) / total_original) * 100

    logger.info(
        f"Completed BOLD filtering: {total_original} → {len(result_df)} sequences "
        f"({reduction:.1f}% reduction)"
    )

    return result_df


### BOLD MERGE ###


@save_to_db(
    table_name="bold_merge",
    cache=FullCache(
        inherit_from=["bold_filter_params", "geo_params"],
        fingerprint_on=["bold_filter_df"],
    ),
)
@preserve_sequence_order("seq_id", "geo_df")
def run_bold_merge(
    bold_filter_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    bold_filter_params: dict | None = None,
    geo_params: dict | None = None,
) -> pd.DataFrame:
    """Merge filtered BOLD results with GEO records by scientific name.

    Args:
        bold_filter_df: DataFrame with deduplicated BOLD sequences
            including 'scientificName', 'seq_id', 'dna_sequence'.
        geo_df: GEO DataFrame with 'scientificName', 'seq_id',
            'dna_sequence', and taxonomy columns.
        bold_filter_params: Parameters from run_bold_filter (for caching).
        geo_params: Parameters from finalize_geo_results (for caching).

    Returns:
        Merged DataFrame with GEO sequences joined to matching BOLD sequences.
        Only rows with BOLD matches are kept; taxonomy columns come from TAX.
    """
    if bold_filter_params is None:
        bold_filter_params = {}
    if geo_params is None:
        geo_params = {}

    # Select only needed GEO columns
    geo_cols = [
        "seq_id",
        "dna_sequence",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "specificEpithet",
        "scientificName",
        "scientificNameAuthorship",
        "taxonRank",
        "taxonID",
        "taxonID_db",
        "taxonURL",
    ]
    missing = [c for c in geo_cols if c not in geo_df.columns]
    if missing:
        raise KeyError(f"Missing columns in geo_df: {missing}")

    # Merge GEO (left) with filtered BOLD on a normalized name key (case /
    # whitespace insensitive) so spelling differences do not drop species.
    geo_left = geo_df[geo_cols].assign(
        _join_key=geo_df["scientificName"].map(normalize_name)
    )
    bold_right = bold_filter_df.assign(
        _join_key=bold_filter_df["scientificName"].map(normalize_name)
    )
    bold_keys = set(bold_right["_join_key"].dropna())
    # Only species that needed a BOLD proxy (not already covered by a direct NCBI
    # hit) can be genuine misses. NCBI-covered (in_mol) species are excluded from
    # the BOLD query upstream, so "no BOLD match" for them is expected, not a miss.
    candidate = geo_df.assign(_join_key=geo_df["scientificName"].map(normalize_name))
    if "in_mol" in candidate.columns:
        candidate = candidate[~candidate["in_mol"].astype(bool)]
    unmatched = sorted(
        candidate.loc[~candidate["_join_key"].isin(bold_keys), "scientificName"]
        .dropna()
        .unique()
    )
    if unmatched:
        logger.warning(
            f"{len(unmatched)} queried species had no BOLD match: {unmatched[:20]}"
        )

    merged_df = geo_left.merge(
        bold_right,
        on="_join_key",
        how="left",
        suffixes=("", "_bold"),
        validate="m:m",  # many GEO per species, many BOLD per species allowed
    ).drop(columns="_join_key")

    # Keep only rows where a BOLD sequence was found
    merged_df = merged_df.dropna(subset=["seq_id_bold"]).reset_index(drop=True)

    # Standardize taxonomy column order
    merged_df = reorder_taxonomy_columns(merged_df)

    # Drop redundant columns
    cols_to_drop = [
        col
        for col in merged_df.columns
        if col.endswith("_bold") and col not in ["seq_id_bold", "dna_sequence_bold"]
    ]
    merged_df = merged_df.drop(columns=cols_to_drop)
    result_df = merged_df.drop_duplicates(keep="first").reset_index(drop=True)

    logger.info(f"Completed BOLD merge: {len(result_df)} rows")
    return result_df


### FINALIZE EXTRA RESULTS ###


@save_to_db(
    table_name="results_extra",
    cache=FullCache(inherit_from=["bold_merge_params"]),
)
@preserve_sequence_order("seq_id", "bold_merge_df")
def finalize_extra_results(
    bold_merge_df: pd.DataFrame, bold_merge_params: dict | None = None
) -> pd.DataFrame:
    """Finalize EXTRA (new BOLD sequences) output for export or analysis.

    Args:
        bold_merge_df: Merged BOLD DataFrame from run_bold_merge().
        bold_merge_params: Parameters from run_bold_merge (for caching).

    Returns:
        Finalized EXTRA DataFrame.
    """
    if bold_merge_params is None:
        bold_merge_params = {}

    extra_df = bold_merge_df.rename(
        columns={
            "seq_id_bold": "seq_id_extra",
            "dna_sequence_bold": "dna_sequence_extra",
        }
    )
    logger.success(f"Finalized EXTRA list: {len(extra_df)} rows")
    return extra_df


### EXPLORATION / SUMMARY ###


@preserve_sequence_order("seq_id", "extra_df")
def build_extra_summary(
    extra_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    bold_input_species: list[str] | None = None,
) -> pd.DataFrame:
    """Build per-(seq_id, species) table showing BOLD record counts.

    Creates a skeleton of all (seq_id, species) pairs that were queried
    in BOLD, then left-joins the actual record counts. Pairs with zero
    records indicate species queried but not found.

    Args:
        extra_df: Finalized EXTRA DataFrame (from finalize_extra_results).
        geo_df: GEO DataFrame — used to build the full set of queried pairs.
        bold_input_species: Species sent to BOLD (from prepare_bold_input).
            Falls back to ``geo_df[~in_mol]`` when not provided.

    Returns:
        DataFrame with columns: seq_id, scientificName, total_records.
    """
    # Records per (seq_id, scientificName) pair from EXTRA results
    extra_stats = (
        extra_df.groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="total_records")
    )

    # Full skeleton of queried (seq_id, species) pairs
    if bold_input_species is not None:
        queried_df = geo_df[geo_df["scientificName"].isin(bold_input_species)]
    else:
        queried_df = geo_df[~geo_df["in_mol"].astype(bool)]

    all_pairs = queried_df[["seq_id", "scientificName"]].drop_duplicates()

    # Left-merge: 0 records for species not found in BOLD
    summary = all_pairs.merge(extra_stats, on=["seq_id", "scientificName"], how="left")
    summary["total_records"] = summary["total_records"].fillna(0).astype(int)

    return summary


def build_extra_seq_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(seq_id, species) summary into one row per seq_id.

    Args:
        pair_summary: Output of build_extra_summary (seq_id, scientificName,
            total_records).

    Returns:
        DataFrame with columns: seq_id, total_records, species_found,
        species_missing.
    """
    return (
        pair_summary.groupby("seq_id", observed=False)
        .agg(
            total_records=("total_records", "sum"),
            species_queried=("total_records", "size"),
            species_found=("total_records", lambda x: (x > 0).sum()),
            species_missing=("total_records", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )
