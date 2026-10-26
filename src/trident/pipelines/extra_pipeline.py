"""BOLD pipeline — produces the EXTRA (new barcode sequences) results list.

Step order:
    1. prepare_bold_input(geo_df, ncbi_search_df)       → list[str]
       prepare_bold_other_names(species_list, name_table) → dict
    2. run_bold_search(species_list, other_names, ...)  → (bold_search_df, params)
    3. run_bold_filter(bold_search_df, ...)             → (bold_filter_df, params)
    4. run_bold_merge(bold_filter_df, geo_df, ...)      → (bold_merge_df, params)
    5. finalize_extra_results(bold_merge_df, ...)       → (extra_df, params)

Exploration:
    build_extra_summary(extra_df)
"""

from collections.abc import Callable

import pandas as pd

from loguru import logger

from trident.core.utils import (
    name_list,
    pluralize,
    species_for_sequences,
    preserve_sequence_order,
    notify_progress,
    normalize_name,
)
from trident.core.constants import BOLD_SIMILARITY
from trident.core.database import save_to_db, PartialCache, FullCache, CustomCache
from trident.core.sequence_selection import select_longest_sequences

from trident.clients.bold import get_records_from_species_list, with_sequences
from trident.clients.worms import WORMS_TAXON_COLUMNS
from trident.pipelines.names import found_by_blast, other_names as names_besides


### DATA PREPARATION ###


def needs_proxy(
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> pd.Series:
    """Which GEO rows need a BOLD proxy: True for each (seq_id, species) pair
    NCBI BLAST did not find directly.

    A pair is covered when MOL kept it (``in_mol``) or when BLAST returned the
    species for that sequence under any of its names, even in a hit the MOL
    filter rejected (see found_by_blast). The BOLD query, the merge of its
    records and the EXTRA summaries all use this rule, so a species queried
    for one sequence brings no proxy to a sequence where NCBI found it.

    Returns:
        Boolean Series aligned on geo_df's index.
    """
    needed = ~found_by_blast(geo_df, ncbi_search_df, name_table)
    if "in_mol" in geo_df.columns:
        needed &= ~geo_df["in_mol"]
    return needed


def prepare_bold_input(
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> tuple[list[str], int]:
    """Extract species needing BOLD querying.

    Returns the species with at least one (seq_id, species) pair that needs a
    proxy (see needs_proxy): a species is only excluded when NCBI BLAST found
    it for every one of its sequences.

    Args:
        geo_df: GEO DataFrame with 'seq_id', 'scientificName', and
            'in_mol' columns.
        ncbi_search_df: Optional NCBI BLAST search results DataFrame. When
            provided, (seq_id, species) pairs found here are excluded from
            BOLD querying.
        name_table: See build_name_table.

    Returns:
        Tuple of (species_list, n_excluded): unique species names to query
        in BOLD, and how many were excluded.
    """
    candidates = geo_df[needs_proxy(geo_df, ncbi_search_df, name_table)]
    all_species = geo_df["scientificName"].unique()
    species_list = candidates["scientificName"].unique().tolist()
    n_excluded = len(all_species) - len(species_list)
    logger.debug(
        f"Prepared BOLD input: {len(species_list)} species to query, "
        f"{n_excluded} excluded (already covered by NCBI BLAST)"
    )
    return species_list, n_excluded


# Sources of the other names a species is searched under in BOLD, in the
# order they are tried: BOLD's taxonomy often follows NCBI or Catalogue of
# Life (GBIF) where they differ from WoRMS (Larus pipixcan is Leucophaeus
# pipixcan in BOLD).
BOLD_OTHER_NAME_SOURCES = ["ncbi", "gbif"]


def prepare_bold_other_names(
    species_list: list[str], name_table: pd.DataFrame | None
) -> dict[str, list[tuple[str, str]]]:
    """Other names of each species to query in BOLD when its own name gives
    no usable sequence (see other_names)."""
    return names_besides(species_list, name_table, BOLD_OTHER_NAME_SOURCES)


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
        result_col="dna_sequence",
    ),
    log_as=("BOLD search", "species"),
)
def run_bold_search(
    species_list: list[str],
    other_names: dict[str, list[tuple[str, str]]] | None = None,
    keep_only_COI5P: bool = True,
    keep_ncbi: bool = False,
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Retrieve barcode records from BOLD for a list of species.

    Args:
        species_list: Species names to query.
        other_names: See prepare_bold_other_names. Not part of the cache
            key: a species cached without a sequence is queried under its
            other names by "Retry empty results".
        keep_only_COI5P: If True, only retain COI-5P marker records.
        keep_ncbi: If True, retain records also found in NCBI.
        progress_handler: Optional progress tracker (dict or tqdm-like).
        user_agent: User-Agent header for HTTP requests.
        failure_sink: If provided, species whose query failed (e.g. network or
            403 errors) are appended here so the cache layer can skip them and
            retry on the next run instead of caching them as empty.
        result_sink: Given by save_to_db, which saves each species as soon as
            its query is done.

    Returns:
        DataFrame with the BOLD sequences of each species, or one row with
        the reason it has none (see get_records_from_species_list).
    """
    bold_search_df, failed_species = get_records_from_species_list(
        species_list,
        other_names,
        keep_only_COI5P=keep_only_COI5P,
        keep_ncbi=keep_ncbi,
        progress_handler=progress_handler,
        user_agent=user_agent,
        result_sink=result_sink,
    )

    if failed_species and failure_sink is not None:
        failure_sink.extend(failed_species)

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
        for species in with_sequences(bold_search_df)["scientificName"].unique()
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
    log_reuse_as="BOLD filter",
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
        bold_search_df: Output of run_bold_search; its rows without a
            sequence are left out.
        longest_n: Max sequences to keep per species (None = keep all non-redundant).
        similarity: Percent identity threshold above which sequences are redundant.
        bold_search_params: Parameter dict from run_bold_search, propagated into
            this step's cache key (caching only).
        progress_handler: Optional progress tracker (dict or tqdm-like).

    Returns:
        Filtered DataFrame with representative sequences per species.
    """

    kept = f"the {longest_n} longest" if longest_n is not None else "all"
    logger.info(
        f"BOLD filter: per species, {kept} BOLD sequences less than {similarity}% "
        "identical to a longer one"
    )

    bold_search_df = with_sequences(bold_search_df)
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
        f"BOLD filter: {len(result_df)}/{total_original} BOLD sequences kept "
        f"({reduction:.0f}% fewer)"
    )

    return result_df


### BOLD MERGE ###

# The proxy's columns, as the BOLD search names them, and as EXTRA and HYPO
# name them: in the other steps seq_id and dna_sequence are the (M)OTU/ASV's.
PROXY_COLUMNS = {
    "seq_id": "proxy_id",
    "dna_sequence": "proxy_sequence",
    "seq_url": "proxy_url",
}

# The GEO list's columns the EXTRA list keeps: the candidate and its TAX
# taxonomy.
GEO_KEPT_COLUMNS = [
    "seq_id",
    "dna_sequence",
    *(col for col in WORMS_TAXON_COLUMNS if col != "genusID"),
]
# The columns of the EXTRA list: each candidate that needs a proxy, with one
# of its BOLD sequences.
EXTRA_COLUMNS = [*GEO_KEPT_COLUMNS, *PROXY_COLUMNS.values()]


@save_to_db(
    table_name="bold_merge",
    cache=FullCache(
        inherit_from=["bold_filter_params", "geo_params"],
        fingerprint_on=["bold_filter_df"],
    ),
    log_reuse_as="EXTRA list",
)
@preserve_sequence_order("seq_id", "geo_df")
def run_bold_merge(
    bold_filter_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
    bold_filter_params: dict | None = None,
    geo_params: dict | None = None,
) -> pd.DataFrame:
    """Merge filtered BOLD results with GEO records by scientific name.

    Args:
        bold_filter_df: DataFrame with deduplicated BOLD sequences
            including 'scientificName' and PROXY_COLUMNS.
        geo_df: GEO DataFrame with 'scientificName', 'seq_id',
            'dna_sequence', and taxonomy columns.
        ncbi_search_df: NCBI BLAST search results, for the pairs that need a
            proxy (see needs_proxy); the in_mol flag is used without it.
        name_table: See build_name_table.
        bold_filter_params: Parameters from run_bold_filter (for caching).
        geo_params: Parameters from finalize_geo_results (for caching).

    Returns:
        Merged DataFrame with GEO sequences joined to matching BOLD sequences
        (proxy_id, proxy_sequence, proxy_url). Only rows with BOLD
        matches are kept, and only for the pairs that need a proxy; taxonomy
        columns come from TAX.
    """
    if bold_filter_params is None:
        bold_filter_params = {}
    if geo_params is None:
        geo_params = {}

    missing = [c for c in GEO_KEPT_COLUMNS if c not in geo_df.columns]
    if missing:
        raise KeyError(f"Missing columns in geo_df: {missing}")

    geo_df = geo_df[needs_proxy(geo_df, ncbi_search_df, name_table)]

    # Join GEO and the filtered BOLD sequences on a normalized name key (case
    # / whitespace insensitive) so spelling differences do not drop species.
    # Of BOLD, only the proxy record joins, under its EXTRA names: the
    # taxonomy comes from TAX.
    geo_left = geo_df[GEO_KEPT_COLUMNS].assign(
        _join_key=geo_df["scientificName"].map(normalize_name)
    )
    bold_right = (
        bold_filter_df[["scientificName", *PROXY_COLUMNS]]
        .rename(columns=PROXY_COLUMNS)
        .assign(_join_key=lambda df: df["scientificName"].map(normalize_name))
        .drop(columns="scientificName")
    )
    bold_keys = set(bold_right["_join_key"].dropna())
    unmatched = sorted(
        geo_left.loc[~geo_left["_join_key"].isin(bold_keys), "scientificName"]
        .dropna()
        .unique()
    )
    if unmatched:
        logger.info(
            f"EXTRA list: {len(unmatched)} {pluralize(len(unmatched), 'species')} "
            f"without BOLD sequences ({name_list(unmatched)})"
        )

    # Only the pairs with a BOLD sequence; many GEO rows and many BOLD
    # records per species.
    merged_df = geo_left.merge(
        bold_right, on="_join_key", how="inner", validate="m:m"
    ).drop(columns="_join_key")
    result_df = merged_df.drop_duplicates(keep="first").reset_index(drop=True)

    logger.debug(f"Completed BOLD merge: {len(result_df)} rows")
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

    extra_df = bold_merge_df.copy()
    n_bold = extra_df["proxy_id"].nunique()
    logger.success(
        f"EXTRA list: {species_for_sequences(extra_df)} "
        f"({n_bold} BOLD {pluralize(n_bold, 'sequence')})"
    )
    return extra_df


### EXPLORATION / SUMMARY ###


@preserve_sequence_order("seq_id", "extra_df")
def build_extra_summary(
    extra_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build per-(seq_id, species) table showing BOLD record counts.

    Creates a skeleton of all (seq_id, species) pairs that were queried
    in BOLD, then left-joins the actual record counts. Pairs with zero
    records indicate species queried but not found.

    Args:
        extra_df: Finalized EXTRA DataFrame (from finalize_extra_results).
        geo_df: GEO DataFrame, for the pairs that needed a proxy.
        ncbi_search_df: NCBI BLAST search results (see needs_proxy).
        name_table: See build_name_table.

    Returns:
        DataFrame with columns: seq_id, scientificName, n_bold_sequences (the
        pair's BOLD sequences kept by the filter).
    """
    # Records per (seq_id, scientificName) pair from EXTRA results
    extra_stats = (
        extra_df.groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="n_bold_sequences")
    )

    # Full skeleton of queried (seq_id, species) pairs
    queried_df = geo_df[needs_proxy(geo_df, ncbi_search_df, name_table)]

    all_pairs = queried_df[["seq_id", "scientificName"]].drop_duplicates()

    # Left-merge: 0 records for species not found in BOLD
    summary = all_pairs.merge(extra_stats, on=["seq_id", "scientificName"], how="left")
    summary["n_bold_sequences"] = summary["n_bold_sequences"].fillna(0).astype(int)

    return summary


def build_extra_seq_summary(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(seq_id, species) summary into one row per seq_id.

    Args:
        pair_summary: Output of build_extra_summary (seq_id, scientificName,
            n_bold_sequences).

    Returns:
        DataFrame with columns: seq_id, n_bold_sequences, n_species_queried,
        n_species_found, n_species_missing.
    """
    return (
        pair_summary.groupby("seq_id", observed=False)
        .agg(
            n_bold_sequences=("n_bold_sequences", "sum"),
            n_species_queried=("n_bold_sequences", "size"),
            n_species_found=("n_bold_sequences", lambda x: (x > 0).sum()),
            n_species_missing=("n_bold_sequences", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )


### EMPTY INPUT ###


def empty_extra_results(geo_df: pd.DataFrame) -> pd.DataFrame:
    """Return the empty EXTRA list, with its usual columns, for no BOLD query.

    The EXTRA steps run on no species and without the database, so nothing is
    queried or cached.
    """
    search_df, _ = run_bold_search([], db_path=None)
    filter_df, _ = run_bold_filter(search_df, db_path=None)
    # No species is sent to BOLD, so none of the GEO species is merged; the
    # empty GEO frame only brings its columns.
    merge_df, _ = run_bold_merge(
        filter_df, geo_df.iloc[:0], None, None, geo_params={}, db_path=None
    )
    extra_df, _ = finalize_extra_results(merge_df, db_path=None)
    return extra_df
