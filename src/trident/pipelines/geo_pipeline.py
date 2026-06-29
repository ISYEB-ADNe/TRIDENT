"""GBIF pipeline — produces the GEO (Geographic scope) results list.

Step order:
    1. prepare_gbif_input(tax_df)                      → list[str]
    2. run_gbif_search(species_list, ...)               → (gbif_search_df, params)
    3. run_gbif_merge(gbif_search_df, tax_df, ...)      → (gbif_merge_df, params)
    4. run_gbif_filter(gbif_merge_df, ...)              → (gbif_filter_df, params)
    5. finalize_geo_results(gbif_filter_df, ...)        → (geo_df, params)

Exploration:
    build_geo_summary(gbif_merge_df)
"""

from collections.abc import Callable

import pandas as pd

from loguru import logger

from trident.clients.gbif import (
    match_taxon_names,
    filter_taxon_matches,
    counts_for_area,
    get_extent_column_name,
)

from trident.core.constants import GBIF_MIN_OCCURRENCES
from trident.core.database import save_to_db, FullCache, CustomCache, warn_empty_params
from trident.core.utils import (
    reorder_taxonomy_columns,
    preserve_sequence_order,
    top_hit_per_group,
    normalize_name,
)


### DATA PREPARATION ###


def prepare_gbif_input(df: pd.DataFrame) -> list[str]:
    """Return a list of unique scientific names for GBIF queries.

    Args:
        df: DataFrame containing a 'scientificName' column.

    Returns:
        List of unique, non-null scientific names to use as GBIF query taxa.
    """

    species_list = df["scientificName"].dropna().unique().tolist()
    logger.debug(f"Prepared GBIF input: {len(species_list)} unique species")
    return species_list


### SEARCH WORKFLOW ###


def _gbif_search_params(
    species_list: list[str],
    latitude: float | None,
    longitude: float | None,
    extents: list[float | None],
    **kwargs,
):
    """Extract parameters for GBIF search step."""
    params = {
        "gbif_latitude": latitude,
        "gbif_longitude": longitude,
        "gbif_extents": extents,
    }
    inputs = [
        {
            "species": species,
            "gbif_latitude": None if extent == "global" else latitude,
            "gbif_longitude": None if extent == "global" else longitude,
            "gbif_extent": extent,
        }
        for species in species_list
        for extent in extents
    ]
    return inputs, params


def _gbif_search_params_rebuild(input_selected, original_inputs):
    """Rebuild GBIF search parameters from stored kwargs."""

    species_to_keep = {item["species"] for item in input_selected}
    extents_to_keep = {item["gbif_extent"] for item in input_selected}
    inputs = original_inputs.copy() | {
        "species_list": list(species_to_keep),
        "extents": list(extents_to_keep),
    }
    return inputs


@save_to_db(
    table_name="gbif_search",
    cache=CustomCache(
        prepare_fn=_gbif_search_params,
        rebuild_fn=_gbif_search_params_rebuild,
        match_map_dict={"species": "scientificName", "gbif_extent": "gbif_extent"},
    ),
)
def run_gbif_search(
    species_list: list[str],
    latitude: float | None = None,
    longitude: float | None = None,
    extents: list[float | str] | None = None,
    progress_handler: Callable[[float, str], None] | None = None,
    user_agent: str | None = None,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """Run GBIF occurrence search for a list of taxa.

    Args:
        species_list: Scientific names to match in GBIF.
        latitude: Center latitude for bounding box calculation.
        longitude: Center longitude for bounding box calculation.
        extents: List of distances (km) from center to calculate bounding boxes, or "global".
        progress_handler: Optional callback for UI progress updates.
        user_agent: User-Agent header for HTTP requests.
        failure_sink: If provided, species whose name match errored (network or
            403, ``matchType == "ERROR"``) are appended here so the cache layer
            skips them and retries on the next run. A genuine no-match (needs
            review) is a valid empty, not a failure.

    Returns:
        DataFrame with GBIF taxon metadata and occurrence count columns.
    """

    # 1. Taxon Matching (0% -> 30%)
    if progress_handler:
        progress_handler(0.1, "Matching names against GBIF Backbone...")

    matches_df = match_taxon_names(species_list, user_agent=user_agent)

    ## Errored matches (network/403) are retry-worthy failures, distinct from
    ## genuine no-matches that need review.
    if "matchType" in matches_df.columns:
        errored = matches_df[matches_df["matchType"] == "ERROR"]["query"].tolist()
        if errored:
            logger.warning(f"{len(errored)} GBIF taxa errored during matching")
            if failure_sink is not None:
                failure_sink.extend(errored)

    ## Log no-matches needing review
    failed_matches = matches_df[
        matches_df["needs_review"] | matches_df["taxonID"].isna()
    ]["query"].tolist()
    n_failed = len(failed_matches)
    if n_failed:
        logger.warning(
            f"{n_failed} GBIF taxa need review or failed to match",
            failed_taxa=failed_matches,
        )

    ## Filter to valid keys
    good_matches = filter_taxon_matches(matches_df)
    taxon_keys = good_matches["taxonID"].astype(str).tolist()

    results = []

    # 2. Extent-based counts (30% -> 90%)
    failed_taxon_keys: set[str] = set()
    if latitude is not None and longitude is not None and extents:
        n_extents = len(extents)
        for i, dist_km in enumerate(extents):
            if progress_handler:
                p_current = 0.3 + (i / n_extents) * 0.5
                label = get_extent_column_name(dist_km)
                progress_handler(p_current, f"Querying {label} extent...")

            res_df, ext_failed = counts_for_area(
                taxon_keys, latitude, longitude, extent=dist_km, user_agent=user_agent
            )
            results.append(res_df)
            failed_taxon_keys.update(ext_failed)

    # 3. Finalizing (90% -> 100%)
    if progress_handler:
        progress_handler(0.9, "Merging and finalizing GBIF data...")

    if not results:
        logger.warning(
            "No extent results to merge — returning empty GEO search DataFrame"
        )
        return pd.DataFrame()

    counts_df = pd.concat(results, ignore_index=True)

    gbif_search_df = good_matches.merge(
        counts_df, on="taxonID", how="left", validate="one_to_many"
    )

    # Occurrence fetches that errored return their taxon keys (not a count). Drop
    # those species' rows so a transient error is not cached as "absent", and
    # route them to failure_sink for retry next run. Drop by taxonID (exact);
    # failure_sink needs the *input* species_list strings (the cache-item keys),
    # which strict matching guarantees equal to the result's canonicalName only
    # case-insensitively, so map back via normalize_name.
    if failed_taxon_keys and "taxonID" in gbif_search_df.columns:
        gbif_search_df = gbif_search_df[
            ~gbif_search_df["taxonID"].isin(failed_taxon_keys)
        ]
        errored_norm = set(
            good_matches.loc[
                good_matches["taxonID"].isin(failed_taxon_keys), "scientificName"
            ]
            .map(normalize_name)
            .dropna()
        )
        errored_inputs = sorted(
            s for s in species_list if normalize_name(s) in errored_norm
        )
        if errored_inputs:
            logger.warning(
                f"{len(errored_inputs)} GBIF taxa errored during occurrence fetch "
                "(will retry on next run)"
            )
            if failure_sink is not None:
                failure_sink.extend(errored_inputs)

    n_species = (
        gbif_search_df["scientificName"].nunique()
        if "scientificName" in gbif_search_df.columns
        else len(gbif_search_df)
    )
    logger.info(
        f"GBIF search complete: {n_species} species, {len(results)} extents, {len(gbif_search_df)} rows"
    )

    if progress_handler:
        progress_handler(1.0, "GBIF search complete.")

    return gbif_search_df


### MERGE PIPELINE ###


def _gbif_merge_params(
    gbif_search_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    gbif_search_params: dict,
    tax_params: dict,
    **kwargs,
):
    """Prepare parameters for GBIF merge step."""
    params = gbif_search_params | tax_params
    inputs = [
        {"gbif_extent": item} | tax_params
        for item in gbif_search_params.get("gbif_extents", [])
    ]
    warn_empty_params(tax_params, "tax_params")

    return inputs, params


@save_to_db(
    table_name="gbif_merge",
    cache=CustomCache(
        prepare_fn=_gbif_merge_params,
        match_map_dict={"gbif_extent": "gbif_extent"},
        fingerprint_on=["gbif_search_df"],
    ),
)
@preserve_sequence_order("seq_id", "tax_df")
def run_gbif_merge(
    gbif_search_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    tax_params: dict | None = None,
    gbif_search_params: dict | None = None,
) -> pd.DataFrame:
    """Merge GBIF counts with TAX sequence information on scientificName.

    Args:
        gbif_search_df: GBIF search DataFrame with 'scientificName' and count/taxon columns.
        tax_df: TAX DataFrame with 'scientificName', 'seq_id', etc.
        tax_params: TAX parameters (for caching).
        gbif_search_params: GBIF search parameters (for caching).

    Returns:
        DataFrame with GBIF data enriched with TAX sequence info.
    """
    if tax_params is None:
        tax_params = {}
    if gbif_search_params is None:
        gbif_search_params = {}

    # Validate inputs
    if gbif_search_df.empty:
        logger.warning("Empty GBIF DataFrame")
        return pd.DataFrame()

    if tax_df.empty:
        logger.warning("Empty TAX DataFrame")
        return gbif_search_df.copy()

    required_cols_gbif = {"scientificName"}
    required_cols_tax = {"scientificName", "seq_id"}

    missing_gbif = required_cols_gbif - set(gbif_search_df.columns)
    missing_tax = required_cols_tax - set(tax_df.columns)
    if missing_gbif or missing_tax:
        raise ValueError(f"Missing columns – GBIF: {missing_gbif}, TAX: {missing_tax}")

    # Join on a normalized key (case / whitespace insensitive) so accidental
    # spelling differences between WoRMS and GBIF do not silently drop species.
    tax_df = tax_df.assign(_join_key=tax_df["scientificName"].map(normalize_name))
    gbif_search_df = gbif_search_df.assign(
        _join_key=gbif_search_df["scientificName"].map(normalize_name)
    )
    gbif_keys = set(gbif_search_df["_join_key"].dropna())
    unmatched = sorted(
        tax_df.loc[~tax_df["_join_key"].isin(gbif_keys), "scientificName"]
        .dropna()
        .unique()
    )
    if unmatched:
        logger.warning(
            f"{len(unmatched)} TAX names had no GBIF match: {unmatched[:20]}"
        )

    merged_df = (
        tax_df.merge(
            gbif_search_df,
            on="_join_key",
            how="left",
            suffixes=("", "_gbif"),
        )
        .drop(columns="_join_key")
        .copy()
    )

    # Keep TAX identifiers, add GBIF URL as separate column
    gbif_suffix_cols = [c for c in merged_df.columns if c.endswith("_gbif")]
    merged_df = merged_df.rename(columns={"taxonURL_gbif": "gbif_taxonURL"})
    # taxonURL_gbif only exists when gbif_search_df carried a taxonURL column.
    if "taxonURL_gbif" in gbif_suffix_cols:
        gbif_suffix_cols.remove("taxonURL_gbif")
    merged_df = merged_df.drop(columns=gbif_suffix_cols)
    merged_df = reorder_taxonomy_columns(merged_df)

    n_tax = merged_df["scientificName"].nunique()
    n_gbif = gbif_search_df["scientificName"].nunique()
    n_unmatched = n_tax - n_gbif
    logger.info(
        f"Merged GBIF+TAX: {n_tax} species ({n_gbif} matched, {n_unmatched} without GBIF data)"
    )

    return merged_df


### FILTER PIPELINE ###


@save_to_db(
    "gbif_filter",
    cache=FullCache(
        local={
            "gbif_filter_extent": "extent",
            "gbif_min_occurrences": "min_occurrences",
        },
        inherit_from=["gbif_merge_params"],
    ),
)
@preserve_sequence_order("seq_id", "gbif_merge_df")
def run_gbif_filter(
    gbif_merge_df: pd.DataFrame,
    extent: float | str,
    min_occurrences: int,
    gbif_merge_params: dict | None = None,
) -> pd.DataFrame:
    """Filter GBIF data by a specific extent and minimum occurrence threshold.

    Args:
        gbif_merge_df: GBIF merge DataFrame.
        extent: Extent in km (numeric) or 'global' to select the filtering column.
        min_occurrences: Minimum number of occurrences required to keep a row.
        gbif_merge_params: Parameters from run_gbif_merge (for caching).

    Returns:
        Filtered DataFrame containing only rows meeting or exceeding the
        occurrence threshold for the given extent.
    """
    if gbif_merge_params is None:
        gbif_merge_params = {}

    logger.info(
        f"Filtering GBIF data for extent={extent} and min_occurrences={min_occurrences}"
    )

    # 1. Validation
    if (
        "gbif_extent" not in gbif_merge_df.columns
        or "occurrences" not in gbif_merge_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'gbif_extent' and 'occurrences' columns."
        )

    # 2. Filter rows
    mask = (gbif_merge_df["gbif_extent"].astype(str) == str(extent)) & (
        gbif_merge_df["occurrences"] >= min_occurrences
    )

    filtered_df = gbif_merge_df[mask].copy()

    # 3. Cleanup
    filtered_df = filtered_df.rename(columns={"occurrences": "gbif_occurrences"})
    filtered_df = filtered_df.drop(columns=["gbif_extent"])

    logger.info(
        f"Filtered GBIF data to {len(filtered_df)} species (threshold: {min_occurrences})"
    )

    return filtered_df


### FINALIZE GEO RESULTS ###


@save_to_db(
    "results_geo",
    cache=FullCache(inherit_from=["gbif_filter_params"]),
)
@preserve_sequence_order("seq_id", "gbif_filter_df")
def finalize_geo_results(
    gbif_filter_df: pd.DataFrame,
    gbif_filter_params: dict | None = None,
    ncbi_search_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Finalize the GEO list — remove (seq_id, species) pairs seen by NCBI but rejected by MOL.

    A pair is removed when ``in_mol`` is False for that row AND the same
    (seq_id, scientificName) pair appeared in the raw NCBI BLAST results.
    This means NCBI evaluated the species for that sequence and it didn't
    pass MOL filtering.

    Args:
        gbif_filter_df: Filtered GBIF occurrence DataFrame (with ``in_mol`` column).
        gbif_filter_params: GBIF filter parameters (for caching).
        ncbi_search_pairs: (seq_id, scientificName) tuples from raw NCBI
            BLAST results.  Rows where ``in_mol=False`` AND the pair is in
            this set are removed.

    Returns:
        Finalized GEO DataFrame.
    """
    if gbif_filter_params is None:
        gbif_filter_params = {}

    geo_df = gbif_filter_df.copy()

    if ncbi_search_pairs and "in_mol" in geo_df.columns:
        pair_col = list(zip(geo_df["seq_id"], geo_df["scientificName"]))
        rejected = ~geo_df["in_mol"].astype(bool) & pd.Series(
            [p in ncbi_search_pairs for p in pair_col], index=geo_df.index
        )
        n_removed = rejected.sum()
        if n_removed:
            n_species = geo_df.loc[rejected, "scientificName"].nunique()
            logger.info(
                f"Removed {n_removed} rows ({n_species} species) "
                f"seen by NCBI but rejected by MOL filtering"
            )
        geo_df = geo_df[~rejected]

    logger.success(f"Finalized GEO list: {len(geo_df)} rows")
    return geo_df


### EXPLORATION / SUMMARY ###


def get_ncbi_rejected_rows(
    gbif_filter_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return rows removed during finalization, enriched with top NCBI hit stats.

    These are (seq_id, species) pairs that NCBI found but MOL filtering rejected.
    """
    keys = ["seq_id", "scientificName"]
    merged = gbif_filter_df.merge(
        geo_df[keys].drop_duplicates(),
        on=keys,
        how="left",
        indicator=True,
    )
    rejected = gbif_filter_df.loc[merged["_merge"] == "left_only"].copy()

    if rejected.empty or ncbi_search_df is None:
        return rejected

    # Compute top identity/query_cover from raw NCBI search results
    top_hits = top_hit_per_group(
        ncbi_search_df[
            ["seq_id", "scientificName", "identity_percentage", "query_cover"]
        ],
        keys=keys,
        sort=["identity_percentage", "query_cover"],
    ).rename(
        columns={
            "identity_percentage": "mol_top_identity_percentage",
            "query_cover": "mol_top_query_cover",
        }
    )
    rejected = rejected.drop(
        columns=["mol_top_identity_percentage", "mol_top_query_cover"], errors="ignore"
    ).merge(top_hits, on=keys, how="left")

    return rejected


@preserve_sequence_order("seq_id", "gbif_merge_df")
def build_geo_summary(
    gbif_merge_df: pd.DataFrame,
    min_occurrences: int = GBIF_MIN_OCCURRENCES,
) -> pd.DataFrame:
    """Build summary table of species counts per sequence and search extent.

    Args:
        gbif_merge_df: GBIF data merged with TAX, including count columns.
        min_occurrences: Minimum occurrences threshold for counting presence.

    Returns:
        DataFrame with one row per seq_id and columns giving, for each extent,
        the number of taxa with occurrences ≥ min_occurrences.
    """
    # 1. Filter and Clean
    df = gbif_merge_df.dropna(subset=["seq_id"]).copy()

    if "gbif_extent" not in df.columns or "occurrences" not in df.columns:
        logger.warning(
            "Required columns 'gbif_extent' or 'occurrences' missing from GEO data."
        )
        return pd.DataFrame(columns=["seq_id"])

    df["gbif_extent"] = df["gbif_extent"].astype(str)

    # 2. Determine Presence
    df["is_present"] = (df["occurrences"] >= min_occurrences).astype(int)

    # 3. Pivot the Long data to Wide
    summary_pivot = df.pivot_table(
        index="seq_id",
        columns="gbif_extent",
        values="is_present",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reset_index()

    # 4. Column Formatting
    summary_pivot.columns = [get_extent_column_name(c) for c in summary_pivot.columns]

    seq_word = "sequence" if len(summary_pivot) == 1 else "sequences"
    logger.info(f"Generated GEO summary for {len(summary_pivot)} unique {seq_word}.")

    return summary_pivot


### PER-SEQUENCE EXTENT CLASSIFICATION ###

IDENTITY_COL = "mol_top_identity_percentage"


def classify_gbif_extents(
    seq_df: pd.DataFrame,
    min_occurrences: int,
    extents: list[float],
    priority_extent: float | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Pivot one sequence's GBIF occurrences to wide form and bucket species by
    the extent(s) at which they pass the occurrence threshold.

    Pure data logic for the GEO per-sequence table; holds no UI calls so the UI
    only iterates and renders the returned buckets.

    Args:
        seq_df: Long-form rows for a single seq_id (scientificName, in_mol,
            gbif_taxonURL, gbif_extent, occurrences, optionally identity).
        min_occurrences: Records within an extent needed to validate a species.
        extents: Search extents (km floats and/or "global").
        priority_extent: The extent currently driving results (km or "global").

    Returns:
        (df_wide, classification) where df_wide is the species x extent table and
        classification is a dict:
            priority_col: column name of the priority extent.
            extent_cols:  ordered extent column names present.
            has_identity: whether the identity column is available.
            buckets:      ordered list of dicts with keys
                kind   ("local" | "smaller" | "larger" | "never"),
                extent_col (str or None for "never"),
                df         (the species subset),
                total      (denominator for "smaller", else None).
    """
    priority_col = get_extent_column_name(priority_extent)
    has_identity = IDENTITY_COL in seq_df.columns

    # Pivot long -> wide (one row per species, one column per extent)
    df_wide = (
        seq_df.pivot_table(
            index=["scientificName", "in_mol", "gbif_taxonURL"],
            columns="gbif_extent",
            values="occurrences",
            fill_value=0,
        )
        .reset_index()
        .infer_objects()
    )

    # Merge identity back (one value per species)
    if has_identity:
        identity_map = seq_df[["scientificName", IDENTITY_COL]].drop_duplicates(
            subset="scientificName"
        )
        df_wide = df_wide.merge(identity_map, on="scientificName", how="left")

    # Standardize column names (e.g. 500 -> "500 km")
    non_extent_cols = {"scientificName", "in_mol", "gbif_taxonURL", IDENTITY_COL}
    df_wide.columns = [
        get_extent_column_name(c) if c not in non_extent_cols else c
        for c in df_wide.columns
    ]

    # Local = validated in the priority extent
    if priority_col in df_wide.columns:
        is_local = df_wide[priority_col] >= min_occurrences
    else:
        is_local = pd.Series([False] * len(df_wide))

    extent_cols = [get_extent_column_name(e) for e in extents]
    if "global" in df_wide.columns and "global" not in extent_cols:
        extent_cols.append("global")

    local_df = df_wide[is_local]
    other_df = df_wide[~is_local]

    buckets: list[dict] = [
        {"kind": "local", "extent_col": priority_col, "df": local_df, "total": None}
    ]

    if other_df.empty:
        return df_wide, {
            "priority_col": priority_col,
            "extent_cols": extent_cols,
            "has_identity": has_identity,
            "buckets": buckets,
        }

    priority_km = (
        float(priority_extent) if priority_extent != "global" else float("inf")
    )

    # Smaller extents: subset of validated species with stronger geographic evidence
    smaller_cols = [
        get_extent_column_name(e)
        for e in sorted(
            (
                e
                for e in extents
                if e != "global"
                and get_extent_column_name(e) != priority_col
                and float(e) < priority_km
            ),
            key=float,
        )
    ]
    for ext_col in smaller_cols:
        if ext_col not in local_df.columns:
            continue
        also_here = local_df[local_df[ext_col] >= min_occurrences]
        if also_here.empty:
            continue
        buckets.append(
            {
                "kind": "smaller",
                "extent_col": ext_col,
                "df": also_here,
                "total": len(local_df),
            }
        )

    # Larger extents: species NOT validated at priority but validated further out
    larger_cols = [
        get_extent_column_name(e)
        for e in sorted(
            (
                e
                for e in extents
                if e != "global"
                and get_extent_column_name(e) != priority_col
                and float(e) > priority_km
            ),
            key=float,
        )
    ]
    if "global" in extent_cols and priority_col != "global":
        larger_cols.append("global")

    already_shown: set = set()
    for ext_col in larger_cols:
        if ext_col not in other_df.columns:
            continue
        validated_here = other_df[
            (other_df[ext_col] >= min_occurrences)
            & (~other_df["scientificName"].isin(already_shown))
        ]
        if validated_here.empty:
            continue
        buckets.append(
            {
                "kind": "larger",
                "extent_col": ext_col,
                "df": validated_here,
                "total": None,
            }
        )
        already_shown.update(validated_here["scientificName"])

    # Species never passing the threshold at any extent
    never_validated = other_df[~other_df["scientificName"].isin(already_shown)]
    if not never_validated.empty:
        buckets.append(
            {"kind": "never", "extent_col": None, "df": never_validated, "total": None}
        )

    return df_wide, {
        "priority_col": priority_col,
        "extent_cols": extent_cols,
        "has_identity": has_identity,
        "buckets": buckets,
    }
