"""GBIF pipeline — produces the GEO (Geographic scope) results list.

Step order:
    1. prepare_gbif_input(tax_df)                      → list[str]
       prepare_gbif_worms_ids(tax_df)                  → dict[str, str]
    2. run_gbif_search(species_list, ...)               → (gbif_search_df, params)
    3. run_gbif_merge(gbif_search_df, tax_df, ...)      → (gbif_merge_df, params)
    4. run_gbif_filter(gbif_merge_df, ...)              → (gbif_filter_df, params)
    5. finalize_geo_results(gbif_filter_df, ...)        → (geo_df, params)

Exploration:
    build_geo_summary(gbif_merge_df)
"""

from collections.abc import Callable
from functools import partial
from typing import Literal

import pandas as pd

from loguru import logger

from trident.clients.gbif import (
    COL_XR_CHECKLIST,
    match_taxon_names,
    match_taxon_names_v1,
    filter_taxon_matches,
    filter_taxon_matches_v1,
    counts_for_area,
    area_label,
    get_extent_column_name,
)

from trident.core.constants import GBIF_MIN_OCCURRENCES
from trident.core.database import save_to_db, FullCache, CustomCache, warn_empty_params
from trident.core.utils import (
    as_ids,
    StepLog,
    name_list,
    pluralize,
    species_for_sequences,
    ensure_columns,
    reorder_taxonomy_columns,
    preserve_sequence_order,
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


def prepare_gbif_worms_ids(df: pd.DataFrame) -> dict[str, str]:
    """Return the WoRMS AphiaID of each species, which the GBIF search matches
    first.

    Args:
        df: TAX DataFrame with 'scientificName', 'taxonID' and 'taxonID_db'.

    Returns:
        Dict of scientific name to AphiaID, for the WoRMS rows with an ID.
    """
    worms = df[df["taxonID_db"] == "WORMS"]
    aphia_ids = as_ids(worms["taxonID"])
    known = aphia_ids.notna() & worms["scientificName"].notna()
    return dict(zip(worms.loc[known, "scientificName"], aphia_ids[known].astype(str)))


### SEARCH WORKFLOW ###


def _gbif_search_params(
    species_list: list[str],
    latitude: float | None,
    longitude: float | None,
    extents: list[float | None],
    area: str,
    taxonomy: str,
    **kwargs,
):
    """Extract parameters for GBIF search step."""
    params = {
        "gbif_latitude": latitude,
        "gbif_longitude": longitude,
        "gbif_extents": extents,
        "gbif_area": area,
        "gbif_taxonomy": taxonomy,
    }
    inputs = [
        {
            "species": species,
            "gbif_latitude": None if extent == "global" else latitude,
            "gbif_longitude": None if extent == "global" else longitude,
            "gbif_extent": extent,
            "gbif_area": None if extent == "global" else area,
            "gbif_taxonomy": taxonomy,
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
    log_as=("GBIF search", "occurrence count"),
)
def run_gbif_search(
    species_list: list[str],
    latitude: float | None = None,
    longitude: float | None = None,
    extents: list[float | str] | None = None,
    area: str = "circle",
    worms_ids: dict[str, str] | None = None,
    taxonomy: Literal["col_xr", "backbone"] = "col_xr",
    progress_handler: Callable[[float, str], None] | None = None,
    user_agent: str | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Run GBIF occurrence search for a list of taxa.

    Args:
        species_list: Scientific names to match in GBIF.
        latitude: Center latitude of the search areas.
        longitude: Center longitude of the search areas.
        extents: List of distances (km) from center, or "global".
        area: "circle" (within each distance of the center) or "box" (the
            bounding box of that circle).
        worms_ids: WoRMS AphiaID of each name, matched first (see
            ``prepare_gbif_worms_ids``).
        taxonomy: "col_xr", GBIF's current taxonomy (v2 match API), or
            "backbone", the GBIF Backbone frozen since 2023 (v1 match API).
        progress_handler: Optional callback for UI progress updates.
        user_agent: User-Agent header for HTTP requests.
        failure_sink: If provided, species whose name match errored (network or
            403, ``matchType == "ERROR"``) are appended here so the cache layer
            skips them and retries on the next run. A genuine no-match (needs
            review) is a valid empty, not a failure. A species whose
            occurrence count errored in one area is appended with that area,
            as (species, extent).
        result_sink: Given by save_to_db, which saves each area as soon as its
            counts are in.

    Returns:
        DataFrame with GBIF taxon metadata and occurrence count columns.
    """
    log = StepLog("GBIF search")
    errored = []

    # 1. Taxon Matching (0% -> 30%)
    if progress_handler:
        progress_handler(0.1, "Matching names in GBIF...")

    if taxonomy == "col_xr":
        match, filter_matches, checklist_key = (
            partial(match_taxon_names, worms_ids=worms_ids),
            filter_taxon_matches,
            COL_XR_CHECKLIST,
        )
    else:
        match, filter_matches, checklist_key = (
            match_taxon_names_v1,
            filter_taxon_matches_v1,
            None,
        )
    matches_df = match(species_list, user_agent=user_agent)

    ## Errored matches (network/403) are retry-worthy failures, distinct from
    ## genuine no-matches that need review.
    if "matchType" in matches_df.columns:
        errored = matches_df[matches_df["matchType"] == "ERROR"]["query"].tolist()
        if errored and failure_sink is not None:
            failure_sink.extend(errored)

    ## Filter to valid keys
    good_matches = filter_matches(matches_df)
    taxon_keys = good_matches["taxonID"].astype(str).tolist()

    # The species whose match did not fail: each of their areas is saved once
    # counted, also for a species GBIF does not hold (a genuine empty).
    match_failed = set(errored)
    queried = [s for s in species_list if s not in match_failed]
    results = []

    # 2. Extent-based counts (30% -> 90%)
    if latitude is not None and longitude is not None and extents:
        n_extents = len(extents)
        for i, dist_km in enumerate(extents):
            if progress_handler:
                p_current = 0.3 + (i / n_extents) * 0.5
                label = get_extent_column_name(dist_km)
                progress_handler(p_current, f"Querying {label} extent...")

            res_df, ext_failed = counts_for_area(
                taxon_keys,
                latitude,
                longitude,
                extent=dist_km,
                area=area,
                checklist_key=checklist_key,
                user_agent=user_agent,
            )
            # Names sharing a GBIF species each get its counts
            area_df = good_matches.merge(
                res_df, on="taxonID", how="left", validate="many_to_many"
            )

            # Occurrence fetches that errored return their taxon keys (not a
            # count). Drop those species' rows in this area so a transient
            # error is not cached as "absent", and route (species, area) to
            # failure_sink for retry next run. Drop by taxonID (exact);
            # failure_sink needs the *input* species_list strings (the
            # cache-item keys). The result's scientificName is the queried
            # name, or with the Backbone GBIF's name, equal to it only
            # case-insensitively, so map back via normalize_name.
            area_errored = []
            if ext_failed and "taxonID" in area_df.columns:
                area_df = area_df[~area_df["taxonID"].isin(ext_failed)]
                errored_norm = set(
                    good_matches.loc[
                        good_matches["taxonID"].isin(ext_failed), "scientificName"
                    ]
                    .map(normalize_name)
                    .dropna()
                )
                area_errored = sorted(
                    s for s in species_list if normalize_name(s) in errored_norm
                )
                errored += [s for s in area_errored if s not in errored]
                if failure_sink is not None:
                    failure_sink.extend((s, dist_km) for s in area_errored)
            results.append(area_df)
            if result_sink is not None:
                done = [(s, dist_km) for s in queried if s not in area_errored]
                result_sink(area_df, done)

    # 3. Finalizing (90% -> 100%)
    if progress_handler:
        progress_handler(0.9, "Merging and finalizing GBIF data...")

    if not results:
        log.done("no area searched", errored, "species")
        return ensure_columns(
            good_matches.iloc[:0].copy(), ["occurrences", "gbif_extent"]
        )

    gbif_search_df = pd.concat(results, ignore_index=True)

    n_species = (
        gbif_search_df["scientificName"].nunique()
        if "scientificName" in gbif_search_df.columns
        else len(gbif_search_df)
    )
    log.done(
        f"{n_species} species, {len(results)} {pluralize(len(results), 'area')}",
        errored,
        "species",
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
        logger.info(
            f"GEO list: {len(unmatched)} TAX {pluralize(len(unmatched), 'species')} "
            f"without GBIF data ({name_list(unmatched)})"
        )

    # The GEO list keeps TAX's columns and adds GBIF's counts and taxon page
    # only: GBIF's name of each species goes to the name table (gbif_name_rows).
    gbif_counts = gbif_search_df.reindex(
        columns=["_join_key", "taxonURL", "occurrences", "gbif_extent"]
    ).rename(columns={"taxonURL": "gbif_taxonURL"})
    merged_df = tax_df.merge(gbif_counts, on="_join_key", how="left").drop(
        columns="_join_key"
    )
    merged_df = reorder_taxonomy_columns(merged_df)

    # TAX species without GBIF data have no extent, so no cache entry can hold
    # them. Drop them here, so the result is the same with or without a
    # database.
    return merged_df.dropna(subset=["gbif_extent"])


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
        f"GEO filter: {filtered_df['scientificName'].nunique()} species with at "
        f"least {min_occurrences} GBIF {pluralize(min_occurrences, 'occurrence')} "
        f"{area_label(extent, gbif_merge_params.get('gbif_area'))}"
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
) -> pd.DataFrame:
    """Finalize the GEO list for export or downstream analysis.

    Args:
        gbif_filter_df: Filtered GBIF occurrence DataFrame.
        gbif_filter_params: GBIF filter parameters (for caching).

    Returns:
        Finalized GEO DataFrame.
    """
    if gbif_filter_params is None:
        gbif_filter_params = {}

    geo_df = gbif_filter_df.copy()
    logger.success(f"GEO list: {species_for_sequences(geo_df)}")
    return geo_df


### EXPLORATION / SUMMARY ###


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

    seq_word = pluralize(len(summary_pivot), "sequence")
    logger.debug(f"Generated GEO summary for {len(summary_pivot)} unique {seq_word}.")

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


### EMPTY INPUT ###


def empty_geo_results(tax_df: pd.DataFrame) -> pd.DataFrame:
    """Return the GEO list of an empty TAX list, with its usual columns.

    The GEO steps run on no species and without the database, so nothing is
    queried or cached.
    """
    search_df, _ = run_gbif_search([], extents=[], db_path=None)
    merge_df, _ = run_gbif_merge(
        search_df, tax_df, tax_params={}, gbif_search_params={}, db_path=None
    )
    filter_df, _ = run_gbif_filter(
        merge_df,
        extent="global",
        min_occurrences=GBIF_MIN_OCCURRENCES,
        db_path=None,
    )
    geo_df, _ = finalize_geo_results(filter_df, db_path=None)
    return geo_df
