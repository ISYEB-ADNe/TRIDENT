"""GBIF pipeline: the GEO (geographic scope) list.

Step order:
    1. prepare_gbif_input(tax_df)                      → list[str]
       prepare_gbif_worms_ids(tax_df)                  → dict[str, str]
    2. run_gbif_search(species_list, ...)               → (gbif_search_df, params)
    3. run_gbif_merge(gbif_search_df, tax_df, ...)      → (gbif_merge_df, params)
    4. run_gbif_filter(gbif_merge_df, ...)              → (gbif_filter_df, params)
    5. finalize_geo_results(gbif_filter_df, ...)        → (geo_df, params)

Exploration:
    build_geo_summary(gbif_merge_df, min_occurrences)
    classify_gbif_extents(seq_df, min_occurrences, extents, priority_extent)
"""

from collections.abc import Callable

import pandas as pd

from loguru import logger

from trident.clients.gbif import (
    match_taxon_names,
    filter_taxon_matches,
    counts_for_area,
    area_label,
    get_extent_column_name,
)

from trident.core.constants import GBIF_MIN_OCCURRENCES
from trident.pipelines.tax_pipeline import TAX_COLUMNS
from trident.core.database import save_to_db, FullCache, CustomCache, warn_empty_params
from trident.core.utils import (
    as_ids,
    StepLog,
    name_list,
    pluralize,
    species_for_sequences,
    ensure_columns,
    preserve_sequence_order,
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
    extents: list[int | str],
    area: str,
    **kwargs,
):
    """Extract parameters for GBIF search step.

    Extents are whole km (or "global"): an extent is a key of the saved
    search, and 500.0 read back as text would no longer match 500.
    """
    not_whole = [
        e
        for e in extents
        if e != "global" and (isinstance(e, bool) or not isinstance(e, int))
    ]
    if not_whole:
        raise ValueError(
            f"Search extents must be whole km or 'global', not {not_whole}"
        )
    params = {
        "gbif_latitude": latitude,
        "gbif_longitude": longitude,
        "gbif_extents": extents,
        "gbif_area": area,
    }
    inputs = [
        {
            "species": species,
            "gbif_latitude": None if extent == "global" else latitude,
            "gbif_longitude": None if extent == "global" else longitude,
            "gbif_extent": extent,
            "gbif_area": None if extent == "global" else area,
        }
        for species in species_list
        for extent in extents
    ]
    return inputs, params


def _gbif_search_params_rebuild(input_selected, original_inputs):
    """The search's kwargs for the (species, extent) pairs not saved yet: each
    extent counts only its own missing species."""
    species_by_extent: dict = {}
    for item in input_selected:
        species_by_extent.setdefault(item["gbif_extent"], []).append(item["species"])
    species = {s for names in species_by_extent.values() for s in names}
    return original_inputs.copy() | {
        "species_list": [
            s for s in dict.fromkeys(original_inputs["species_list"]) if s in species
        ],
        "extents": [
            e
            for e in dict.fromkeys(original_inputs["extents"])
            if e in species_by_extent
        ],
        "_species_by_extent": species_by_extent,
    }


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
    extents: list[int | str] | None = None,
    area: str = "circle",
    worms_ids: dict[str, str] | None = None,
    progress_handler: Callable[[float, str], None] | None = None,
    user_agent: str | None = None,
    _species_by_extent: dict[int | str, list[str]] | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Count the GBIF occurrences of each species in each area.

    Args:
        species_list: Scientific names to match in GBIF.
        latitude: Center latitude of the search areas.
        longitude: Center longitude of the search areas.
        extents: Distances (whole km) from the center, or "global".
        area: "circle" (within each distance of the center) or "box" (the
            bounding box of that circle).
        worms_ids: WoRMS AphiaID of each name, matched first (see
            ``prepare_gbif_worms_ids``).
        progress_handler: Optional callback for UI progress updates.
        user_agent: User-Agent header for HTTP requests.
        _species_by_extent: Set by the cache only, for a resumed search: the
            species to count in each extent, the pairs not saved yet.
        failure_sink: If provided, species whose name match errored (network or
            403, ``matchType == "ERROR"``) are appended here so the cache layer
            skips them and retries on the next run. A genuine no-match (needs
            review) is a valid empty, not a failure. A species whose
            occurrence count errored in one area is appended with that area,
            as (species, extent).
        result_sink: Given by save_to_db, which saves each area as soon as its
            counts are in.

    Returns:
        One row per (species, extent) with GBIF data: the match columns
        (filter_taxon_matches), occurrences and gbif_extent.
    """
    log = StepLog("GBIF search")
    errored = []

    # 1. Taxon matching
    if progress_handler:
        progress_handler(0.1, "Matching names in GBIF...")

    matches_df = match_taxon_names(
        species_list, worms_ids=worms_ids, user_agent=user_agent
    )

    # Errored matches (network/403) are retry-worthy failures, distinct from
    # genuine no-matches that need review.
    if "matchType" in matches_df.columns:
        errored = matches_df[matches_df["matchType"] == "ERROR"]["query"].tolist()
        if errored and failure_sink is not None:
            failure_sink.extend(errored)

    # Filter to valid keys
    good_matches = filter_taxon_matches(matches_df)

    # The species whose match did not fail: each of their areas is saved once
    # counted, also for a species GBIF does not hold (a genuine empty).
    match_failed = set(errored)
    queried = [s for s in species_list if s not in match_failed]
    results = []

    # 2. Counts in each area
    species_by_extent = _species_by_extent or {e: species_list for e in extents or []}
    if latitude is not None and longitude is not None and extents:
        n_extents = len(extents)
        for i, dist_km in enumerate(extents):
            if progress_handler:
                p_current = 0.3 + (i / n_extents) * 0.5
                label = get_extent_column_name(dist_km)
                progress_handler(p_current, f"Querying {label} extent...")

            wanted = set(species_by_extent[dist_km])
            here = good_matches[good_matches["scientificName"].isin(wanted)]
            in_area = [s for s in queried if s in wanted]
            res_df, ext_failed = counts_for_area(
                here["taxonID"].astype(str).tolist(),
                latitude,
                longitude,
                extent=dist_km,
                area=area,
                user_agent=user_agent,
            )
            # Names sharing a GBIF species each get its counts
            area_df = here.merge(
                res_df, on="taxonID", how="left", validate="many_to_many"
            )

            # Occurrence fetches that errored return their taxon keys (not a
            # count). Drop those species' rows in this area so a transient
            # error is not cached as "absent", and route (species, area) to
            # failure_sink for retry next run. Drop by taxonID (exact); the
            # result's scientificName is the queried name, the cache-item key
            # failure_sink needs.
            area_errored = []
            if ext_failed and "taxonID" in area_df.columns:
                area_df = area_df[~area_df["taxonID"].isin(ext_failed)]
                area_errored = sorted(
                    set(here.loc[here["taxonID"].isin(ext_failed), "scientificName"])
                )
                errored += [s for s in area_errored if s not in errored]
                if failure_sink is not None:
                    failure_sink.extend((s, dist_km) for s in area_errored)
            results.append(area_df)
            if result_sink is not None:
                done = [(s, dist_km) for s in in_area if s not in area_errored]
                result_sink(area_df, done)

    # 3. Finalizing
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

# The GBIF search's columns the GEO list takes, and their GEO names: the
# species' GBIF page, its occurrences and the area they were counted in.
GEO_COLUMNS = {
    "taxonURL": "geo_url",
    "occurrences": "geo_occurrences",
    "gbif_extent": "geo_extent",
}

# The columns of the GEO list: the TAX list's, with the species' GBIF page
# and its occurrences in the filtering area.
GEO_LIST_COLUMNS = [*TAX_COLUMNS, "geo_url", "geo_occurrences"]


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
        match_map_dict={"gbif_extent": "geo_extent"},
        fingerprint_on=["gbif_search_df"],
        # Every extent's rows have the same columns, from this merge.
        whole_result=True,
        versioned=True,
    ),
    log_reuse_as="GEO list",
    bool_columns=("in_mol",),
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
        tax_params: Params of finalize_tax_results, for the cache key only.
        gbif_search_params: Params of run_gbif_search, for the cache key only.

    Returns:
        The TAX rows of the species with GBIF data (inner join), one per
        extent, with geo_url, geo_occurrences and geo_extent (GEO_COLUMNS).
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

    # The GBIF search's scientificName is the queried TAX name
    gbif_names = set(gbif_search_df["scientificName"].dropna())
    unmatched = sorted(
        tax_df.loc[~tax_df["scientificName"].isin(gbif_names), "scientificName"]
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
        columns=["scientificName", *GEO_COLUMNS]
    ).rename(columns=GEO_COLUMNS)
    # Only the TAX species with GBIF data: the others have no extent, so no
    # cache entry could hold them, and joining them first would turn the
    # counts into decimals.
    return tax_df.merge(gbif_counts, on="scientificName", how="inner")


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
    log_reuse_as="GEO filter",
    bool_columns=("in_mol",),
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
        gbif_merge_params: Params of run_gbif_merge, for the cache key only.

    Returns:
        Filtered DataFrame containing only rows meeting or exceeding the
        occurrence threshold for the given extent.
    """
    if gbif_merge_params is None:
        gbif_merge_params = {}

    # 1. Validation
    if (
        "geo_extent" not in gbif_merge_df.columns
        or "geo_occurrences" not in gbif_merge_df.columns
    ):
        raise ValueError(
            "DataFrame must contain 'geo_extent' and 'geo_occurrences' columns."
        )

    # 2. Filter rows
    # Compared by name: 250, 250.0 and "250" (as saved) are the same extent.
    in_extent = gbif_merge_df["geo_extent"].map(
        get_extent_column_name
    ) == get_extent_column_name(extent)
    mask = in_extent & (gbif_merge_df["geo_occurrences"] >= min_occurrences)

    filtered_df = gbif_merge_df[mask].copy()

    # 3. Cleanup
    filtered_df = filtered_df[GEO_LIST_COLUMNS]

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
    bool_columns=("in_mol",),
)
@preserve_sequence_order("seq_id", "gbif_filter_df")
def finalize_geo_results(
    gbif_filter_df: pd.DataFrame,
    gbif_filter_params: dict | None = None,
) -> pd.DataFrame:
    """Finalize the GEO list for export or downstream analysis.

    Args:
        gbif_filter_df: Filtered GBIF occurrence DataFrame.
        gbif_filter_params: Params of run_gbif_filter, for the cache key only.

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
        the number of species with occurrences ≥ min_occurrences.
    """
    # 1. Filter and Clean
    df = gbif_merge_df.dropna(subset=["seq_id"]).copy()

    if "geo_extent" not in df.columns or "geo_occurrences" not in df.columns:
        logger.warning(
            "Required columns 'geo_extent' or 'geo_occurrences' missing from GEO data."
        )
        return pd.DataFrame(columns=["seq_id"])

    df["geo_extent"] = df["geo_extent"].astype(str)

    # 2. Determine Presence
    df["is_present"] = (df["geo_occurrences"] >= min_occurrences).astype(int)

    # 3. Pivot the Long data to Wide
    summary_pivot = df.pivot_table(
        index="seq_id",
        columns="geo_extent",
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

IDENTITY_COL = "mol_identity"


def classify_gbif_extents(
    seq_df: pd.DataFrame,
    min_occurrences: int,
    extents: list[int | str],
    priority_extent: float | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Pivot one sequence's GBIF occurrences to wide form and bucket species by
    the extent(s) at which they pass the occurrence threshold.

    Pure data logic for the GEO per-sequence table; holds no UI calls so the UI
    only iterates and renders the returned buckets.

    Args:
        seq_df: Long-form rows for a single seq_id (scientificName, in_mol,
            geo_url, geo_extent, geo_occurrences, optionally mol_identity).
        min_occurrences: Records within an extent needed to keep a species.
        extents: Search extents (whole km and/or "global").
        priority_extent: The filtering area: the extent Minimum occurrences
            applies to (km or "global").

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
            index=["scientificName", "in_mol", "geo_url"],
            columns="geo_extent",
            values="geo_occurrences",
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
    non_extent_cols = {"scientificName", "in_mol", "geo_url", IDENTITY_COL}
    df_wide.columns = [
        get_extent_column_name(c) if c not in non_extent_cols else c
        for c in df_wide.columns
    ]

    # Local: kept in the filtering area
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

    # Smaller extents: the kept species also kept closer to the site
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

    # Larger extents: species not kept in the filtering area but kept further out
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
