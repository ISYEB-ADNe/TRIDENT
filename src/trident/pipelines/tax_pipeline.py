"""WoRMS pipeline — produces the TAX (Taxonomic scope) results list.

Step order:
    1. prepare_worms_input(mol_df)               → list[str]
    2. run_worms_search(genera_list, ...)         → (worms_search_df, params)
    3. run_worms_merge(worms_search_df, ...)      → (worms_merge_df, params)
    4. finalize_tax_results(worms_merge_df, ...)  → (tax_df, params)

Exploration:
    build_tax_summary(tax_df)
"""

from typing import NamedTuple

import pandas as pd
import numpy as np

from loguru import logger

from trident.core.database import save_to_db, PartialCache, FullCache
from trident.core.http import create_session
from trident.core.utils import (
    preserve_sequence_order,
    top_hit_per_group,
    normalize_name,
    notify_progress,
)

from trident.clients.worms import get_species_from_genus_list, resolve_accepted_name


### DATA PREPARATION ###


def prepare_worms_input(df: pd.DataFrame) -> list[str]:
    """Extract unique genera from the MOL DataFrame for WoRMS querying.

    When name resolution (R3) has run, expand the WoRMS-accepted genus (the
    first token of acceptedName) so synonyms reassigned to a different genus
    (e.g. Allocentrotus -> Strongylocentrotus) expand the correct genus.
    Otherwise fall back to the raw 'genus' column.

    Args:
        df: DataFrame with a 'genus' column, and optionally 'acceptedName'.

    Returns:
        List of unique genera (non-null) to query in WoRMS.
    """
    if "acceptedName" in df.columns:
        genera = df["acceptedName"].map(
            lambda name: name.split()[0] if isinstance(name, str) and name else None
        )
    else:
        genera = df["genus"]
    # A hit without a name has no genus to expand.
    genera = genera.dropna().unique().tolist()
    genus_word = "genus" if len(genera) == 1 else "genera"
    logger.debug(f"Prepared WoRMS input: {len(genera)} unique {genus_word}")
    return genera


### NAME RESOLUTION (R3) ###


def prepare_resolution_input(mol_df: pd.DataFrame) -> list[str]:
    """Unique non-null MOL scientific names to canonicalise via WoRMS."""
    return mol_df["scientificName"].dropna().unique().tolist()


def prepare_resolution_taxids(mol_df: pd.DataFrame) -> dict[str, int]:
    """NCBI taxid of each MOL name, for names WoRMS does not find by name.

    A binomial can come from hits with several taxids (e.g. subspecies); the
    most frequent one is kept.
    """
    if "ncbi_taxid" not in mol_df.columns:
        return {}
    hits = mol_df[["scientificName", "ncbi_taxid"]].dropna()
    return {
        name: int(taxids.mode().iloc[0])
        for name, taxids in hits.groupby("scientificName")["ncbi_taxid"]
    }


@save_to_db(
    table_name="name_resolution",
    cache=PartialCache(items_kwarg="names", item_key="scientificName"),
)
def run_name_resolution(
    names: list[str],
    ncbi_taxids: dict[str, int] | None = None,
    user_agent: str | None = None,
    progress_handler: dict | object | None = None,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """Resolve MOL (NCBI) names to their WoRMS accepted name + AphiaID (R3).

    One AphiaRecordsByName lookup per unique name (cached per item, so only new
    names cost a call). A name WoRMS does not know is looked up a second time
    through its NCBI taxid, when given. Names found neither way are returned
    unchanged (acceptedName == input, no AphiaID). Only genuine network/HTTP
    errors reach failure_sink; a 404 is a valid not-found, not a failure.

    Args:
        names: Unique MOL scientific names to resolve.
        ncbi_taxids: Optional NCBI taxid per name (prepare_resolution_taxids).
        user_agent: User-Agent header for the WoRMS requests.
        progress_handler: Optional progress tracker (dict 'current' / .update()).
        failure_sink: If provided, names whose query errored are appended so the
            cache layer skips them and retries next run.

    Returns:
        DataFrame with columns scientificName, acceptedName, acceptedNameUsageID,
        is_synonym, matchedBy ('name', 'ncbi_taxid', or empty when not found),
        one row per input name.
    """
    logger.info(f"Resolving {len(names)} MOL names against WoRMS")
    session = create_session(user_agent=user_agent)
    ncbi_taxids = ncbi_taxids or {}
    rows: list[dict] = []
    failed: list[str] = []
    try:
        for name in names:
            try:
                resolved = resolve_accepted_name(
                    name, ncbi_taxid=ncbi_taxids.get(name), session=session
                )
            except Exception as exc:
                logger.warning(f"WoRMS resolution failed for '{name}': {exc}")
                failed.append(name)
                continue
            finally:
                notify_progress(progress_handler)
            rows.append(
                {
                    "scientificName": name,
                    "acceptedName": resolved["accepted_name"],
                    "acceptedNameUsageID": resolved["aphia_id"],
                    "is_synonym": resolved["is_synonym"],
                    "matchedBy": resolved["matched_by"],
                }
            )
    finally:
        session.close()

    if failed:
        logger.warning(f"{len(failed)} names failed WoRMS resolution")
        if failure_sink is not None:
            failure_sink.extend(failed)

    n_syn = sum(1 for r in rows if r["is_synonym"])
    n_taxid = sum(1 for r in rows if r["matchedBy"] == "ncbi_taxid")
    logger.info(
        f"Resolved {len(rows)} names ({n_syn} synonyms canonicalised, "
        f"{n_taxid} found through their NCBI taxid)"
    )
    return pd.DataFrame(
        rows,
        columns=[
            "scientificName",
            "acceptedName",
            "acceptedNameUsageID",
            "is_synonym",
            "matchedBy",
        ],
    )


def apply_name_resolution(
    mol_df: pd.DataFrame, resolution_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Attach acceptedName / acceptedNameUsageID / matchedBy to mol_df.

    scientificName stays the raw NCBI name (the MOL tab shows it verbatim);
    acceptedName holds the WoRMS accepted name (== the raw name when it is not a
    synonym or is absent from WoRMS). The boundary joins (genus expansion,
    in_mol, MOL+GEO) key on acceptedName, so synonyms reconcile to the accepted
    taxon while the original identification is preserved. matchedBy says how
    WoRMS found the name: 'name', 'ncbi_taxid', or empty.

    Args:
        mol_df: MOL DataFrame with a 'scientificName' column.
        resolution_df: Output of run_name_resolution, or None to skip (acceptedName
            falls back to the raw name).

    Returns:
        mol_df copy with acceptedName, acceptedNameUsageID and matchedBy columns.
    """
    out = mol_df.copy()
    if resolution_df is None or resolution_df.empty:
        out["acceptedName"] = out["scientificName"]
        out["acceptedNameUsageID"] = pd.NA
        out["matchedBy"] = pd.NA
        return out

    columns = ["acceptedName", "acceptedNameUsageID", "matchedBy"]
    lookup = (
        resolution_df.assign(_key=resolution_df["scientificName"].map(normalize_name))
        .reindex(columns=["_key", *columns])
        .drop_duplicates("_key")
        .set_index("_key")
    )
    norm_key = out["scientificName"].map(normalize_name)
    for col in columns:
        out[col] = norm_key.map(lookup[col])
    out["acceptedName"] = out["acceptedName"].fillna(out["scientificName"])
    return out


class NameResolutionSummary(NamedTuple):
    """What the TAX tab reports about the names of the MOL list."""

    n_names: int  # distinct MOL (NCBI) names
    n_species: int  # distinct WoRMS accepted names they resolve to
    renamed: pd.DataFrame  # scientificName, acceptedName, matchedBy
    not_found: pd.DataFrame  # scientificName, genusInTax: names with no record


def summarize_name_resolution(
    mol_df: pd.DataFrame,
    tax_df: pd.DataFrame | None = None,
    failed_names: list[str] | None = None,
) -> NameResolutionSummary:
    """Relate the NCBI names of the MOL list to the WoRMS names TAX works on.

    TAX compares and expands the WoRMS accepted names, so two NCBI names that
    WoRMS treats as one species count once there (e.g. Gadus ogac and Gadus
    macrocephalus). Names are compared case- and whitespace-insensitively, and
    hits without a name are left out.

    Args:
        mol_df: MOL DataFrame after apply_name_resolution (scientificName,
            acceptedName, acceptedNameUsageID, matchedBy).
        tax_df: TAX list, used to tell whether the genus of a name not found in
            WoRMS was still expanded (so its sequence lost no candidates).
        failed_names: Names whose WoRMS lookup errored. They are retried on the
            next run, so they are not reported as not found.

    Returns:
        A NameResolutionSummary. renamed has one row per MOL name whose WoRMS
        accepted name differs; not_found one row per MOL name with no WoRMS
        record, sorted, with genusInTax True when its genus is in the TAX list.
    """
    names = mol_df.loc[
        mol_df["scientificName"].notna(),
        ["scientificName", "acceptedName", "acceptedNameUsageID", "matchedBy"],
    ]
    names = names.assign(
        _name=names["scientificName"].map(normalize_name),
        _accepted=names["acceptedName"].map(normalize_name),
    ).drop_duplicates("_name")

    is_renamed = names["_name"] != names["_accepted"]
    renamed = (
        names.loc[is_renamed, ["scientificName", "acceptedName", "matchedBy"]]
        .sort_values("scientificName")
        .reset_index(drop=True)
    )

    failed = set(failed_names or [])
    is_not_found = names["acceptedNameUsageID"].isna() & ~names["scientificName"].isin(
        failed
    )
    tax_genera = (
        set(tax_df["genus"].dropna().map(normalize_name))
        if tax_df is not None and "genus" in tax_df.columns
        else set()
    )
    not_found = (
        names.loc[is_not_found, ["scientificName"]]
        .assign(
            genusInTax=lambda d: (
                d["scientificName"]
                .str.split()
                .str[0]
                .map(normalize_name)
                .isin(tax_genera)
            )
        )
        .sort_values("scientificName")
        .reset_index(drop=True)
    )
    return NameResolutionSummary(
        n_names=len(names),
        n_species=names["_accepted"].nunique(),
        renamed=renamed,
        not_found=not_found,
    )


### WORMS SEARCH ###


@save_to_db(
    table_name="worms_search",
    cache=PartialCache(
        items_kwarg="genera_list",
        item_key="genus",
        params={"worms_marine_only": "marine_only"},
    ),
)
def run_worms_search(
    genera_list: list[str],
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    marine_only: bool = True,
    failure_sink: list | None = None,
) -> pd.DataFrame:
    """Query WoRMS for species belonging to the given genera.

    Args:
        genera_list: List of genera to query in WoRMS.
        progress_handler: Optional progress tracker (dict with 'current' key,
            or object with .update() method).
        user_agent: User-Agent header for HTTP requests.
        marine_only: If True (default), WoRMS returns only marine-flagged
            species; uncheck to include non-marine WoRMS records.
        failure_sink: If provided, genera whose query errored are appended here
            so the cache layer skips them and retries on the next run.

    Returns:
        DataFrame with WoRMS species data for the requested genera.
    """
    genus_word = "genus" if len(genera_list) == 1 else "genera"
    logger.info(f"Starting WoRMS search: {len(genera_list)} unique {genus_word}")

    worms_search_df, failed_genera = get_species_from_genus_list(
        genera_list,
        delay=0.1,
        progress_handler=progress_handler,
        user_agent=user_agent,
        marine_only=marine_only,
    )

    if failed_genera:
        logger.warning(f"{len(failed_genera)} genera failed in WoRMS search")
        if failure_sink is not None:
            failure_sink.extend(failed_genera)

    logger.info(f"WoRMS search completed: retrieved {len(worms_search_df)} species")
    return worms_search_df


### WORMS MERGE ###


@save_to_db(
    table_name="worms_merge",
    cache=FullCache(
        inherit_from=["mol_params", "worms_search_params"],
        local={"worms_resolve_names": "resolve_names"},
        fingerprint_on=["worms_search_df"],
    ),
)
@preserve_sequence_order("seq_id", "mol_df")
def run_worms_merge(
    worms_search_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    mol_params: dict | None = None,
    worms_search_params: dict | None = None,
    resolve_names: bool = True,
) -> pd.DataFrame:
    """Merge WoRMS species with MOL hits by genus, preserving scores for exact matches.

    Cross-joins WoRMS species with MOL hits on genus, then marks which WoRMS
    species were also found directly by BLAST (in_mol). Identity and query_cover
    scores are kept only for exact name matches; all other rows get NaN.

    Args:
        worms_search_df: WoRMS species DataFrame from run_worms_search().
        mol_df: MOL DataFrame from finalize_mol_results().
        mol_params: Parameter dict from finalize_mol_results, used for caching only.
        worms_search_params: Parameter dict from run_worms_search, propagated into
            this step's cache key (caching only).
        resolve_names: Whether name resolution (R3) ran; recorded in the cache key
            so toggling it re-runs the merge.

    Returns:
        Merged DataFrame with WoRMS taxonomy and MOL scores where applicable.
    """
    if mol_params is None:
        mol_params = {}

    # Match on the WoRMS-accepted MOL name (R3); fall back to the raw name when
    # resolution did not run.
    mol_df = mol_df.copy()
    if "acceptedName" not in mol_df.columns:
        mol_df["acceptedName"] = mol_df["scientificName"]

    # 1a. Top hit (scores) per seq_id + accepted name.
    mol_subset = top_hit_per_group(
        mol_df[
            [
                "seq_id",
                "dna_sequence",
                "acceptedName",
                "identity_percentage",
                "query_cover",
            ]
        ],
        keys=["seq_id", "acceptedName"],
        sort=["identity_percentage", "query_cover"],
    )

    # 1b. NCBI synonyms: the raw NCBI names that differ from the accepted name
    # (e.g. accepted Gadus macrocephalus also hit under the synonym Gadus ogac).
    # Blank when NCBI already used the accepted name. Surfaces that a synonym was
    # involved regardless of which raw name was the top hit.
    syn = mol_df[
        mol_df["scientificName"].map(normalize_name)
        != mol_df["acceptedName"].map(normalize_name)
    ]
    if not syn.empty:
        syn_names = (
            syn.groupby(["seq_id", "acceptedName"], observed=False)["scientificName"]
            .agg(lambda s: ", ".join(sorted(set(s))))
            .reset_index()
            .rename(columns={"scientificName": "verbatimIdentification"})
        )
        mol_subset = mol_subset.merge(
            syn_names, on=["seq_id", "acceptedName"], how="left"
        )
    else:
        mol_subset["verbatimIdentification"] = np.nan

    # 2. Merge on the accepted genus (first token of the accepted name), case /
    # whitespace insensitive. No column clashes (mol_subset carries no genus or
    # scientificName), so no _mol suffixes are produced.
    worms_search_df = worms_search_df.assign(
        _join_key=worms_search_df["genus"].map(normalize_name)
    )
    mol_subset = mol_subset.assign(
        _join_key=mol_subset["acceptedName"].str.split().str[0].map(normalize_name)
    )
    merged_df = worms_search_df.merge(
        mol_subset, on="_join_key", how="left", suffixes=("", "_mol")
    ).drop(columns="_join_key")

    # 3. Define the Match (Vectorized): the WoRMS species equals the MOL hit's
    # accepted name, so synonyms (e.g. Gadus ogac) credit the accepted species.
    merged_df["in_mol"] = (
        merged_df["scientificName"].str.lower().str.strip()
        == merged_df["acceptedName"].str.lower().str.strip()
    )

    # 4. Protect MOL-only fields: scores and the verbatim name are meaningful
    # only where a direct MOL hit matched (in_mol).
    score_cols = ["identity_percentage", "query_cover"]
    merged_df.loc[~merged_df["in_mol"], score_cols] = np.nan
    merged_df.loc[~merged_df["in_mol"], "verbatimIdentification"] = np.nan

    # 5. Deduplicate, Clean and Rename
    merged_df = (
        merged_df.sort_values(
            ["scientificName", "in_mol", "identity_percentage"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["seq_id", "scientificName"])
        .drop(columns=["acceptedName"])
    )
    rename_map = {
        "identity_percentage": "mol_top_identity_percentage",
        "query_cover": "mol_top_query_cover",
    }

    # Standardize column order
    first_cols = ["seq_id", "dna_sequence"]
    remaining = [c for c in merged_df.columns if c not in first_cols]
    merged_df = merged_df[first_cols + remaining].rename(columns=rename_map)

    logger.info(
        f"Merged {len(merged_df)} rows. "
        f"({merged_df['in_mol'].sum()} exact species matches found)"
    )

    return merged_df.reset_index(drop=True)


### FINALIZE ###


@save_to_db(
    "results_tax",
    cache=FullCache(inherit_from=["worms_merge_params"]),
)
def finalize_tax_results(
    worms_merge_df: pd.DataFrame, worms_merge_params: dict | None = None
) -> pd.DataFrame:
    """Finalize the TAX list for export or downstream analysis.

    Args:
        worms_merge_df: Merged DataFrame from run_worms_merge().
        worms_merge_params: Parameter dict from run_worms_merge, used for caching only.

    Returns:
        Finalized TAX DataFrame.
    """
    if worms_merge_params is None:
        worms_merge_params = {}

    tax_df = worms_merge_df.copy()
    logger.success(f"Finalized TAX list: {len(tax_df)} rows")
    return tax_df


### SUMMARY / EXPLORATION ###


@preserve_sequence_order("seq_id", "tax_df")
def build_tax_summary(
    tax_df: pd.DataFrame,
    mol_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-sequence summary of TAX (and optionally MOL) species counts.

    Args:
        tax_df: TAX results from finalize_tax_results().
        mol_df: Optional MOL results with 'seq_id' and 'scientificName' columns.

    Returns:
        DataFrame with one row per seq_id.
        If mol_df is provided: columns seq_id, MOL Species, TAX Species, Overlap, New Species.
        Otherwise: columns seq_id, TAX Species.
    """
    summary_data: list[dict] = []

    # Pre-compute TAX species sets per seq_id
    tax_grouped = (
        tax_df.dropna(subset=["seq_id"])
        .groupby("seq_id", observed=False)["scientificName"]
        .agg(lambda s: set(s.unique()))
    )

    # If no MOL data, just count TAX species
    if mol_df is None:
        for seq_id, tax_species in sorted(tax_grouped.items(), key=lambda x: x[0]):
            summary_data.append(
                {
                    "seq_id": seq_id,
                    "TAX Species": len(tax_species),
                }
            )
        return pd.DataFrame(summary_data)

    # Count MOL species by their WoRMS-accepted name when resolution ran, so the
    # overlap with the (accepted) TAX list reflects synonyms correctly.
    mol_name_col = (
        "acceptedName" if "acceptedName" in mol_df.columns else "scientificName"
    )
    mol_grouped = (
        mol_df.dropna(subset=["seq_id"])
        .groupby("seq_id", observed=False)[mol_name_col]
        .agg(lambda s: set(s.unique()))
    )

    for seq_id, tax_species in sorted(tax_grouped.items(), key=lambda x: x[0]):
        mol_species = mol_grouped.get(seq_id, set())

        overlap = len(mol_species & tax_species)
        new_species = len(tax_species - mol_species)

        summary_data.append(
            {
                "seq_id": seq_id,
                "MOL Species": len(mol_species),
                "TAX Species": len(tax_species),
                "Overlap": overlap,
                "New Species": new_species,
            }
        )

    return pd.DataFrame(summary_data)


### EMPTY INPUT ###


def empty_tax_results(mol_df: pd.DataFrame) -> pd.DataFrame:
    """Return the TAX list of an empty MOL list, with its usual columns.

    The TAX steps run on no genus and without the database, so nothing is
    queried or cached.
    """
    worms_df, _ = run_worms_search([], db_path=None)
    merge_df, _ = run_worms_merge(worms_df, mol_df, mol_params={}, db_path=None)
    tax_df, _ = finalize_tax_results(merge_df, db_path=None)
    return tax_df
