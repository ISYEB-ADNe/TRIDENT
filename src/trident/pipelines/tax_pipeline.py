"""WoRMS pipeline: the TAX (taxonomic scope) list.

Step order:
    0. prepare_resolution_input(mol_df), prepare_resolution_taxids(mol_df)
       run_name_resolution(names, ...)          → (resolution_df, params)
       build_name_table(resolution_df, taxids)  → name_table
    1. prepare_worms_input(mol_df, name_table)   → list[str]
    2. run_worms_search(genera_list, ...)         → (worms_search_df, params)
    3. run_worms_merge(worms_search_df, ...)      → (worms_merge_df, params)
    4. finalize_tax_results(worms_merge_df, ...)  → (tax_df, params)

Exploration:
    build_tax_summary(tax_df, mol_df, name_table)
    mol_rejected_rows(tax_df, ncbi_search_df, name_table)

Names: build_name_table (see names.py) makes the name table of the MOL names
from run_name_resolution; the later steps take it to compare names.
"""

from collections.abc import Callable
from typing import NamedTuple

import pandas as pd
import numpy as np

from loguru import logger

from trident.core.database import save_to_db, PartialCache, FullCache
from trident.core.http import create_session
from trident.core.utils import (
    StepLog,
    as_ids,
    pluralize,
    species_for_sequences,
    preserve_sequence_order,
    normalize_name,
    notify_progress,
)

from trident.clients.worms import (
    WORMS_TAXON_COLUMNS,
    get_species_from_genus_list,
    resolve_accepted_name,
    worms_search_url,
)
from trident.pipelines.names import (
    best_hits,
    name_keys,
    resolved_names,
    species_key,
    species_of,
)


### DATA PREPARATION ###


def _genus_of(name) -> str | None:
    """First word of a name; None for a missing or empty name."""
    return name.split()[0] if isinstance(name, str) and name else None


def prepare_worms_input(df: pd.DataFrame, name_table: pd.DataFrame | None) -> list[str]:
    """Extract unique genera from the MOL DataFrame for WoRMS querying.

    The genus of a name is the one of its species (see species_of), so a
    synonym reassigned to a different genus (e.g. Allocentrotus ->
    Strongylocentrotus) expands the correct genus.

    Args:
        df: DataFrame with a 'scientificName' column (NCBI names).
        name_table: See build_name_table; None before names are checked.

    Returns:
        List of unique genera (non-null) to query in WoRMS.
    """
    genera = species_of(df["scientificName"], name_table).map(_genus_of)
    # A hit without a name has no genus to expand.
    genera = genera.dropna().unique().tolist()
    genus_word = pluralize(len(genera), "genus")
    logger.debug(f"Prepared WoRMS input: {len(genera)} unique {genus_word}")
    return genera


### NAME RESOLUTION ###


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
    log_as=("WoRMS name check", "MOL name"),
    bool_columns=("is_synonym",),
)
def run_name_resolution(
    names: list[str],
    ncbi_taxids: dict[str, int] | None = None,
    user_agent: str | None = None,
    progress_handler: dict | object | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Resolve MOL (NCBI) names to their WoRMS accepted name + AphiaID.

    One AphiaRecordsByName lookup per unique name (cached per item, so only new
    names cost a call). A name WoRMS does not know, or whose records give no
    single accepted species, is looked up a second time through its NCBI
    taxid, when given. Names found neither way are returned
    unchanged (acceptedName == input, no AphiaID). Only genuine network/HTTP
    errors reach failure_sink; a 204 or 404 is a valid not-found, not a failure.

    Args:
        names: Unique MOL scientific names to resolve.
        ncbi_taxids: Optional NCBI taxid per name (prepare_resolution_taxids).
        user_agent: User-Agent header for the WoRMS requests.
        progress_handler: Optional progress tracker (a dict with "current", or
            tqdm-like).
        failure_sink: If provided, names whose query errored are appended so the
            cache layer skips them and retries next run.
        result_sink: Given by save_to_db, which saves each name as soon as
            it is resolved.

    Returns:
        DataFrame with columns scientificName, acceptedName, acceptedNameUsageID,
        is_synonym, matchedBy ('name', 'ncbi_taxid', or empty when not found),
        note (why records of a name not found did not resolve it), one row
        per input name.
    """
    session = create_session(user_agent=user_agent)
    ncbi_taxids = ncbi_taxids or {}
    rows: list[dict] = []
    failed: list[str] = []
    log = StepLog("WoRMS name check", len(names))
    try:
        for name in names:
            try:
                with log.query(name):
                    resolved = resolve_accepted_name(
                        name, ncbi_taxid=ncbi_taxids.get(name), session=session
                    )
            except Exception as exc:
                log.failed(name, f"{exc} (check: {worms_search_url(name)})")
                failed.append(name)
                continue
            finally:
                notify_progress(progress_handler)
            if not resolved["matched_by"]:
                link = worms_search_url(name)
                why = resolved["note"] or "not in WoRMS"
                log.item(name, f"{why} (check: {link})", notable=True)
            elif resolved["is_synonym"]:
                log.item(name, f"synonym of {resolved['accepted_name']}")
            else:
                log.item(name, "accepted name")
            row = {
                "scientificName": name,
                "acceptedName": resolved["accepted_name"],
                "acceptedNameUsageID": resolved["aphia_id"],
                "is_synonym": resolved["is_synonym"],
                "matchedBy": resolved["matched_by"],
                "note": resolved["note"],
            }
            rows.append(row)
            if result_sink is not None:
                result_sink(pd.DataFrame([row]), [name])
    finally:
        session.close()

    if failed and failure_sink is not None:
        failure_sink.extend(failed)

    n_syn = sum(1 for r in rows if r["is_synonym"])
    n_taxid = sum(1 for r in rows if r["matchedBy"] == "ncbi_taxid")
    log.done(
        f"{len(rows)} {pluralize(len(rows), 'name')} checked ({n_syn} "
        f"{pluralize(n_syn, 'synonym')}, {n_taxid} found through their NCBI taxid)",
        failed,
        "name",
    )
    return pd.DataFrame(
        rows,
        columns=[
            "scientificName",
            "acceptedName",
            "acceptedNameUsageID",
            "is_synonym",
            "matchedBy",
            "note",
        ],
    )


class NameResolutionSummary(NamedTuple):
    """What the TAX tab reports about the names of the MOL list."""

    n_names: int  # distinct MOL (NCBI) names
    n_species: int  # distinct WoRMS accepted names they resolve to
    renamed: pd.DataFrame  # scientificName, accepted_name
    not_found: pd.DataFrame  # scientificName, note, genus_in_tax: names unresolved


def summarize_name_resolution(
    mol_df: pd.DataFrame,
    name_table: pd.DataFrame | None,
    tax_df: pd.DataFrame | None = None,
) -> NameResolutionSummary:
    """Relate the NCBI names of the MOL list to the WoRMS names TAX works on.

    TAX compares and expands the WoRMS accepted names, so two NCBI names that
    WoRMS treats as one species count once there (e.g. Gadus ogac and Gadus
    macrocephalus). Names are compared case- and whitespace-insensitively, and
    hits without a name are left out. A name whose WoRMS lookup failed has no
    row in the name table: it is retried on the next run, not reported as not
    found.

    Args:
        mol_df: MOL DataFrame (scientificName: NCBI names).
        name_table: See build_name_table.
        tax_df: TAX list, used to tell whether the genus of a name not found in
            WoRMS was still expanded (so its sequence lost no candidates).

    Returns:
        A NameResolutionSummary. renamed has one row per MOL name whose WoRMS
        accepted name differs; not_found one row per MOL name WoRMS did not
        resolve, sorted, with the note saying why when WoRMS has records of it
        (see resolve_accepted_name), and genus_in_tax True when its genus is in
        the TAX list.
    """
    raw = mol_df["scientificName"].dropna()
    table = resolved_names(name_table).drop_duplicates("name_key")

    is_renamed = table["name_key"] != species_key(table["species"], None)
    renamed = (
        table.loc[is_renamed, ["name", "species"]]
        .rename(columns={"name": "scientificName", "species": "accepted_name"})
        .sort_values("scientificName")
        .reset_index(drop=True)
    )

    tax_genera = (
        set(name_keys(tax_df["genus"].dropna()))
        if tax_df is not None and "genus" in tax_df.columns
        else set()
    )
    not_found = (
        table.loc[table["species_id"].isna(), ["name", "note"]]
        .rename(columns={"name": "scientificName"})
        .assign(
            genus_in_tax=lambda d: (
                d["scientificName"].map(_genus_of).map(normalize_name).isin(tax_genera)
            )
        )
        .sort_values("scientificName")
        .reset_index(drop=True)
    )
    return NameResolutionSummary(
        n_names=name_keys(raw).nunique(),
        n_species=species_key(raw, name_table).nunique(),
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
    log_as=("WoRMS search", "genus"),
)
def run_worms_search(
    genera_list: list[str],
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    marine_only: bool = True,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Query WoRMS for species belonging to the given genera.

    Args:
        genera_list: List of genera to query in WoRMS.
        progress_handler: Optional progress tracker (a dict with "current", or
            tqdm-like).
        user_agent: User-Agent header for HTTP requests.
        marine_only: If True (default), WoRMS returns only marine-flagged
            species; False also returns non-marine records.
        failure_sink: If provided, genera whose query errored are appended here
            so the cache layer skips them and retries on the next run.
        result_sink: Given by save_to_db, which saves each genus as soon as
            its species are fetched.

    Returns:
        DataFrame with WoRMS species data for the requested genera.
    """
    worms_search_df, failed_genera = get_species_from_genus_list(
        genera_list,
        delay=0.1,
        progress_handler=progress_handler,
        user_agent=user_agent,
        marine_only=marine_only,
        result_sink=result_sink,
    )

    if failed_genera and failure_sink is not None:
        failure_sink.extend(failed_genera)

    return worms_search_df


### GENUS HOMONYMS ###


def genus_homonyms(worms_search_df: pd.DataFrame, tax_df: pd.DataFrame) -> pd.DataFrame:
    """Report the genus names expanded as several homonyms, and what TAX kept.

    run_worms_search expands every accepted genus of a name (e.g. Liparis, a
    fish and a plant), and run_worms_merge keeps per sequence the ones its MOL
    species belong to (all of them when none does).

    Args:
        worms_search_df: WoRMS species from run_worms_search(), with genusID.
        tax_df: TAX list from run_worms_merge(), with genusID and in_mol.

    Returns:
        One row per homonym of the genus names with several: genus, genusID,
        kingdom, class, n_species, mol_species (the MOL species found in it),
        sequences ("kept/all": the sequences with this genus name whose TAX
        list keeps it). Empty when there is no homonym.
    """
    columns = [
        "genus",
        "genusID",
        "kingdom",
        "class",
        "n_species",
        "mol_species",
        "sequences",
    ]
    if any("genusID" not in df.columns for df in (worms_search_df, tax_df)):
        return pd.DataFrame(columns=columns)

    species = worms_search_df.assign(
        genusID=as_ids(worms_search_df["genusID"]),
        _genus=worms_search_df["genus"].map(normalize_name),
    ).dropna(subset=["genusID"])
    groups = (
        species.groupby(["_genus", "genusID"], observed=True)
        .agg(
            genus=("genus", "first"),
            kingdom=("kingdom", "first"),
            **{"class": ("class", "first")},
            n_species=("scientificName", "nunique"),
        )
        .reset_index()
    )
    groups = groups[groups.groupby("_genus")["genusID"].transform("size") > 1]
    if groups.empty:
        return pd.DataFrame(columns=columns)

    tax = tax_df.assign(
        _gid=as_ids(tax_df["genusID"]), _genus=tax_df["genus"].map(normalize_name)
    )
    kept = tax.groupby("_gid")["seq_id"].nunique()
    total = tax.groupby("_genus")["seq_id"].nunique()
    mol_species = (
        tax[tax["in_mol"]]
        .groupby("_gid")["scientificName"]
        .agg(lambda names: ", ".join(sorted(set(names))))
    )
    groups["mol_species"] = groups["genusID"].map(mol_species).fillna("")
    groups["sequences"] = [
        f"{int(kept.get(gid, 0))}/{int(total.get(genus, 0))}"
        for gid, genus in zip(groups["genusID"], groups["_genus"])
    ]
    return groups[columns].sort_values(["genus", "genusID"]).reset_index(drop=True)


### WORMS MERGE ###

# The columns of a sequence's best MOL hit on a species, as TAX names them
# (and the GEO list and the exports after it).
MOL_HIT_COLUMNS = {
    "identity_percentage": "mol_identity",
    "query_cover": "mol_query_cover",
    "hit_url": "mol_hit_url",
}

# The columns of the TAX list: each WoRMS species expanded for a sequence,
# with the sequence's best MOL hit on it when it is a MOL species (in_mol).
TAX_COLUMNS = [
    "seq_id",
    "dna_sequence",
    *WORMS_TAXON_COLUMNS,
    *MOL_HIT_COLUMNS.values(),
    "in_mol",
]


@save_to_db(
    table_name="worms_merge",
    cache=FullCache(
        inherit_from=["mol_params", "worms_search_params"],
        fingerprint_on=["worms_search_df", "name_table"],
    ),
    bool_columns=("in_mol",),
)
@preserve_sequence_order("seq_id", "mol_df")
def run_worms_merge(
    worms_search_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    name_table: pd.DataFrame | None,
    mol_params: dict | None = None,
    worms_search_params: dict | None = None,
) -> pd.DataFrame:
    """Pair each sequence's MOL species with the WoRMS species of their genera.

    A WoRMS species that is one of the sequence's MOL species (in_mol,
    synonyms included through the name table) keeps the sequence's best MOL
    hit on it (MOL_HIT_COLUMNS); the others have none. Of a genus name
    expanded as several homonyms, a sequence keeps those its MOL species
    belong to, all of them when none does.

    Args:
        worms_search_df: WoRMS species DataFrame from run_worms_search().
        mol_df: MOL DataFrame from finalize_mol_results().
        name_table: See build_name_table: the MOL names match on their species,
            so synonyms (e.g. Gadus ogac) credit the accepted species.
        mol_params: Params of finalize_mol_results, for the cache key only.
        worms_search_params: Params of run_worms_search, for the cache key only.

    Returns:
        Merged DataFrame with WoRMS taxonomy and MOL scores where applicable.
    """
    if mol_params is None:
        mol_params = {}

    # 1. Top hit (scores) per seq_id and species of the MOL names.
    mol_subset = best_hits(
        mol_df, name_table, ["dna_sequence", *MOL_HIT_COLUMNS]
    ).rename(columns={"species_key": "_species"})

    # 2. Merge on the genus of the species, case / whitespace insensitive. No
    # column clashes (mol_subset carries no genus or scientificName), so no
    # _mol suffixes are produced.
    # The search's own columns, also for a search saved before one existed.
    worms_search_df = worms_search_df.reindex(columns=WORMS_TAXON_COLUMNS).assign(
        _join_key=name_keys(worms_search_df["genus"])
    )
    mol_subset = mol_subset.assign(_join_key=mol_subset["_species"].map(_genus_of))
    merged_df = worms_search_df.merge(
        mol_subset, on="_join_key", how="left", suffixes=("", "_mol")
    )

    # 3. The WoRMS species is a MOL species of the sequence.
    merged_df["in_mol"] = (
        species_key(merged_df["scientificName"], name_table) == merged_df["_species"]
    )

    # 4. Genus homonyms (run_worms_search expands each accepted genus of a
    # name): a sequence keeps the ones its MOL species belong to, all of them
    # when none of its species is in any.
    if "genusID" in merged_df.columns:
        merged_df["_gid"] = as_ids(merged_df["genusID"])
        groups = ["seq_id", "_join_key"]
        merged_df["_hit"] = merged_df.groupby([*groups, "_gid"], dropna=False)[
            "in_mol"
        ].transform("any")
        any_hit = merged_df.groupby(groups, dropna=False)["_hit"].transform("any")
        merged_df = merged_df[~any_hit | merged_df["_hit"]].drop(
            columns=["_gid", "_hit"]
        )
    merged_df = merged_df.drop(columns="_join_key")

    # 5. Scores are meaningful only where a direct MOL hit matched (in_mol).
    merged_df.loc[~merged_df["in_mol"], list(MOL_HIT_COLUMNS)] = np.nan

    # 6. Deduplicate, Clean and Rename
    merged_df = (
        merged_df.sort_values(
            ["scientificName", "in_mol", "identity_percentage"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["seq_id", "scientificName"])
        .drop(columns=["_species"])
    )
    return merged_df.rename(columns=MOL_HIT_COLUMNS)[TAX_COLUMNS].reset_index(drop=True)


### FINALIZE ###


def mol_rejected_rows(
    tax_df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """TAX rows of species that BLAST returned for that same sequence but MOL rejected.

    The test is per (seq_id, species) pair: a species can be a MOL hit for one
    sequence and rejected for another. A TAX row that is not a MOL hit
    (in_mol) but whose species BLAST returned for its sequence (see
    found_by_blast) was rejected by the MOL filter.

    Args:
        tax_df: TAX list with 'seq_id', 'scientificName' and 'in_mol'.
        ncbi_search_df: Raw BLAST hits, before the MOL filter.
        name_table: See build_name_table.

    Returns:
        The matching tax_df rows (original index kept), with the best hit
        BLAST gave the species for that sequence (MOL_HIT_COLUMNS: a hit the
        MOL filter rejected, where TAX's own rows have MOL hits).
    """
    score_cols = list(MOL_HIT_COLUMNS.values())
    if ncbi_search_df is None or ncbi_search_df.empty or tax_df.empty:
        return tax_df.iloc[:0].assign(**{c: pd.Series(dtype=float) for c in score_cols})

    seen = best_hits(ncbi_search_df, name_table, list(MOL_HIT_COLUMNS)).rename(
        columns={"species_key": "_key", **MOL_HIT_COLUMNS}
    )

    expanded = tax_df[~tax_df["in_mol"]].drop(columns=score_cols, errors="ignore")
    rejected = (
        expanded.assign(_key=species_key(expanded["scientificName"], name_table))
        .reset_index()
        .merge(seen, on=["seq_id", "_key"], how="inner")
        .set_index("index")
        .drop(columns="_key")
    )
    rejected.index.name = None
    return rejected


@save_to_db(
    "results_tax",
    cache=FullCache(
        inherit_from=["worms_merge_params"],
        local={"tax_drop_mol_rejected": "drop_mol_rejected"},
        fingerprint_on=["name_table"],
    ),
    log_reuse_as="TAX list",
    bool_columns=("in_mol",),
)
def finalize_tax_results(
    worms_merge_df: pd.DataFrame,
    worms_merge_params: dict | None = None,
    *,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
    drop_mol_rejected: bool = True,
) -> pd.DataFrame:
    """Finalize the TAX list for export or downstream analysis.

    Args:
        worms_merge_df: Merged DataFrame from run_worms_merge().
        worms_merge_params: Params of run_worms_merge, for the cache key only.
        ncbi_search_df: Raw BLAST hits, to find the species MOL rejected.
        name_table: See build_name_table.
        drop_mol_rejected: Remove the species that BLAST returned for a
            sequence but MOL rejected, for that sequence only.

    Returns:
        Finalized TAX DataFrame.
    """
    if worms_merge_params is None:
        worms_merge_params = {}

    tax_df = worms_merge_df.copy()
    if drop_mol_rejected:
        rejected = mol_rejected_rows(tax_df, ncbi_search_df, name_table)
        if not rejected.empty:
            n_rejected = rejected["scientificName"].nunique()
            logger.info(
                f"TAX list: {n_rejected} {pluralize(n_rejected, 'species')} removed "
                "where BLAST found them but the MOL filter rejected them"
            )
        tax_df = tax_df.drop(index=rejected.index).reset_index(drop=True)
    n_mol = (
        tax_df.loc[tax_df["in_mol"], "scientificName"].nunique()
        if "in_mol" in tax_df
        else 0
    )
    logger.success(
        f"TAX list: {species_for_sequences(tax_df)}, {n_mol} of them in the MOL list"
    )
    return tax_df


### SUMMARY / EXPLORATION ###


@preserve_sequence_order("seq_id", "tax_df")
def build_tax_summary(
    tax_df: pd.DataFrame,
    mol_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a per-sequence summary of TAX (and optionally MOL) species counts.

    MOL names count as their species (see species_of), so a synonym overlaps
    with its TAX species.

    Args:
        tax_df: TAX results from finalize_tax_results().
        mol_df: Optional MOL results with 'seq_id' and 'scientificName' columns.
        name_table: See build_name_table.

    Returns:
        DataFrame with one row per seq_id.
        If mol_df is provided: columns seq_id, n_mol_species, n_tax_species, n_both
        (in both), n_added (TAX only).
        Otherwise: columns seq_id, n_tax_species.
    """
    summary_data: list[dict] = []

    # Pre-compute TAX species sets per seq_id
    tax = tax_df.dropna(subset=["seq_id"])
    tax_grouped = (
        species_key(tax["scientificName"], name_table)
        .groupby(tax["seq_id"], observed=False)
        .agg(lambda s: set(s.dropna().unique()))
    )

    # If no MOL data, just count TAX species
    if mol_df is None:
        for seq_id, tax_species in sorted(tax_grouped.items(), key=lambda x: x[0]):
            summary_data.append(
                {
                    "seq_id": seq_id,
                    "n_tax_species": len(tax_species),
                }
            )
        return pd.DataFrame(summary_data)

    mol = mol_df.dropna(subset=["seq_id"])
    mol_grouped = (
        species_key(mol["scientificName"], name_table)
        .groupby(mol["seq_id"], observed=False)
        .agg(lambda s: set(s.dropna().unique()))
    )

    for seq_id, tax_species in sorted(tax_grouped.items(), key=lambda x: x[0]):
        mol_species = mol_grouped.get(seq_id, set())

        overlap = len(mol_species & tax_species)
        new_species = len(tax_species - mol_species)

        summary_data.append(
            {
                "seq_id": seq_id,
                "n_mol_species": len(mol_species),
                "n_tax_species": len(tax_species),
                "n_both": overlap,
                "n_added": new_species,
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
    merge_df, _ = run_worms_merge(worms_df, mol_df, None, mol_params={}, db_path=None)
    tax_df, _ = finalize_tax_results(
        merge_df, ncbi_search_df=None, name_table=None, db_path=None
    )
    return tax_df
