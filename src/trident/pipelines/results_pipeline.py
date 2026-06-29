"""Results assembly — combines MOL+GEO and HYPO species into a unified results table.

Unlike other pipelines, this module performs read-only aggregation of existing
pipeline outputs. No external searches or database caching (except GBIF export
which does WoRMS lookups for higher-level taxa).

    build_results_df(sequences_df, geo_df, mol_df, ...) → DataFrame
    find_sequence_exclusion_step(all_seq_ids, ...) → DataFrame
    build_gbif_export_df(results_df) → DataFrame (one row per MOTU/ASV)
    add_below_mol(df, mol_df, ncbi_search_df) → DataFrame
    add_low_identity_warning(df, threshold) → DataFrame
"""

import pandas as pd
from loguru import logger

from trident.clients.worms import clean_authorships, get_aphia_record
from trident.core.utils import (
    ensure_columns,
    find_exclusion_pipeline_step,
    preserve_sequence_order,
)


RESULT_COLS = [
    "seq_id",
    "validation_step",
    "scientificName",
    "taxonURL",
    "gbif_taxonURL",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specificEpithet",
    "scientificNameAuthorship",
    "ncbi_top_identity_percentage",
    "ncbi_top_query_cover",
    "ncbi_top_hit_url",
    "bold_seq_url",
    "gbif_occurrences",
]

# Column order for CSV/file export
EXPORT_COLS = [
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
    "validation_step",
    "ncbi_top_identity_percentage",
    "ncbi_top_query_cover",
    "gbif_occurrences",
    "proxy_identity_percentage",
    "proxy_query_cover",
    "taxonURL",
    "gbif_taxonURL",
    "ncbi_top_hit_url",
    "bold_seq_url",
]


def get_rejected_max_identity(mol_df, ncbi_search_df) -> pd.Series:
    """Return max identity among MOL-rejected hits, per seq_id."""
    if ncbi_search_df is None or ncbi_search_df.empty or mol_df is None or mol_df.empty:
        return pd.Series(dtype=float, name="_rejected_max")
    mol_keys = set(
        zip(mol_df["seq_id"], mol_df["scientificName"], mol_df["identity_percentage"])
    )
    # Same set-membership test as a row-wise apply, but iterate columns directly
    # to avoid building a Series per row. Semantics are identical (same tuples,
    # same NaN != NaN behaviour as the set of tuples).
    in_mol = [
        (seq_id, name, identity) in mol_keys
        for seq_id, name, identity in zip(
            ncbi_search_df["seq_id"],
            ncbi_search_df["scientificName"],
            ncbi_search_df["identity_percentage"],
        )
    ]
    rejected = ncbi_search_df[~pd.Series(in_mol, index=ncbi_search_df.index)]
    if rejected.empty:
        return pd.Series(dtype=float, name="_rejected_max")
    return (
        rejected.groupby("seq_id")["identity_percentage"].max().rename("_rejected_max")
    )


def add_below_mol(
    df,
    mol_df,
    ncbi_search_df,
    identity_col="ncbi_top_identity_percentage",
):
    """Flag rows where identity is at or below the highest rejected MOL hit."""
    df["below_mol"] = False
    if identity_col not in df.columns:
        return df
    rejected_max = get_rejected_max_identity(mol_df, ncbi_search_df)
    if rejected_max.empty:
        return df
    df = df.merge(rejected_max, on="seq_id", how="left")
    df[identity_col] = pd.to_numeric(df[identity_col], errors="coerce")
    df["_rejected_max"] = pd.to_numeric(df["_rejected_max"], errors="coerce")
    df["below_mol"] = (
        df[identity_col].notna()
        & df["_rejected_max"].notna()
        & (df[identity_col] <= df["_rejected_max"])
    )
    return df.drop(columns="_rejected_max")


def add_low_identity_warning(
    df, threshold, identity_col="ncbi_top_identity_percentage"
):
    """Flag rows where identity is present and below threshold."""
    df[identity_col] = pd.to_numeric(df[identity_col], errors="coerce")
    # `lt` returns False (not NA) for NaN, and `&` of two boolean Series is
    # always a clean boolean, so no fillna is needed.
    df["low_identity_warning"] = (
        df[identity_col].lt(threshold) & df[identity_col].notna()
    )
    return df


def _select_columns(df, extra=None):
    """Keep only target columns that exist in df."""
    target = list(EXPORT_COLS) + (extra or [])
    return df[[c for c in target if c in df.columns]].copy()


def _build_mol_geo_part(geo_df, mol_df):
    """Build MOL+GEO rows: species confirmed via NCBI + geographically validated."""
    mol_geo = geo_df[geo_df["in_mol"].astype(bool)].copy()
    mol_geo["validation_step"] = "MOL+GEO"

    # GEO carries the WoRMS-accepted name, so match the best MOL hit on the
    # accepted name (R3) to attach NCBI scores to synonyms; fall back to raw.
    mol_df = mol_df.copy()
    match_name = (
        "acceptedName" if "acceptedName" in mol_df.columns else "scientificName"
    )

    best_mol_cols = ["seq_id", match_name, "identity_percentage", "query_cover"]
    if "hit_url" in mol_df.columns:
        best_mol_cols.append("hit_url")
    best_mol = (
        mol_df.sort_values("identity_percentage", ascending=False)
        .drop_duplicates(subset=["seq_id", match_name])[best_mol_cols]
        .rename(
            columns={
                match_name: "scientificName",
                "identity_percentage": "ncbi_top_identity_percentage",
                "query_cover": "ncbi_top_query_cover",
                "hit_url": "ncbi_top_hit_url",
            }
        )
    )
    return mol_geo.merge(best_mol, on=["seq_id", "scientificName"], how="left")


def _build_hypo_part(hypo_df, geo_df):
    """Build HYPO rows: species confirmed via BOLD/NCBI proxy."""
    hypo = hypo_df.copy()
    hypo["validation_step"] = "HYPO"

    hypo = hypo.rename(
        columns={
            "identity_percentage": "proxy_identity_percentage",
            "query_cover": "proxy_query_cover",
            "seq_url": "bold_seq_url",
        }
    )

    # Fill gbif_occurrences and gbif_taxonURL from geo_df (HYPO doesn't carry them)
    geo_fill_cols = [
        c for c in ("gbif_occurrences", "gbif_taxonURL") if c in geo_df.columns
    ]
    if geo_fill_cols:
        geo_fill = geo_df[["seq_id", "scientificName"] + geo_fill_cols].drop_duplicates(
            subset=["seq_id", "scientificName"]
        )
        hypo = hypo.merge(geo_fill, on=["seq_id", "scientificName"], how="left")

    return hypo


@preserve_sequence_order("seq_id", "sequences_df")
def build_results_df(sequences_df, geo_df, mol_df, hypo_df=None, ncbi_search_df=None):
    """Build combined results DataFrame from MOL+GEO and HYPO species.

    Progressive: works with just geo_df + mol_df, adds HYPO rows when available.
    Empty sequences (no species assigned) get a row with just seq_id.

    Args:
        sequences_df: Original sequences DataFrame (used for ordering and full seq_id list).
        geo_df: GEO results (must have 'in_mol' column).
        mol_df: MOL results (NCBI BLAST hits).
        hypo_df: HYPO results (optional, proxy-validated species).
        ncbi_search_df: Raw NCBI search results (optional, used for below_mol flag).
    """
    parts = [_select_columns(_build_mol_geo_part(geo_df, mol_df))]

    if hypo_df is not None and not hypo_df.empty:
        hypo_part = _select_columns(_build_hypo_part(hypo_df, geo_df))
        hypo_part = add_below_mol(hypo_part, mol_df, ncbi_search_df)
        parts.append(hypo_part)

    results = pd.concat(parts, ignore_index=True)
    ensure_columns(results, EXPORT_COLS)

    # Cast numeric columns that may be strings from DB cache
    for col in (
        "ncbi_top_identity_percentage",
        "ncbi_top_query_cover",
        "proxy_identity_percentage",
        "proxy_query_cover",
        "gbif_occurrences",
    ):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    # Ensure below_mol is always a clean bool column
    if "below_mol" not in results.columns:
        results["below_mol"] = False
    else:
        results["below_mol"] = (
            results["below_mol"].where(results["below_mol"].notna(), False).astype(bool)
        )

    # Empty sequences — add rows for seq_ids with no species assigned
    all_seq_ids = sequences_df["seq_id"].unique()
    present_seqs = set(results["seq_id"].dropna().unique())
    missing_seqs = [s for s in all_seq_ids if s not in present_seqs]
    if missing_seqs:
        empty_rows = sequences_df[sequences_df["seq_id"].isin(missing_seqs)][
            ["seq_id"] + [c for c in ("dna_sequence",) if c in sequences_df.columns]
        ].drop_duplicates(subset="seq_id")
        results = pd.concat([results, empty_rows], ignore_index=True)

    return results


def _seq_ids_in(df):
    """Extract unique seq_id set from a DataFrame."""
    return set(df["seq_id"].dropna().unique())


def find_sequence_exclusion_step(
    all_seq_ids,
    ncbi_search_df=None,
    mol_df=None,
    tax_df=None,
    geo_df=None,
):
    """Find at which pipeline step each empty sequence lost all species.

    Args:
        all_seq_ids: Full list of sequence IDs.
        ncbi_search_df: Raw NCBI search results (optional).
        mol_df: Filtered MOL results (optional).
        tax_df: WoRMS taxonomy results (optional).
        geo_df: GBIF geographic results (optional).

    Returns:
        DataFrame with columns: seq_id, pipeline_step.
    """
    steps = []
    if ncbi_search_df is not None:
        steps.append(("MOL Search", _seq_ids_in(ncbi_search_df)))
    if mol_df is not None:
        steps.append(("MOL Filter", _seq_ids_in(mol_df)))
    if tax_df is not None:
        steps.append(("TAX", _seq_ids_in(tax_df)))
    if geo_df is not None:
        steps.append(("GEO", _seq_ids_in(geo_df)))

    if not steps:
        return pd.DataFrame(columns=["seq_id", "pipeline_step"])

    return find_exclusion_pipeline_step(all_seq_ids, steps)


# ---------------------------------------------------------------------------
# GBIF export (one row per MOTU/ASV)
# ---------------------------------------------------------------------------

TAXONOMY_HIERARCHY = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specificEpithet",
]
TAXONOMY_RANKS = {
    "specificEpithet": "Species",
    "genus": "Genus",
    "family": "Family",
    "order": "Order",
    "class": "Class",
    "phylum": "Phylum",
    "kingdom": "Kingdom",
}

GBIF_EXPORT_COLS = [
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
    "verbatimIdentification",
    "accession_id",
    "accession_id_ref_db",
    "percent_match",
    "percent_query_cover",
    "confidence_score",
    "identificationRemarks",
]


def _find_common_taxonomy(group: pd.DataFrame) -> dict:
    """Find the lowest common taxonomic rank for a group of species rows.

    Returns a dict with taxonomy columns filled to the common level,
    plus scientificName, taxonRank, and identificationRemarks.
    """
    species_names = group["scientificName"].dropna().unique().tolist()

    if len(species_names) <= 1:
        # Single species or empty — keep as-is
        row = group.iloc[0]
        result = {col: row.get(col) for col in TAXONOMY_HIERARCHY}
        result["scientificName"] = row.get("scientificName")
        result["scientificNameAuthorship"] = row.get("scientificNameAuthorship")
        result["taxonRank"] = row.get("taxonRank")
        result["taxonID"] = row.get("taxonID")
        result["taxonID_db"] = row.get("taxonID_db")
        result["identificationRemarks"] = pd.NA
        return result

    # Walk hierarchy bottom-up to find lowest common level
    result = {}
    resolved_rank = None
    resolved_name = None

    for col in reversed(TAXONOMY_HIERARCHY):
        values = group[col].dropna().unique()
        if len(values) == 1:
            resolved_rank = col
            resolved_name = values[0]
            # Fill taxonomy up to this level
            for c in TAXONOMY_HIERARCHY:
                vals = group[c].dropna().unique()
                result[c] = vals[0] if len(vals) == 1 else pd.NA
            break
        else:
            result[col] = pd.NA

    if resolved_rank is None:
        # No common level found — clear all taxonomy
        for c in TAXONOMY_HIERARCHY:
            result[c] = pd.NA
        result["scientificName"] = pd.NA
        result["taxonRank"] = pd.NA
    else:
        result["scientificName"] = resolved_name
        result["taxonRank"] = TAXONOMY_RANKS.get(resolved_rank, resolved_rank)

    # Higher-level taxonID/authorship left empty for now (needs WoRMS lookup)
    result["scientificNameAuthorship"] = pd.NA
    result["taxonID"] = pd.NA
    result["taxonID_db"] = pd.NA
    result["identificationRemarks"] = ", ".join(sorted(species_names))

    return result


@preserve_sequence_order("seq_id", "results_df")
def filter_excluded_results(
    results_df: pd.DataFrame, excluded: set[str] | None
) -> pd.DataFrame:
    """Drop user-excluded species rows, keeping emptied seq_ids as blank rows.

    Args:
        results_df: Results DataFrame with 'seq_id' and 'scientificName' columns.
        excluded: Set of "{seq_id}||{scientificName}" keys to remove. Falsy = no-op.

    Returns:
        Filtered DataFrame. Seq_ids that lose all species are retained as empty
        rows (seq_id + dna_sequence) so they still appear in the curated output.
    """
    if not excluded:
        return results_df

    mask = results_df.apply(
        lambda r: (
            f"{r['seq_id']}||{r['scientificName']}" not in excluded
            if pd.notna(r.get("scientificName"))
            else True
        ),
        axis=1,
    )
    filtered = results_df[mask].reset_index(drop=True)

    # Re-add empty rows for seq_ids that lost all species
    all_seq_ids = results_df["seq_id"].unique()
    present_seq_ids = set(filtered["seq_id"].dropna().unique())
    missing_seq_ids = [s for s in all_seq_ids if s not in present_seq_ids]
    if missing_seq_ids:
        seq_col = ["dna_sequence"] if "dna_sequence" in results_df.columns else []
        empty_rows = results_df[results_df["seq_id"].isin(missing_seq_ids)][
            ["seq_id"] + seq_col
        ].drop_duplicates(subset="seq_id")
        filtered = pd.concat([filtered, empty_rows], ignore_index=True)

    return filtered


@preserve_sequence_order("seq_id", "results_df")
def build_gbif_export_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build a GBIF export with one row per MOTU/ASV.

    Multi-species MOTUs are resolved to their lowest common taxon,
    with individual species listed in identificationRemarks.
    """
    rows = []
    for seq_id, group in results_df.groupby("seq_id", sort=False):
        species_rows = group.dropna(subset=["scientificName"])

        row = {"seq_id": seq_id}

        # Keep dna_sequence
        if "dna_sequence" in group.columns:
            row["dna_sequence"] = group["dna_sequence"].iloc[0]

        if species_rows.empty:
            # Empty sequence — no species assigned
            rows.append(row)
            continue

        # Resolve common taxonomy
        taxonomy = _find_common_taxonomy(species_rows)
        row.update(taxonomy)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all GBIF export columns exist
    ensure_columns(df, GBIF_EXPORT_COLS)

    # Fill taxonID/authorship for higher-level taxa via WoRMS
    needs_lookup = df["identificationRemarks"].notna() & df["scientificName"].notna()
    if needs_lookup.any():
        _fill_worms_records(df, needs_lookup)

    return df[GBIF_EXPORT_COLS]


def _fill_worms_records(df: pd.DataFrame, mask: pd.Series) -> None:
    """Look up WoRMS records for higher-level taxa and fill taxonID/authorship in-place."""
    lookups = (
        df.loc[mask, ["scientificName", "taxonRank"]].drop_duplicates().values.tolist()
    )
    logger.info(f"Looking up {len(lookups)} higher-level taxa in WoRMS")

    cache = {}
    for name, rank in lookups:
        record = get_aphia_record(name)
        if record and record.get("rank", "").lower() == rank.lower():
            cache[name] = {
                "taxonID": str(record.get("valid_AphiaID", record["AphiaID"])),
                "scientificNameAuthorship": record.get("valid_authority")
                or record.get("authority"),
                "taxonID_db": "WORMS",
            }
        elif record:
            logger.warning(
                f"WoRMS rank mismatch for '{name}': expected {rank}, got {record.get('rank')}"
            )

    # Clean authorships
    if cache:
        auths = clean_authorships(
            pd.Series([v["scientificNameAuthorship"] for v in cache.values()])
        )
        for (name, rec), auth in zip(cache.items(), auths):
            rec["scientificNameAuthorship"] = auth

    for idx in df.index[mask]:
        name = df.at[idx, "scientificName"]
        if name in cache:
            for col, val in cache[name].items():
                df.at[idx, col] = val
