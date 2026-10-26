"""Results assembly — combines MOL+GEO and HYPO species into a unified results table.

Unlike other pipelines, this module mostly aggregates the outputs of the
other steps. Its one search is the GBIF export's WoRMS lookup of higher-level
taxa (run_higher_taxa_lookup), saved per name like the other steps' searches.

    build_results_df(sequences_df, geo_df, mol_df, ...) → DataFrame
    build_gbif_export_df(results_df, ...) → DataFrame (one row per MOTU/ASV)
    run_higher_taxa_lookup(names, ...) → (records_df, params)
    resolve_sequences(results_df) → DataFrame (species count and shared rank per MOTU/ASV)
    add_check_hit(df) → DataFrame
    add_low_identity_warning(df, threshold) → DataFrame
    best_identity_per_sequence(df, query_cover) → DataFrame (identity and via per MOTU/ASV)
    flag_low_identity(df, threshold, best_identity) → DataFrame
    build_species_flow(seq_id, ..., name_table) → list of (source, target, names) links
    species_flow_table(seq_id, links, ...) → DataFrame (one row per name)
    flow_lost_at(links) → (step, names) where a sequence without a species lost them
    plot_species_flow(links) → Plotly Sankey figure
"""

from collections.abc import Callable

import pandas as pd
from loguru import logger

from trident.clients.worms import (
    INFRASPECIFIC_RANKS,
    clean_authorships,
    get_aphia_records,
    pick_record,
    worms_search_url,
)
from trident.core.database import PartialCache, save_to_db
from trident.core.http import create_session
from trident.core.utils import (
    StepLog,
    ensure_columns,
    name_list,
    notify_progress,
    pluralize,
    normalize_name,
    preserve_sequence_order,
)
from trident.pipelines.names import (
    best_hits,
    name_keys,
    ncbi_names,
    species_key,
    species_of,
)


# HYPO only, shown in the app but not exported: the MOL species a HYPO
# species' CO1 proxy matched, and the sequence's MOL identity to it.
VIA_COLS = ["proxy_hit_species", "via_mol_identity"]

# MOL+GEO only, for the GBIF export: the NCBI names of the species' MOL hits,
# joined by ", " as the export writes them.
VERBATIM_COLS = ["verbatimIdentification"]

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
    "mol_identity",
    "mol_query_cover",
    "geo_occurrences",
    "proxy_identity",
    "proxy_query_cover",
    "hypo_check_identity",
    "hypo_check_query_cover",
    "taxonURL",
    "geo_url",
    "mol_hit_url",
    "proxy_url",
    "hypo_check_hit_url",
]


def add_check_hit(df):
    """Flag (hypo_check_hit) the HYPO species whose target marker the HYPO
    check found in NCBI: a hit in hypo_check_identity, empty otherwise.

    A hypothetical species is one whose marker is absent from GenBank, so any
    check hit puts it up for review, whatever its identity or cover.
    """
    df["hypo_check_hit"] = pd.to_numeric(
        df["hypo_check_identity"], errors="coerce"
    ).notna()
    return df


def add_low_identity_warning(df, threshold, identity_col="mol_identity"):
    """Flag rows where identity is present and below threshold."""
    df[identity_col] = pd.to_numeric(df[identity_col], errors="coerce")
    # `lt` returns False (not NA) for NaN, and `&` of two boolean Series is
    # always a clean boolean, so no fillna is needed.
    df["low_identity_warning"] = (
        df[identity_col].lt(threshold) & df[identity_col].notna()
    )
    return df


def best_identity_per_sequence(df, query_cover):
    """Best NCBI identity of each sequence's species, and the MOL species it
    comes through (via) when it is not the species' own hit.

    A MOL+GEO species has its MOL hit, a HYPO species its check hit; only hits
    covering at least the MOL query cover count (a partial hit's identity is
    over part of the sequence only). A HYPO species without such a check hit
    has no hit of its own: it takes the sequence's MOL hit on the species its CO1
    proxy matched, the only NCBI evidence for the sequence.

    Returns:
        DataFrame indexed by seq_id with columns identity and via.
    """
    direct = (
        df["mol_identity"]
        .where(df["mol_query_cover"] >= query_cover)
        .fillna(
            df["hypo_check_identity"].where(df["hypo_check_query_cover"] >= query_cover)
        )
    )
    hypo = df["validation_step"].eq("HYPO")
    via = df["via_mol_identity"].where(hypo & direct.isna())
    rows = pd.DataFrame(
        {
            "seq_id": df["seq_id"],
            "identity": direct.fillna(via),
            "via": df["proxy_hit_species"].where(via.notna()),
        }
    )
    # On a tie, the species' own hit wins over one through another species.
    rows["_direct"] = rows["via"].isna()
    return (
        rows.dropna(subset=["identity"])
        .sort_values(["identity", "_direct"], ascending=False)
        .drop_duplicates("seq_id")
        .set_index("seq_id")[["identity", "via"]]
    )


def flag_low_identity(df, threshold, best_identity):
    """Flag the rows of sequences whose best identity is below threshold."""
    df = df.copy()
    best = df["seq_id"].map(best_identity)
    df["low_identity_warning"] = best.notna() & (best < threshold)
    return df


def _select_columns(df, extra=None):
    """df with the export's columns (and *extra*), empty where df lacks one."""
    return df.reindex(columns=[*EXPORT_COLS, *(extra or [])])


def _build_mol_geo_part(geo_df, mol_df, name_table):
    """Build MOL+GEO rows: species confirmed via NCBI + geographically validated.

    Each row keeps the best MOL hit of its species for its sequence (the
    mol_* columns TAX gave it) and gets the NCBI names of its MOL hits,
    matched on their species (see species_of).
    """
    mol_geo = geo_df[geo_df["in_mol"]].copy()
    mol_geo["validation_step"] = "MOL+GEO"
    mol_geo["verbatimIdentification"] = ncbi_names(
        mol_geo, mol_df, name_table
    ).str.join(", ")
    return mol_geo


def _build_hypo_part(hypo_df, geo_df, mol_df, name_table):
    """Build HYPO rows: hypothetical species supported by a BOLD/NCBI proxy."""
    hypo = hypo_df.copy()
    hypo["validation_step"] = "HYPO"

    # The sequence's own evidence for a HYPO species: how well it matched the
    # MOL species the proxy links it to (its marker is not in GenBank). The
    # proxy hit and the MOL hits may name that species differently.
    via_identity = best_hits(mol_df, name_table, ["identity_percentage"]).rename(
        columns={"identity_percentage": "via_mol_identity"}
    )
    hypo = (
        hypo.assign(species_key=species_key(hypo["proxy_hit_species"], name_table))
        .merge(via_identity, on=["seq_id", "species_key"], how="left")
        .drop(columns="species_key")
    )

    # Fill geo_occurrences and geo_url from geo_df (HYPO doesn't carry them)
    geo_fill_cols = [c for c in ("geo_occurrences", "geo_url") if c in geo_df.columns]
    if geo_fill_cols:
        geo_fill = geo_df[["seq_id", "scientificName"] + geo_fill_cols].drop_duplicates(
            subset=["seq_id", "scientificName"]
        )
        hypo = hypo.merge(geo_fill, on=["seq_id", "scientificName"], how="left")

    return hypo


@preserve_sequence_order("seq_id", "sequences_df")
def build_results_df(sequences_df, geo_df, mol_df, hypo_df=None, *, name_table):
    """Build combined results DataFrame from MOL+GEO and HYPO species.

    Progressive: works with just geo_df + mol_df, adds HYPO rows when available.
    Empty sequences (no species assigned) get a row with just seq_id.

    Args:
        sequences_df: Original sequences DataFrame (used for ordering and full seq_id list).
        geo_df: GEO results (must have 'in_mol' column).
        mol_df: MOL results (NCBI BLAST hits).
        hypo_df: HYPO results (optional, proxy-validated species).
        name_table: See build_name_table, to match MOL names on their species.
    """
    parts = [
        _select_columns(
            _build_mol_geo_part(geo_df, mol_df, name_table), extra=VERBATIM_COLS
        )
    ]

    if hypo_df is not None and not hypo_df.empty:
        hypo_part = _select_columns(
            _build_hypo_part(hypo_df, geo_df, mol_df, name_table), extra=VIA_COLS
        )
        parts.append(hypo_part)

    results = pd.concat(parts, ignore_index=True)
    ensure_columns(results, EXPORT_COLS + VIA_COLS + VERBATIM_COLS)

    # Cast numeric columns that may be strings from DB cache
    for col in (
        "mol_identity",
        "mol_query_cover",
        "proxy_identity",
        "proxy_query_cover",
        "hypo_check_identity",
        "hypo_check_query_cover",
        "geo_occurrences",
        "via_mol_identity",
    ):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    # Empty sequences — add rows for seq_ids with no species assigned
    all_seq_ids = sequences_df["seq_id"].unique()
    present_seqs = set(results["seq_id"].dropna().unique())
    missing_seqs = [s for s in all_seq_ids if s not in present_seqs]
    if missing_seqs:
        empty_rows = sequences_df[sequences_df["seq_id"].isin(missing_seqs)][
            ["seq_id"] + [c for c in ("dna_sequence",) if c in sequences_df.columns]
        ].drop_duplicates(subset="seq_id")
        results = pd.concat([results, empty_rows], ignore_index=True)

    return add_check_hit(results)


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


def _lone_taxon(row) -> dict:
    """The taxon of a sequence with one species: that species' own row."""
    result = {col: row.get(col) for col in TAXONOMY_HIERARCHY}
    for col in (
        "scientificName",
        "scientificNameAuthorship",
        "taxonRank",
        "taxonID",
        "taxonID_db",
    ):
        result[col] = row.get(col)
    result["identificationRemarks"] = pd.NA
    return result


def _find_common_taxonomy(group: pd.DataFrame) -> dict:
    """Find the lowest common taxonomic rank for a group of species rows.

    Returns a dict with taxonomy columns filled to the common level,
    plus scientificName, taxonRank, and identificationRemarks.
    """
    species_names = group["scientificName"].dropna().unique().tolist()

    if len(species_names) <= 1:
        return _lone_taxon(group.iloc[0])

    # Walk the hierarchy top-down: a level is common only when every level
    # above it is too, so homonyms (e.g. the genus Liparis, a fish and a plant)
    # do not make a common taxon. A level missing for every species is skipped.
    result = {col: pd.NA for col in TAXONOMY_HIERARCHY}
    resolved_rank = None
    resolved_name = None
    # Plain lists: pandas calls on these small groups cost more than the work.
    filled = {
        col: [v for v in group[col].tolist() if pd.notna(v)]
        for col in TAXONOMY_HIERARCHY
    }
    genus_only = any(" " not in str(name).strip() for name in species_names)

    for col in TAXONOMY_HIERARCHY:
        values = list(dict.fromkeys(filled[col]))
        # An epithet names a species only within a known genus, and not when a
        # candidate is a genus alone (a one-word name keeps it at genus).
        if len(values) > 1 or (
            col == "specificEpithet" and (resolved_rank != "genus" or genus_only)
        ):
            break
        if len(values) == 1:
            result[col] = values[0]
            resolved_rank = col
            resolved_name = values[0]

    if resolved_rank is None:
        # No common level found — clear all taxonomy
        for c in TAXONOMY_HIERARCHY:
            result[c] = pd.NA
        result["scientificName"] = pd.NA
        result["taxonRank"] = pd.NA
    else:
        # A species is named by its binomial, not by its epithet alone.
        if resolved_rank == "specificEpithet" and pd.notna(result["genus"]):
            resolved_name = f"{result['genus']} {resolved_name}"
        result["scientificName"] = resolved_name
        result["taxonRank"] = TAXONOMY_RANKS.get(resolved_rank, resolved_rank)

    # Higher-level taxonID/authorship left empty for now (needs WoRMS lookup)
    result["scientificNameAuthorship"] = pd.NA
    result["taxonID"] = pd.NA
    result["taxonID_db"] = pd.NA
    result["identificationRemarks"] = ", ".join(sorted(species_names))

    return result


# WoRMS ranks at species level or below, lower case.
SPECIES_RANKS = {"species"} | {rank.lower() for rank in INFRASPECIFIC_RANKS}


def resolve_sequences(results_df: pd.DataFrame) -> pd.DataFrame:
    """Give each sequence its number of species and the taxon they resolve to.

    The taxon and its rank are the ones the GBIF export writes for the sequence
    (same routines): the species itself, or the lowest taxon its species share,
    empty when none is shared or no species is assigned. species_level is True
    when that taxon is a species or below (a species and its subspecies resolve
    to the species), and species then names that species by genus and epithet,
    so a subspecies counts with its species.

    Returns:
        DataFrame with columns: seq_id, n_species, taxonRank, scientificName,
        species_level, species.
    """
    species_rows = results_df.dropna(subset=["scientificName"])
    n_species = species_rows.groupby("seq_id")["scientificName"].nunique()

    # A lone species is its own taxon, taken from its first row as the GBIF
    # export does; only sequences with several species need the common taxon.
    single = n_species.index[n_species == 1]
    first_rows = species_rows[species_rows["seq_id"].isin(single)].drop_duplicates(
        "seq_id"
    )
    common = {row["seq_id"]: _lone_taxon(row) for row in first_rows.to_dict("records")}
    several = species_rows[species_rows["seq_id"].isin(n_species.index[n_species > 1])]
    for seq_id, group in several.groupby("seq_id", sort=False):
        common[seq_id] = _find_common_taxonomy(group)

    rows = []
    for seq_id in results_df["seq_id"].dropna().drop_duplicates():
        n = int(n_species.get(seq_id, 0))
        row = {"seq_id": seq_id, "n_species": n, "species_level": False}
        if n:
            taxon = common[seq_id]
            row["taxonRank"] = taxon["taxonRank"]
            row["scientificName"] = taxon["scientificName"]
            rank = str(taxon["taxonRank"]).lower()
            name = str(taxon["scientificName"])
            # A lone binomial whose rank WoRMS left empty is still a species.
            row["species_level"] = rank in SPECIES_RANKS or (
                n == 1 and pd.isna(taxon["taxonRank"]) and " " in name
            )
            if row["species_level"]:
                genus, epithet = taxon["genus"], taxon["specificEpithet"]
                row["species"] = (
                    f"{genus} {epithet}"
                    if pd.notna(genus) and pd.notna(epithet)
                    else name
                )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "seq_id",
            "n_species",
            "taxonRank",
            "scientificName",
            "species_level",
            "species",
        ],
    )


def exclusion_keys(df: pd.DataFrame) -> pd.Series:
    """The "{seq_id}||{scientificName}" key of each row, as exclusions store it."""
    return df["seq_id"] + "||" + df["scientificName"]


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

    mask = results_df["scientificName"].isna() | ~exclusion_keys(results_df).isin(
        excluded
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
def build_gbif_export_df(results_df: pd.DataFrame, db_path=None) -> pd.DataFrame:
    """Build a GBIF export with one row per MOTU/ASV.

    Multi-species MOTUs are resolved to their lowest common taxon,
    with individual species listed in identificationRemarks. The taxonID and
    authorship of that taxon come from WoRMS (run_higher_taxa_lookup, saved in
    *db_path*). When the WoRMS request for a name fails, the rows of that name
    keep both empty and the name is listed in ``attrs["failed_items"]``.
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
        # The NCBI names its MOL+GEO species were identified under.
        if "verbatimIdentification" in species_rows.columns:
            names = sorted(
                {
                    name
                    for joined in species_rows["verbatimIdentification"].dropna()
                    for name in joined.split(", ")
                }
            )
            row["verbatimIdentification"] = ", ".join(names) if names else pd.NA
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all GBIF export columns exist
    ensure_columns(df, GBIF_EXPORT_COLS)

    # Fill taxonID/authorship for higher-level taxa via WoRMS
    needs_lookup = df["identificationRemarks"].notna() & df["scientificName"].notna()
    failed = []
    if needs_lookup.any():
        names = sorted(df.loc[needs_lookup, "scientificName"].unique())
        records_df, _ = run_higher_taxa_lookup(names, db_path=db_path)
        failed = records_df.attrs.get("failed_items", [])
        _fill_worms_records(df, needs_lookup, records_df)

    df = df[GBIF_EXPORT_COLS]
    df.attrs["failed_items"] = failed
    return df


# The fields of a WoRMS record that pick_record and the export read.
HIGHER_TAXA_COLS = [
    "scientificName",
    "AphiaID",
    "valid_AphiaID",
    "rank",
    "status",
    "kingdom",
    "class",
    "authority",
    "valid_authority",
]


@save_to_db(
    table_name="worms_higher_taxa",
    cache=PartialCache(items_kwarg="names", item_key="scientificName"),
    log_as=("WoRMS higher taxa", "taxon"),
)
def run_higher_taxa_lookup(
    names: list[str],
    user_agent: str | None = None,
    progress_handler: dict | object | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Every WoRMS record of each higher-level taxon name of the GBIF export.

    All the records of a name are kept, not one: the export chooses among them
    by the rank and group of each sequence's taxon (see _fill_worms_records).
    Names whose request failed go to failure_sink, so they are not saved and
    are looked up again on the next export.
    result_sink, given by save_to_db, saves each name as soon as its records
    are fetched.

    Returns:
        DataFrame with HIGHER_TAXA_COLS, one row per record (none for a name
        WoRMS does not know).
    """
    session = create_session(user_agent=user_agent)
    rows: list[dict] = []
    failed: list[str] = []
    log = StepLog("WoRMS higher taxa", len(names))
    try:
        for name in names:
            try:
                with log.query(name):
                    records = get_aphia_records(name, session=session)
            except Exception as exc:
                log.failed(name, f"{exc} (check: {worms_search_url(name)})")
                failed.append(name)
                continue
            finally:
                notify_progress(progress_handler)
            n = len(records)
            log.item(name, f"{n} {pluralize(n, 'record')}", notable=not records)
            name_rows = [
                {
                    **{col: r.get(col) for col in HIGHER_TAXA_COLS},
                    "scientificName": name,
                }
                for r in records
            ]
            rows.extend(name_rows)
            if result_sink is not None:
                result_sink(pd.DataFrame(name_rows, columns=HIGHER_TAXA_COLS), [name])
    finally:
        session.close()

    if failed and failure_sink is not None:
        failure_sink.extend(failed)
    found = len({row["scientificName"] for row in rows})
    log.done(
        f"{found}/{len(names)} {pluralize(len(names), 'taxon')} found",
        failed,
        "taxon",
        retry="Build the one-row-per-sequence table again to retry them.",
    )
    return pd.DataFrame(rows, columns=HIGHER_TAXA_COLS)


def _fill_worms_records(
    df: pd.DataFrame, mask: pd.Series, records_df: pd.DataFrame
) -> None:
    """Fill taxonID/authorship of the higher-level taxa in-place, from their
    WoRMS records (run_higher_taxa_lookup).

    A name can be a taxon of that rank in several groups (e.g. the genus
    Liparis, a fish and a plant): the record in the row's class, else its
    kingdom, is taken (see pick_record).
    """
    keys = ["scientificName", "taxonRank", "class", "kingdom"]

    def key_of(values) -> tuple:
        # None for missing values, which (unlike NaN) compare equal as keys.
        return tuple(None if pd.isna(v) else v for v in values)

    lookups = sorted(
        {key_of(row) for row in df.loc[mask, keys].values.tolist()}, key=str
    )
    # None for missing fields, as in a WoRMS answer, so "or" falls back on them.
    fields = records_df.astype(object).where(records_df.notna(), None)
    records_of = {
        name: group.to_dict("records")
        for name, group in fields.groupby("scientificName")
    }
    cache = {}
    for name, rank, class_, kingdom in lookups:
        records = records_of.get(name, [])
        record = pick_record(records, rank, {"class": class_, "kingdom": kingdom})
        if record:
            cache[(name, rank, class_, kingdom)] = {
                # An ID read back from the analysis can be a float, or its
                # text ("126160.0") when the column was first saved empty.
                "taxonID": str(
                    int(float(record.get("valid_AphiaID") or record["AphiaID"]))
                ),
                "scientificNameAuthorship": record.get("valid_authority")
                or record.get("authority"),
                "taxonID_db": "WORMS",
            }
        elif records:
            found = sorted({r.get("rank") or "?" for r in records})
            logger.warning(f"No WoRMS {rank} record for '{name}' (found: {found})")

    # Clean authorships
    if cache:
        auths = clean_authorships(
            pd.Series([v["scientificNameAuthorship"] for v in cache.values()])
        )
        for (name, rec), auth in zip(cache.items(), auths):
            rec["scientificNameAuthorship"] = auth

    for idx in df.index[mask]:
        key = key_of(df.loc[idx, keys])
        if key in cache:
            for col, val in cache[key].items():
                df.at[idx, col] = val


# ---------------------------------------------------------------------------
# Species flow (one MOTU/ASV through the steps)
# ---------------------------------------------------------------------------


# Sankey of one sequence's candidates: each node's column (left to right),
# label and RGB colour (losses grey, the MOL stream blue, TAX additions orange).
FLOW_NODES = {
    "ncbi": (0, "NCBI hits", "76, 120, 168"),
    "filter_out": (1, "Removed by the MOL filter", "186, 176, 172"),
    "mol": (1, "MOL list", "76, 120, 168"),
    "tax_out": (2, "Not kept at TAX", "186, 176, 172"),
    "tax_mol": (2, "MOL species (TAX)", "76, 120, 168"),
    "tax_added": (2, "Added by TAX (congeners)", "245, 133, 24"),
    "geo_out": (3, "Not in the GEO area", "186, 176, 172"),
    "geo_not_run": (3, "GEO not run", "186, 176, 172"),
    "mol_geo": (3, "MOL+GEO", "76, 120, 168"),
    "added_geo": (3, "Added, in the GEO area", "245, 133, 24"),
    "hypo": (4, "Confirmed by HYPO", "245, 133, 24"),
    "no_proxy": (4, "No BOLD proxy (EXTRA)", "186, 176, 172"),
    "hypo_out": (4, "Not confirmed by HYPO", "186, 176, 172"),
    "extra_not_run": (4, "EXTRA not run", "186, 176, 172"),
    "hypo_not_run": (4, "HYPO not run", "186, 176, 172"),
    "excluded": (5, "Excluded by hand", "186, 176, 172"),
    "kept": (5, "Kept", "84, 162, 75"),
}
FLOW_HOVER_NAMES = 25
# Nodes of the TAX additions' stream; every other node is on the MOL stream.
FLOW_ADDED = {"tax_added", "added_geo", "hypo"}


def _names(df, seq_id, mask=None):
    """Names of a sequence's rows in a step table, optionally masked."""
    if df is None:
        return set()
    rows = df[df["seq_id"] == seq_id]
    if mask is not None:
        rows = rows[mask(rows)]
    return set(rows["scientificName"].dropna())


def _in_mol(rows):
    """Rows of a species found by MOL itself."""
    return rows["in_mol"]


def _added(rows):
    """Rows of a species TAX added (a congener of a MOL species)."""
    return ~_in_mol(rows)


class _SpeciesNames:
    """species_key and species_of of names, each computed once per name: one
    lookup serves all the flows of build_species_flows (both map each name on
    its own, so computing them for more names at once gives the same)."""

    def __init__(self, name_table):
        self.name_table = name_table
        self._key = {}
        self._species = {}

    def _fill(self, known, fn, names):
        missing = sorted({name for name in names if name not in known})
        if missing:
            series = pd.Series(missing, dtype=object)
            known.update(zip(missing, fn(series, self.name_table)))

    def keys(self, names):
        """{name: species key} of a set of names."""
        self._fill(self._key, species_key, names)
        return {name: self._key[name] for name in names}

    def species(self, names):
        """The species of a set of names (see species_of)."""
        self._fill(self._species, species_of, names)
        return {self._species[name] for name in names}

    def without(self, names, keys):
        """The names whose species is not in keys."""
        return {name for name, key in self.keys(names).items() if key not in keys}


def build_species_flow(
    seq_id,
    ncbi_search_df,
    mol_df,
    tax_df=None,
    geo_df=None,
    extra_df=None,
    hypo_df=None,
    excluded=(),
    *,
    name_table,
    species_names=None,
):
    """One sequence's candidates through the steps, for the steps run so far.

    NCBI hits and the MOL list carry NCBI names; from TAX on, names are WoRMS
    accepted names, so a MOL name merged as a synonym counts once at TAX.
    NCBI names are compared with the others through their species (see
    species_of).

    Args:
        excluded: "{seq_id}||{scientificName}" keys of the candidates excluded
            by hand.
        name_table: See build_name_table.
        species_names: The lookup of the names' species, shared by
            build_species_flows; one for this sequence otherwise.

    Returns:
        List of (source, target, names) links between FLOW_NODES, each with
        at least one name.
    """
    lookup = species_names or _SpeciesNames(name_table)
    hits = _names(ncbi_search_df, seq_id)
    mol = _names(mol_df, seq_id)
    key_of = lookup.keys(hits | mol)
    hit_keys = {key_of[name] for name in hits}
    mol_keys = {key_of[name] for name in mol}
    # A hit of a species that MOL kept, under any of its names, is not a loss.
    links = [
        ("ncbi", "filter_out", {name for name in hits if key_of[name] not in mol_keys}),
        ("ncbi", "mol", mol),
    ]
    if tax_df is None:
        return _nonempty(links)

    tax_mol = _names(tax_df, seq_id, _in_mol)
    # A congener BLAST returned and the MOL filter rejected, which TAX keeps
    # on request, stays a filter loss: EXTRA seeks no proxy for it.
    tax_added = lookup.without(_names(tax_df, seq_id, _added), hit_keys)
    # MOL species (by their WoRMS name) that TAX does not list.
    mol_species = lookup.species(mol)
    tax_mol_keys = set(lookup.keys(tax_mol).values())
    links += [
        ("mol", "tax_out", lookup.without(mol_species, tax_mol_keys)),
        ("mol", "tax_mol", tax_mol),
    ]
    if geo_df is None:
        links += [
            ("tax_mol", "geo_not_run", tax_mol),
            ("tax_added", "geo_not_run", tax_added),
        ]
        return _nonempty(links)

    mol_geo = _names(geo_df, seq_id, _in_mol)
    added_geo = lookup.without(_names(geo_df, seq_id, _added), hit_keys)
    links += [
        ("tax_mol", "geo_out", tax_mol - mol_geo),
        ("tax_mol", "mol_geo", mol_geo),
        ("tax_added", "geo_out", tax_added - added_geo),
        ("tax_added", "added_geo", added_geo),
    ]

    names = sorted(mol_geo | added_geo)
    candidates = pd.DataFrame(
        {"seq_id": [seq_id] * len(names), "scientificName": names}, dtype=str
    )
    excluded = set(
        candidates.loc[exclusion_keys(candidates).isin(excluded), "scientificName"]
    )
    links += [
        ("mol_geo", "kept", mol_geo - excluded),
        ("mol_geo", "excluded", mol_geo & excluded),
    ]
    if extra_df is None:
        links.append(("added_geo", "extra_not_run", added_geo))
        return _nonempty(links)
    # HYPO tests only the added species EXTRA found a BOLD proxy for.
    proxied = _names(extra_df, seq_id)
    links.append(("added_geo", "no_proxy", added_geo - proxied))
    if hypo_df is None:
        links.append(("added_geo", "hypo_not_run", added_geo & proxied))
        return _nonempty(links)
    hypo = _names(hypo_df, seq_id) & added_geo & proxied
    links += [
        ("added_geo", "hypo_out", (added_geo & proxied) - hypo),
        ("added_geo", "hypo", hypo),
        ("hypo", "kept", hypo - excluded),
        ("hypo", "excluded", hypo & excluded),
    ]
    return _nonempty(links)


def build_species_flows(
    seq_ids,
    ncbi_search_df,
    mol_df,
    tax_df=None,
    geo_df=None,
    extra_df=None,
    hypo_df=None,
    excluded=(),
    *,
    name_table,
):
    """build_species_flow of each of *seq_ids*, as {seq_id: links}. The rows
    of these sequences are split by sequence once, instead of each step table
    being filtered again for every sequence (seconds on thousands of
    sequences)."""
    tables = {
        key: None if df is None else df[df["seq_id"].isin(seq_ids)]
        for key, df in {
            "ncbi_search_df": ncbi_search_df,
            "mol_df": mol_df,
            "tax_df": tax_df,
            "geo_df": geo_df,
            "extra_df": extra_df,
            "hypo_df": hypo_df,
        }.items()
    }
    by_seq = {
        key: (dict(tuple(df.groupby("seq_id", sort=False))), df.iloc[:0])
        for key, df in tables.items()
        if df is not None
    }

    # The names' species keys, looked up once for all the flows, and the
    # species of the MOL names (the only ones a flow asks the species of).
    lookup = _SpeciesNames(name_table)
    names = set().union(
        *(df["scientificName"].dropna() for df in tables.values() if df is not None)
    )
    mol_df = tables["mol_df"]
    mol = set() if mol_df is None else set(mol_df["scientificName"].dropna())
    lookup.keys(names | lookup.species(mol))

    def rows_of(key, seq_id):
        if key not in by_seq:
            return None
        groups, empty = by_seq[key]
        return groups.get(seq_id, empty)

    return {
        seq_id: build_species_flow(
            seq_id,
            **{key: rows_of(key, seq_id) for key in tables},
            excluded=excluded,
            name_table=name_table,
            species_names=lookup,
        )
        for seq_id in seq_ids
    }


def _nonempty(links):
    """The links that carry at least one name."""
    return [link for link in links if link[2]]


# The step each loss node of the flow stands for, as Results names it.
FLOW_LOSS_STEPS = {
    "filter_out": "MOL Filter",
    "tax_out": "TAX",
    "geo_out": "GEO",
    "geo_not_run": "GEO not run",
    "no_proxy": "EXTRA",
    "extra_not_run": "EXTRA not run",
    "hypo_out": "HYPO",
    "hypo_not_run": "HYPO not run",
    "excluded": "Curation",
}
# Loss nodes whose candidates still wait for a step.
FLOW_NOT_RUN = {"geo_not_run", "extra_not_run", "hypo_not_run"}


def flow_lost_at(links):
    """Where a sequence without a species lost its last ones, with the names
    that ended there. Curation when species it had were excluded by hand (the
    user's decision empties it, whatever its other candidates became); else a
    step not run yet when a candidate still waits for it; else the flow's
    furthest loss (FLOW_NODES runs left to right): a sequence with some species
    HYPO did not confirm and others without a BOLD proxy lost its last chance
    at HYPO.

    Returns:
        (step, names); ("MOL Search", set()) for a sequence without NCBI hits,
        (None, set()) when its flow loses no candidate.
    """
    if not links:
        return "MOL Search", set()
    rank = {node: i for i, node in enumerate(FLOW_NODES)}
    losses = [(dst, names) for _, dst, names in links if dst in FLOW_LOSS_STEPS]
    if not losses:
        return None, set()
    last, _ = max(
        losses,
        key=lambda loss: (
            loss[0] == "excluded",
            loss[0] in FLOW_NOT_RUN,
            rank[loss[0]],
        ),
    )
    return FLOW_LOSS_STEPS[last], set().union(
        *(names for dst, names in losses if dst == last)
    )


def _hover_names(names):
    """Names for a hover box, cut after FLOW_HOVER_NAMES."""
    return name_list(sorted(names), FLOW_HOVER_NAMES, sep="<br>", more="<br>… and")


def _is_end(node, links):
    """Whether no flow leaves node: where its species ended."""
    return not any(src == node for src, _, _ in links)


def species_flow_table(seq_id, links, ncbi_search_df, name_table):
    """One row per name of build_species_flow's links: where it came from and
    where it ended, with its best NCBI identity for a name the sequence's hits
    carry: an NCBI name's own hits, or a species' hits under any of its names."""
    by_name, by_species = {}, {}
    if ncbi_search_df is not None:
        rows = ncbi_search_df[ncbi_search_df["seq_id"] == seq_id]
        # Cached tables can return numbers as strings.
        identity = pd.to_numeric(rows["identity_percentage"], errors="coerce")
        names = rows["scientificName"]
        by_name = identity.groupby(name_keys(names)).max().to_dict()
        by_species = identity.groupby(species_key(names, name_table)).max().to_dict()

    def best_identity(src, name):
        if src in FLOW_ADDED:
            return None
        best = (by_name if src == "ncbi" else by_species).get(normalize_name(name))
        return None if pd.isna(best) else best

    order = list(FLOW_NODES)
    rows = [
        {
            "species": name,
            "origin": "Added by TAX" if src in FLOW_ADDED else "NCBI hit",
            "outcome": FLOW_NODES[dst][1],
            "identity": best_identity(src, name),
            "_order": order.index(dst),
        }
        for src, dst, names in links
        if _is_end(dst, links)
        for name in names
    ]
    return (
        pd.DataFrame(
            rows, columns=["species", "origin", "outcome", "identity", "_order"]
        )
        .sort_values(["_order", "species"], ascending=[False, True])
        .drop(columns="_order")
    )


def plot_species_flow(links):
    """Sankey of build_species_flow's links, nodes placed by column."""
    import plotly.graph_objects as go

    used = list(dict.fromkeys(n for src, dst, _ in links for n in (src, dst)))
    index = {node: i for i, node in enumerate(used)}

    # A node's names: what flows into it, or out of it for a first node. The
    # MOL list's names come in as NCBI names and leave as WoRMS names.
    targets = {dst for _, dst, _ in links}
    node_names = {node: set() for node in used}
    for src, dst, names in links:
        node_names[dst] |= names
        if src not in targets:
            node_names[src] |= names

    columns = sorted({FLOW_NODES[n][0] for n in used})
    x = {c: i / max(len(columns) - 1, 1) for i, c in enumerate(columns)}
    by_column = {c: [n for n in used if FLOW_NODES[n][0] == c] for c in columns}
    y = {
        n: (i + 0.5) / len(nodes)
        for nodes in by_column.values()
        for i, n in enumerate(nodes)
    }

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            textfont=dict(shadow="none"),
            node=dict(
                label=[f"{FLOW_NODES[n][1]} ({len(node_names[n])})" for n in used],
                color=[f"rgb({FLOW_NODES[n][2]})" for n in used],
                x=[min(max(x[FLOW_NODES[n][0]], 0.001), 0.999) for n in used],
                y=[min(max(y[n], 0.001), 0.999) for n in used],
                customdata=[_hover_names(node_names[n]) for n in used],
                hovertemplate="%{customdata}<extra></extra>",
                pad=20,
            ),
            link=dict(
                source=[index[src] for src, _, _ in links],
                target=[index[dst] for _, dst, _ in links],
                value=[len(names) for _, _, names in links],
                color=[f"rgba({FLOW_NODES[src][2]}, 0.35)" for src, _, _ in links],
                customdata=[_hover_names(names) for _, _, names in links],
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    return fig
