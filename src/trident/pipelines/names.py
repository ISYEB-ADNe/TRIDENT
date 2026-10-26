"""Species name table: the names each species goes by, across sources.

One row per (species, name, source). species is the WoRMS accepted name, or
the name itself when WoRMS does not know it. TAX builds the table from the
WoRMS check of the MOL (NCBI) names, and GEO adds the name GBIF counts each
species under, and EXTRA the name BOLD files its sequences under. Every step
that compares names from two sources goes through
species_of, so a synonym counts as its species; names match ignoring case and
spacing. A name_table of None (names not checked in WoRMS) matches names as
they are.

    build_name_table(resolution_df, ncbi_taxids) → DataFrame
    gbif_name_rows(gbif_search_df) → DataFrame
    bold_name_rows(bold_search_df) → DataFrame
    replace_source_rows(name_table, rows, source) → DataFrame
    resolved_names(name_table) → DataFrame
    name_in_source(species, name_table, source) → Series
    other_names(species, name_table, sources) → dict
    species_of(names, name_table) → Series
    species_key(names, name_table) → Series
    best_hits(hits, name_table, columns) → DataFrame
    found_by_blast(df, ncbi_search_df, name_table) → Series
    ncbi_names(df, mol_df, name_table) → Series
    same_species_groups(species, name_table) → DataFrame
"""

import itertools

import pandas as pd

from trident.core.utils import (
    as_ids,
    ensure_columns,
    normalize_name,
    top_hit_per_group,
)


NAME_TABLE_COLUMNS = [
    "species",
    "species_id",
    "name",
    "source",
    "source_id",
    "matched_by",
    "note",
    "name_key",
    "species_key",
]

# Sources whose names stand for their species in the comparisons (species_of):
# names WoRMS resolved to its accepted species. Rows of other sources record
# what that source calls a species, which can be another WoRMS species (GBIF
# calls Larus saundersi Chroicocephalus saundersi): they never merge two WoRMS
# species.
SYNONYM_SOURCES = ["ncbi"]


def build_name_table(
    resolution_df: pd.DataFrame | None,
    ncbi_taxids: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Name table of the MOL names, from run_name_resolution.

    Args:
        resolution_df: Output of run_name_resolution, or None when names were
            not checked in WoRMS (the table is then empty).
        ncbi_taxids: NCBI taxid of each name (prepare_resolution_taxids), kept
            as the row's source_id.

    Returns:
        DataFrame with NAME_TABLE_COLUMNS, source "ncbi", name_key and
        species_key the name and species as join keys (normalize_name). A name
        WoRMS did not find is its own species, without species_id or matched_by.
    """
    if resolution_df is None or resolution_df.empty:
        return pd.DataFrame(columns=NAME_TABLE_COLUMNS)
    # Rows cached before matchedBy or note existed lack those columns.
    resolved = resolution_df.reindex(
        columns=[
            "scientificName",
            "acceptedName",
            "acceptedNameUsageID",
            "matchedBy",
            "note",
        ]
    )
    names = resolved["scientificName"]
    species = resolved["acceptedName"].fillna(names)
    # Fixed dtypes: rows checked now and rows read back from the cache must
    # give the same table, whose fingerprint is part of later cache keys.
    return pd.DataFrame(
        {
            "species": species.astype(str),
            "species_id": as_ids(resolved["acceptedNameUsageID"]),
            "name": names.astype(str),
            "source": "ncbi",
            "source_id": as_ids(names.map(ncbi_taxids or {})),
            "matched_by": resolved["matchedBy"].astype("string"),
            "note": resolved["note"].astype("string"),
            "name_key": names.map(normalize_name),
            "species_key": species.map(normalize_name),
        }
    )[NAME_TABLE_COLUMNS]


def gbif_name_rows(gbif_search_df: pd.DataFrame) -> pd.DataFrame:
    """Name table rows of the GBIF search: GBIF's name of each species.

    Args:
        gbif_search_df: Output of run_gbif_search (scientificName the TAX
            species, gbif_scientificName, taxonID the GBIF key,
            gbif_speciesID, gbif_matched_by, aphia_id, gbif_note).

    Returns:
        DataFrame with NAME_TABLE_COLUMNS, source "gbif", one row per species
        GBIF matched (a subspecies name when GBIF holds it as one), source_id
        GBIF's species for it (the counted key in a search saved before it
        was kept), note saying how a synonym or subspecies in GBIF is
        counted.
    """
    matched = (
        ensure_columns(
            gbif_search_df.copy(),
            [
                "gbif_scientificName",
                "gbif_speciesID",
                "gbif_matched_by",
                "aphia_id",
                "gbif_note",
            ],
        )
        .dropna(subset=["scientificName", "gbif_scientificName"])
        .drop_duplicates("scientificName")
    )
    return _source_rows(
        matched,
        "gbif",
        name_col="gbif_scientificName",
        matched_by_col="gbif_matched_by",
        species_ids=as_ids(matched["aphia_id"]),
        notes=matched["gbif_note"],
        source_ids=matched["gbif_speciesID"]
        .astype("string")
        .fillna(matched["taxonID"].astype("string")),
    )


def bold_name_rows(bold_search_df: pd.DataFrame) -> pd.DataFrame:
    """Name table rows of the BOLD search: the name BOLD files each species'
    sequences under.

    Args:
        bold_search_df: Output of run_bold_search (scientificName the queried
            species, bold_scientificName, taxonID the BOLD taxid,
            bold_matched_by, bold_outcome).

    Returns:
        DataFrame with NAME_TABLE_COLUMNS, source "bold", one row per species
        and BOLD name with sequences (a species BOLD splits has several);
        empty for a search saved before BOLD names were kept.
    """
    found = (
        ensure_columns(
            bold_search_df.copy(),
            ["dna_sequence", "bold_scientificName", "bold_matched_by", "taxonID"],
        )
        .dropna(subset=["dna_sequence", "scientificName", "bold_scientificName"])
        .drop_duplicates(["scientificName", "bold_scientificName"])
    )
    return _source_rows(
        found, "bold", name_col="bold_scientificName", matched_by_col="bold_matched_by"
    )


def _source_rows(
    df: pd.DataFrame,
    source: str,
    name_col: str,
    matched_by_col: str,
    species_ids: pd.Series | None = None,
    notes: pd.Series | None = None,
    source_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Name table rows of a search: the species from scientificName, the name
    the source gives it from *name_col*, source_id from *source_ids* or
    taxonID."""
    species = df["scientificName"].astype(str)
    names = df[name_col].astype(str)
    if species_ids is None:
        species_ids = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if notes is None:
        notes = pd.Series(pd.NA, index=df.index)
    return pd.DataFrame(
        {
            "species": species,
            "species_id": species_ids,
            "name": names,
            "source": source,
            "source_id": (df["taxonID"] if source_ids is None else source_ids).astype(
                "string"
            ),
            "matched_by": df[matched_by_col].astype("string"),
            "note": notes.astype("string"),
            "name_key": names.map(normalize_name),
            "species_key": species.map(normalize_name),
        }
    )[NAME_TABLE_COLUMNS].reset_index(drop=True)


def replace_source_rows(
    name_table: pd.DataFrame | None, rows: pd.DataFrame, source: str
) -> pd.DataFrame:
    """The name table with the rows of *source* replaced by *rows*: a step run
    again gives the names of its new results."""
    if name_table is None:
        name_table = pd.DataFrame(columns=NAME_TABLE_COLUMNS)
    kept = name_table[name_table["source"] != source]
    if rows.empty:
        return kept.reset_index(drop=True)
    if kept.empty:
        return rows.reset_index(drop=True)
    return pd.concat([kept, rows], ignore_index=True)


def name_keys(names: pd.Series) -> pd.Series:
    """normalize_name of each name, computed once per distinct name: hit
    tables repeat their names."""
    distinct = pd.Series(pd.unique(names.dropna()), dtype=object)
    keys = dict(zip(distinct, distinct.map(normalize_name)))
    # Object even when no name is given, so it joins with other name keys.
    return names.map(keys).astype(object)


def resolved_names(name_table: pd.DataFrame | None) -> pd.DataFrame:
    """The rows of SYNONYM_SOURCES: the names WoRMS resolved, which stand for
    their species. Empty (with its columns) without a name table."""
    if name_table is None:
        return pd.DataFrame(columns=NAME_TABLE_COLUMNS)
    return name_table[name_table["source"].isin(SYNONYM_SOURCES)]


def name_in_source(
    species: pd.Series, name_table: pd.DataFrame | None, source: str
) -> pd.Series:
    """The name *source* gives each species, where it differs from the
    species name; empty (NaN) elsewhere. For display only: these names stand
    for their species nowhere else (see SYNONYM_SOURCES)."""
    if name_table is None or name_table.empty:
        return pd.Series(pd.NA, index=species.index, dtype=object)
    rows = name_table[
        (name_table["source"] == source)
        & (name_table["name_key"] != name_table["species_key"])
    ].drop_duplicates("species_key")
    return name_keys(species).map(rows.set_index("species_key")["name"])


def other_names(
    species: list[str],
    name_table: pd.DataFrame | None,
    sources: list[str],
) -> dict[str, list[tuple[str, str]]]:
    """The names *sources* give each species besides its own, to search
    another source under when the species name finds nothing.

    A name that stands for another species is left out: another species of
    the table, a name WoRMS resolved to another species, or a name the
    sources give to several species. A source can lump
    two WoRMS species under one name (GBIF calls Larus saundersi
    Chroicocephalus saundersi), and searching it would give one species the
    other's data.

    Returns:
        {species: [(name, source), ...]}, sources in the order given, each
        name once; species without another name are left out.
    """
    if name_table is None or name_table.empty or not species:
        return {}
    queried = pd.Series(species, dtype=object)
    queried_keys = dict(zip(name_keys(queried), queried))
    species_keys = set(name_table["species_key"]) | set(queried_keys)
    rows = name_table[
        name_table["source"].isin(sources)
        & name_table["species_key"].isin(queried_keys)
        & (name_table["name_key"] != name_table["species_key"])
        & ~name_table["name_key"].isin(species_keys)
    ]
    resolved_to = rows["name_key"].map(_by_name(name_table)["species_key"])
    rows = rows[resolved_to.isna() | (resolved_to == rows["species_key"])]
    named = name_table[name_table["source"].isin(sources)]
    n_species = named.groupby("name_key")["species_key"].nunique()
    rows = rows[rows["name_key"].map(n_species) == 1]
    order = {source: i for i, source in enumerate(sources)}
    rows = rows.sort_values("source", key=lambda s: s.map(order), kind="stable")
    rows = rows.drop_duplicates(["species_key", "name_key"])
    names = {}
    for key, name, source in zip(rows["species_key"], rows["name"], rows["source"]):
        names.setdefault(queried_keys[key], []).append((name, source))
    return names


def _by_name(name_table: pd.DataFrame | None) -> pd.DataFrame:
    """resolved_names indexed by name_key, one row per name (the first)."""
    return resolved_names(name_table).drop_duplicates("name_key").set_index("name_key")


def species_of(names: pd.Series, name_table: pd.DataFrame | None) -> pd.Series:
    """The species each name stands for: its species in the name table, the
    name itself when the table does not list it."""
    by_name = _by_name(name_table)
    if by_name.empty:
        return names.copy()
    return name_keys(names).map(by_name["species"]).fillna(names)


def species_key(names: pd.Series, name_table: pd.DataFrame | None) -> pd.Series:
    """species_of as a join key (normalize_name)."""
    keys = name_keys(names)
    by_name = _by_name(name_table)
    if by_name.empty:
        return keys
    return keys.map(by_name["species_key"]).fillna(keys).astype(object)


def best_hits(
    hits: pd.DataFrame,
    name_table: pd.DataFrame | None,
    columns: list[str] = (),
) -> pd.DataFrame:
    """Best hit (identity, then query cover) of each species for each
    sequence, under any of its names. Hits without a name are left out.

    Returns:
        DataFrame with columns seq_id, species_key and *columns* of the hit,
        scores as numbers (cached tables can return them as strings).
    """
    hits = hits.dropna(subset=["scientificName"])
    return top_hit_per_group(
        hits.assign(
            species_key=species_key(hits["scientificName"], name_table),
            identity_percentage=pd.to_numeric(
                hits["identity_percentage"], errors="coerce"
            ),
            query_cover=pd.to_numeric(hits["query_cover"], errors="coerce"),
        ),
        keys=["seq_id", "species_key"],
        sort=["identity_percentage", "query_cover"],
        columns=["seq_id", "species_key", *columns],
    )


def _rows_of(hits: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """The hits of the sequences of df."""
    return hits[hits["seq_id"].isin(df["seq_id"].unique())]


def _pairs(df: pd.DataFrame, name_table: pd.DataFrame | None) -> pd.MultiIndex:
    """(seq_id, species key) of each row of df."""
    return pd.MultiIndex.from_arrays(
        [df["seq_id"], species_key(df["scientificName"], name_table)]
    )


def found_by_blast(
    df: pd.DataFrame,
    ncbi_search_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
) -> pd.Series:
    """Whether NCBI BLAST returned each row's species for the row's sequence,
    under any of its names (a hit the MOL filter rejected counts too).

    Names of hits are known as synonyms only when they reached the MOL list
    (the name table is built from it).

    Args:
        df: Rows with seq_id and scientificName (a species name).
        ncbi_search_df: Raw BLAST hits, before the MOL filter.
        name_table: See build_name_table.

    Returns:
        Boolean Series aligned on df's index.
    """
    if ncbi_search_df is None or ncbi_search_df.empty or df.empty:
        return pd.Series(False, index=df.index)
    hits = _pairs(_rows_of(ncbi_search_df, df), name_table)
    return pd.Series(_pairs(df, name_table).isin(hits), index=df.index)


def ncbi_names(
    df: pd.DataFrame,
    mol_df: pd.DataFrame | None,
    name_table: pd.DataFrame | None,
    synonyms_only: bool = False,
) -> pd.Series:
    """The NCBI names the MOL hits of each row's sequence gave its species
    (Darwin Core verbatimIdentification).

    Args:
        df: Rows with seq_id and scientificName (a species name).
        mol_df: MOL list (NCBI names).
        name_table: See build_name_table.
        synonyms_only: Leave out the names equal to the species name.

    Returns:
        Series aligned on df's index: the sorted list of names, empty (NaN)
        for a species no MOL hit of the sequence names.
    """
    if mol_df is None or mol_df.empty or df.empty:
        return pd.Series(pd.NA, index=df.index, dtype=object)
    hits = _rows_of(mol_df, df)[["seq_id", "scientificName"]].dropna().drop_duplicates()
    hits = hits.assign(_key=species_key(hits["scientificName"], name_table))
    if synonyms_only:
        hits = hits[name_keys(hits["scientificName"]) != hits["_key"]]
    names = (
        hits.sort_values("scientificName")
        .groupby(["seq_id", "_key"], observed=True)["scientificName"]
        .agg(list)
    )
    return pd.Series(
        names.reindex(_pairs(df, name_table)).to_numpy(), index=df.index, dtype=object
    )


SAME_SPECIES_COLUMNS = ["scientificName", "same_species_as", "group"]


def same_species_groups(
    species: pd.Series, name_table: pd.DataFrame | None
) -> pd.DataFrame:
    """Species that another source counts as one species.

    WoRMS can keep one species under two accepted names: a genus change left
    Larus saundersi accepted next to Chroicocephalus saundersi. Sources can
    also disagree with WoRMS on synonyms. Two species are linked when a
    source of the name table outside SYNONYM_SOURCES gives them the same ID
    or the same name, or gives one the name of the other. For GBIF the ID is GBIF's species for the name, so a
    synonym links to its accepted species and a subspecies to its species.

    Linked species form groups: A linked to B and B to C is one group.
    Nothing is merged: the groups are for the user to check.

    Args:
        species: Species names (repeats allowed); only links between them count.
        name_table: See build_name_table, with the rows the sources added.

    Returns:
        One row per linked species, sorted by group: scientificName,
        same_species_as (the species a source links it to, each with its
        sources, e.g. "Larus saundersi (GBIF)"), group (a number). Empty when
        no species is linked.
    """
    if name_table is None or name_table.empty:
        return pd.DataFrame(columns=SAME_SPECIES_COLUMNS)
    distinct = pd.Series(pd.unique(species.dropna()), dtype=object)
    names = dict(zip(name_keys(distinct), distinct))
    rows = name_table[
        ~name_table["source"].isin(SYNONYM_SOURCES)
        & name_table["species_key"].isin(names)
    ]
    links = {}  # species key -> {linked species key: sources}

    def link(a, b, source):
        links.setdefault(a, {}).setdefault(b, set()).add(source.upper())
        links.setdefault(b, {}).setdefault(a, set()).add(source.upper())

    for column in ("source_id", "name_key"):
        by_taxon = rows.dropna(subset=[column]).groupby(["source", column])
        for (source, _), keys in by_taxon["species_key"]:
            for a, b in itertools.combinations(set(keys), 2):
                link(a, b, source)
    # A source can give a species the name of another species of the list.
    for a, b, source in zip(rows["species_key"], rows["name_key"], rows["source"]):
        if b in names and b != a:
            link(a, b, source)
    if not links:
        return pd.DataFrame(columns=SAME_SPECIES_COLUMNS)

    # Groups are the connected components of the links.
    group_of, number = {}, 0
    for start in sorted(links, key=names.get):
        if start in group_of:
            continue
        number += 1
        todo = [start]
        while todo:
            key = todo.pop()
            if key not in group_of:
                group_of[key] = number
                todo += links[key]

    out = []
    for key, linked in links.items():
        same_as = sorted(
            f"{names[other]} ({', '.join(sorted(sources))})"
            for other, sources in linked.items()
        )
        out.append((names[key], ", ".join(same_as), group_of[key]))
    return (
        pd.DataFrame(out, columns=SAME_SPECIES_COLUMNS)
        .sort_values(["group", "scientificName"])
        .reset_index(drop=True)
    )
