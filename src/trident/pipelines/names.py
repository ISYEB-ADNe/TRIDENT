"""Species name table: the names each species goes by, across sources.

One row per (species, name, source). species is the WoRMS accepted name, or
the name itself when WoRMS does not know it. TAX builds the table from the
WoRMS check of the MOL (NCBI) names. Every step that compares names from two
sources goes through species_of, so a synonym counts as its species; names
match ignoring case and spacing. A name_table of None (names not checked in
WoRMS) matches names as they are.

    build_name_table(resolution_df, ncbi_taxids) → DataFrame
    species_of(names, name_table) → Series
    species_key(names, name_table) → Series
    best_hits(hits, name_table, columns) → DataFrame
    found_by_blast(df, ncbi_search_df, name_table) → Series
    ncbi_names(df, mol_df, name_table) → Series
"""

import pandas as pd

from trident.core.utils import normalize_name, top_hit_per_group


NAME_TABLE_COLUMNS = [
    "species",
    "aphia_id",
    "name",
    "source",
    "source_id",
    "matched_by",
    "name_key",
    "species_key",
]


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
        WoRMS did not find is its own species, without aphia_id or matched_by.
    """
    if resolution_df is None or resolution_df.empty:
        return pd.DataFrame(columns=NAME_TABLE_COLUMNS)
    # Rows cached before matchedBy existed lack that column.
    resolved = resolution_df.reindex(
        columns=["scientificName", "acceptedName", "acceptedNameUsageID", "matchedBy"]
    )
    names = resolved["scientificName"]
    species = resolved["acceptedName"].fillna(names)
    # Fixed dtypes: rows checked now and rows read back from the cache must
    # give the same table, whose fingerprint is part of later cache keys.
    return pd.DataFrame(
        {
            "species": species.astype(str),
            "aphia_id": _ids(resolved["acceptedNameUsageID"]),
            "name": names.astype(str),
            "source": "ncbi",
            "source_id": _ids(names.map(ncbi_taxids or {})),
            "matched_by": resolved["matchedBy"].astype("string"),
            "name_key": names.map(normalize_name),
            "species_key": species.map(normalize_name),
        }
    )[NAME_TABLE_COLUMNS]


def _ids(values: pd.Series) -> pd.Series:
    """Numeric ids as nullable integers, whatever type they were stored as."""
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def name_keys(names: pd.Series) -> pd.Series:
    """normalize_name of each name, computed once per distinct name: hit
    tables repeat their names."""
    distinct = pd.Series(pd.unique(names.dropna()), dtype=object)
    keys = dict(zip(distinct, distinct.map(normalize_name)))
    # Object even when no name is given, so it joins with other name keys.
    return names.map(keys).astype(object)


def _by_name(name_table: pd.DataFrame) -> pd.DataFrame:
    """The name table indexed by name_key, one row per name (the first)."""
    return name_table.drop_duplicates("name_key").set_index("name_key")


def species_of(names: pd.Series, name_table: pd.DataFrame | None) -> pd.Series:
    """The species each name stands for: its species in the name table, the
    name itself when the table does not list it."""
    if name_table is None or name_table.empty:
        return names.copy()
    return name_keys(names).map(_by_name(name_table)["species"]).fillna(names)


def species_key(names: pd.Series, name_table: pd.DataFrame | None) -> pd.Series:
    """species_of as a join key (normalize_name)."""
    keys = name_keys(names)
    if name_table is None or name_table.empty:
        return keys
    return keys.map(_by_name(name_table)["species_key"]).fillna(keys).astype(object)


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
