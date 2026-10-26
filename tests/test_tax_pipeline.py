"""Tests for TAX pipeline name resolution (R3) and the joins on species.

Covers the pure logic: the name table and species_of (a MOL name stands for
its WoRMS species), prepare_worms_input (accepted-genus expansion),
run_worms_merge (in_mol credited on the species), genus_homonyms (which genus
homonym is kept), finalize_tax_results (species MOL rejected),
summarize_name_resolution (what the TAX tab reports about the MOL names) and
ncbi_names (the NCBI names of a species). The cached run_name_resolution step
is exercised via its client contract in test_worms.py.
"""

import inspect

import pandas as pd

from trident.core.database import fingerprint
from trident.pipelines.names import (
    bold_name_rows,
    build_name_table,
    gbif_name_rows,
    ncbi_names,
    replace_source_rows,
    same_species_groups,
    species_of,
)
from trident.pipelines.tax_pipeline import (
    prepare_worms_input,
    prepare_resolution_input,
    genus_homonyms,
    run_worms_merge,
    finalize_tax_results,
    summarize_name_resolution,
)

# run_worms_merge is wrapped by @save_to_db + @preserve_sequence_order; unwrap to
# the pure logic so we can test it without a database.
_merge = inspect.unwrap(run_worms_merge)
_finalize = inspect.unwrap(finalize_tax_results)


def _table(accepted: dict, aphia_ids: dict | None = None) -> pd.DataFrame:
    """Name table of {NCBI name: WoRMS accepted name} (None: not in WoRMS)."""
    aphia_ids = aphia_ids or {}
    return build_name_table(
        pd.DataFrame(
            {
                "scientificName": list(accepted),
                "acceptedName": list(accepted.values()),
                "acceptedNameUsageID": [aphia_ids.get(n) for n in accepted],
                "matchedBy": ["name" if a else None for a in accepted.values()],
            }
        )
    )


# --- species_of ---


def test_a_name_stands_for_its_species_ignoring_case_and_spacing():
    """A name the table lists stands for its WoRMS species, also written with
    another case or spacing; a name it does not list, or one WoRMS did not
    find, stands for itself."""
    table = _table({"Gadus ogac": "Gadus macrocephalus", "Liparis sp.": None})
    names = pd.Series(["gadus  Ogac", "Liparis sp.", "Gadus morhua", None])
    assert species_of(names, table).tolist()[:3] == [
        "Gadus macrocephalus",
        "Liparis sp.",
        "Gadus morhua",
    ]
    assert species_of(names, None).tolist()[:3] == names.tolist()[:3]


def test_gbif_names_are_recorded_but_never_merge_two_worms_species():
    """GEO records the name GBIF counts each species under; that name can be
    another WoRMS species, which still stands for itself, and is no WoRMS
    rename or name missing from WoRMS."""
    table = _table({"Larus saundersi": "Larus saundersi"}, {"Larus saundersi": 343919})
    gbif_search = pd.DataFrame(
        {
            "scientificName": ["Larus saundersi"],
            "gbif_scientificName": ["Chroicocephalus saundersi"],
            "taxonID": ["5YGFL"],
            "gbif_matched_by": ["WoRMS ID"],
            "aphia_id": [343919],
        }
    )
    rows = gbif_name_rows(gbif_search)
    table = replace_source_rows(replace_source_rows(table, rows, "gbif"), rows, "gbif")

    gbif = table[table["source"] == "gbif"]
    assert gbif[["species", "name", "source_id"]].values.tolist() == [
        ["Larus saundersi", "Chroicocephalus saundersi", "5YGFL"]
    ]
    names = pd.Series(["Chroicocephalus saundersi", "Larus saundersi"])
    assert species_of(names, table).tolist() == names.tolist()
    summary = summarize_name_resolution(
        pd.DataFrame({"scientificName": ["Larus saundersi"]}), table
    )
    assert summary.renamed.empty and summary.not_found.empty


def test_species_a_source_counts_as_one_are_grouped_for_checking():
    """Species a source gives one ID or one name are linked, each to the
    species a source links it to, with that source; links from several
    sources form one group. GBIF links on its species for each name, also
    when it counts the names under different keys; a source that gives a
    species the name of another links them, even when that other species has
    no row from the source."""
    gbif = gbif_name_rows(
        pd.DataFrame(
            {
                "scientificName": ["Larus saundersi", "Chroicocephalus saundersi"],
                "gbif_scientificName": ["Larus saundersi", "Chroicocephalus saundersi"],
                "taxonID": ["9ABCD", "5YGFL"],
                "gbif_speciesID": ["5YGFL"] * 2,
                "gbif_matched_by": ["WoRMS ID"] * 2,
                "aphia_id": [343919, 1750258],
            }
        )
    )
    bold = bold_name_rows(
        pd.DataFrame(
            {
                "scientificName": ["Larus saundersi"],
                "bold_scientificName": ["Saundersilarus saundersi"],
                "taxonID": ["701"],
                "bold_matched_by": ["name"],
                "dna_sequence": ["acgt"],
            }
        )
    )
    table = replace_source_rows(replace_source_rows(None, gbif, "gbif"), bold, "bold")
    species = pd.Series(
        [
            "Larus saundersi",
            "Chroicocephalus saundersi",
            "Saundersilarus saundersi",
            "Chroicocephalus ridibundus",
            "Larus saundersi",
        ]
    )

    groups = same_species_groups(species, table)

    assert groups.set_index("scientificName").to_dict("index") == {
        "Chroicocephalus saundersi": {
            "same_species_as": "Larus saundersi (GBIF)",
            "group": 1,
        },
        "Larus saundersi": {
            "same_species_as": "Chroicocephalus saundersi (GBIF), "
            "Saundersilarus saundersi (BOLD)",
            "group": 1,
        },
        "Saundersilarus saundersi": {
            "same_species_as": "Larus saundersi (BOLD)",
            "group": 1,
        },
    }
    assert same_species_groups(species.iloc[:1], table).empty


def test_name_table_is_the_same_checked_now_or_read_from_the_cache():
    """Its fingerprint is part of the TAX cache key: WoRMS ids come back from
    the API as numbers (NaN when not found) and from SQLite as text."""
    fresh = pd.DataFrame(
        {
            "scientificName": ["Gadus ogac", "Liparis sp."],
            "acceptedName": ["Gadus macrocephalus", "Liparis sp."],
            "acceptedNameUsageID": [254538.0, float("nan")],
            "matchedBy": ["name", ""],
        }
    )
    cached = fresh.astype(object).assign(acceptedNameUsageID=["254538", None])
    taxids = {"Gadus ogac": 8053}
    assert fingerprint(build_name_table(fresh, taxids)) == fingerprint(
        build_name_table(cached, {"Gadus ogac": 8053.0})
    )


# --- prepare_worms_input ---


def test_prepare_uses_accepted_genus_cross_genus():
    # NCBI genus 'Allocentrotus' but accepted is 'Strongylocentrotus fragilis'.
    mol = pd.DataFrame({"scientificName": ["Allocentrotus fragilis", "Gadus morhua"]})
    table = _table({"Allocentrotus fragilis": "Strongylocentrotus fragilis"})
    assert prepare_worms_input(mol, table) == ["Strongylocentrotus", "Gadus"]


# --- run_worms_merge: in_mol on the species ---


def test_merge_credits_synonym_in_mol():
    # MOL hit 'Gadus ogac' stands for the accepted 'Gadus macrocephalus'.
    mol = pd.DataFrame(
        {
            "seq_id": ["s1"],
            "dna_sequence": ["ACGT"],
            "scientificName": ["Gadus ogac"],
            "identity_percentage": [99.0],
            "query_cover": [100.0],
            "hit_url": ["url1"],
        }
    )
    # WoRMS genus expansion of Gadus -> two accepted species.
    worms = pd.DataFrame(
        {
            "scientificName": ["Gadus macrocephalus", "Gadus morhua"],
            "genus": ["Gadus", "Gadus"],
        }
    )
    out = _merge(worms, mol, _table({"Gadus ogac": "Gadus macrocephalus"}))

    macro = out[out["scientificName"] == "Gadus macrocephalus"].iloc[0]
    morhua = out[out["scientificName"] == "Gadus morhua"].iloc[0]
    # synonym credited against the accepted species
    assert bool(macro["in_mol"]) is True
    assert (macro["mol_identity"], macro["mol_hit_url"]) == (
        99.0,
        "url1",
    )
    # the other expanded species is not a MOL hit
    assert bool(morhua["in_mol"]) is False


def test_ncbi_names_of_a_species_for_its_sequence():
    """A species gets the NCBI names its sequence's MOL hits gave it: all of
    them for the export, only those differing from it for the synonym column."""
    mol = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["Gadus macrocephalus", "Gadus ogac", "Gadus morhua"],
        }
    )
    rows = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["Gadus macrocephalus", "Gadus morhua", "Gadus morhua"],
        }
    )
    table = _table({"Gadus ogac": "Gadus macrocephalus"})
    names = ncbi_names(rows, mol, table)
    assert names.iloc[0] == ["Gadus macrocephalus", "Gadus ogac"]
    assert pd.isna(names.iloc[1])  # no MOL hit of s1 names Gadus morhua
    synonyms = ncbi_names(rows, mol, table, synonyms_only=True)
    assert synonyms.iloc[0] == ["Gadus ogac"]
    assert pd.isna(synonyms.iloc[2])  # NCBI used the accepted name


# --- genus homonyms (Liparis: a fish and a plant) ---


def _liparis_worms():
    """WoRMS species of the two accepted Liparis genera."""
    return pd.DataFrame(
        {
            "scientificName": [
                "Liparis liparis",
                "Liparis fabricii",
                "Liparis loeselii",
            ],
            "genus": ["Liparis"] * 3,
            "genusID": [126160, 126160, 425907],
            "kingdom": ["Animalia", "Animalia", "Plantae"],
            "class": ["Teleostei", "Teleostei", "Magnoliopsida"],
        }
    )


def _liparis_mol():
    """One sequence per case: a plant hit, a fish synonym, open nomenclature."""
    return pd.DataFrame(
        {
            "seq_id": ["plant", "fish", "open"],
            "dna_sequence": ["A", "C", "G"],
            "scientificName": ["Liparis loeselii", "Liparis oldname", "Liparis sp."],
            "identity_percentage": [99.0, 99.0, 99.0],
            "query_cover": [100.0, 100.0, 100.0],
            "hit_url": ["url1", "url2", "url3"],
        }
    )


_LIPARIS_NAMES = _table({"Liparis oldname": "Liparis fabricii"})


def test_each_sequence_keeps_the_genus_homonym_its_mol_species_belong_to():
    """Each sequence keeps the homonym its MOL species belong to, by accepted
    name (so a synonym counts too), and all of them when none of its species
    is in any (open nomenclature)."""
    tax = _merge(_liparis_worms(), _liparis_mol(), _LIPARIS_NAMES)
    kept = tax.groupby("seq_id")["scientificName"].agg(sorted).to_dict()
    assert kept == {
        "plant": ["Liparis loeselii"],
        "fish": ["Liparis fabricii", "Liparis liparis"],
        "open": ["Liparis fabricii", "Liparis liparis", "Liparis loeselii"],
    }


def test_homonym_report_says_which_mol_species_and_sequences_keep_each():
    """The TAX tab report lists each homonym with the MOL species found in it
    and how many of the sequences with that genus name keep it: the fish and
    the plant are each kept by their own sequence and by the open one."""
    worms = _liparis_worms()
    report = genus_homonyms(worms, _merge(worms, _liparis_mol(), _LIPARIS_NAMES))
    assert report["genusID"].tolist() == [126160, 425907]
    assert report["mol_species"].tolist() == ["Liparis fabricii", "Liparis loeselii"]
    assert report["sequences"].tolist() == ["2/3", "2/3"]


# --- summarize_name_resolution ---


def _resolved_mol():
    """The example's two synonyms, a name WoRMS does not know, a failed lookup,
    and their name table."""
    mol = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2", "s2", "s3", "s4"],
            "scientificName": [
                "Gadus ogac",
                "Gadus macrocephalus",
                "Myoxocephalus verrucosus",
                "Myoxocephalus scorpius",
                "Liparis sp.",
                "Atherinomorus lacunosus",
            ],
        }
    )
    # Atherinomorus lacunosus is absent from the table: its lookup failed.
    table = _table(
        {
            "Gadus ogac": "Gadus macrocephalus",
            "Gadus macrocephalus": "Gadus macrocephalus",
            "Myoxocephalus verrucosus": "Myoxocephalus scorpius",
            "Myoxocephalus scorpius": "Myoxocephalus scorpius",
            "Liparis sp.": None,
        },
        aphia_ids={
            "Gadus ogac": 254538,
            "Gadus macrocephalus": 254538,
            "Myoxocephalus verrucosus": 127203,
            "Myoxocephalus scorpius": 127203,
        },
    )
    return mol, table


def test_summary_counts_and_lists_the_synonyms_merged():
    """MOL names are counted as NCBI gives them and species as WoRMS accepts
    them; the synonyms WoRMS merges are listed."""
    summary = summarize_name_resolution(*_resolved_mol())
    assert (summary.n_names, summary.n_species) == (6, 4)
    assert summary.renamed[["scientificName", "accepted_name"]].values.tolist() == [
        ["Gadus ogac", "Gadus macrocephalus"],
        ["Myoxocephalus verrucosus", "Myoxocephalus scorpius"],
    ]


def test_not_found_names_exclude_failed_lookups_and_tell_their_genus():
    """A failed lookup (no row in the name table) is retried, not absent. A
    name not found whose genus is in TAX lost no candidates."""
    tax = pd.DataFrame({"genus": ["Liparis"]})
    summary = summarize_name_resolution(*_resolved_mol(), tax_df=tax)
    not_found = summary.not_found[["scientificName", "genus_in_tax"]]
    assert not_found.values.tolist() == [["Liparis sp.", True]]


def test_hits_without_a_name_are_neither_resolved_nor_expanded():
    """Also when no hit has a name at all."""
    mol = pd.DataFrame({"scientificName": [None, "Gadus morhua"]})
    assert prepare_resolution_input(mol) == ["Gadus morhua"]
    assert prepare_worms_input(mol, None) == ["Gadus"]
    assert summarize_name_resolution(mol, None).n_names == 1

    nameless = mol.iloc[:1]
    assert prepare_worms_input(nameless, None) == []
    assert summarize_name_resolution(nameless, None).n_names == 0


def test_species_rejected_by_mol_are_dropped_for_that_sequence_only():
    """A species BLAST returned for a sequence but MOL rejected leaves that
    sequence's TAX list, under its NCBI name or a synonym, and stays elsewhere."""
    tax = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s1", "s1", "s2"],
            "scientificName": [
                "Gadus morhua",
                "Gadus chalcogrammus",
                "Gadus macrocephalus",
                "Gadus nunavut",
                "Gadus chalcogrammus",
            ],
            "in_mol": [True, False, False, False, True],
        }
    )
    ncbi = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s1", "s2"],
            "scientificName": [
                "Gadus morhua",
                "Gadus chalcogrammus",
                "Gadus ogac",
                "Gadus chalcogrammus",
            ],
            "identity_percentage": [100.0, 95.0, 94.0, 100.0],
            "query_cover": [100.0, 100.0, 80.0, 100.0],
            "hit_url": ["url1", "url2", "url3", "url4"],
        }
    )
    # Gadus ogac reached the MOL list for another sequence, so its species is known.
    table = _table({"Gadus ogac": "Gadus macrocephalus"})

    out = _finalize(tax, ncbi_search_df=ncbi, name_table=table)
    assert set(zip(out["seq_id"], out["scientificName"])) == {
        ("s1", "Gadus morhua"),
        ("s1", "Gadus nunavut"),
        ("s2", "Gadus chalcogrammus"),
    }

    kept = _finalize(
        tax, ncbi_search_df=ncbi, name_table=table, drop_mol_rejected=False
    )
    assert len(kept) == len(tax)
