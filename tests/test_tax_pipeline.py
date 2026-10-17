"""Tests for TAX pipeline name resolution (R3) and the accepted-name joins.

Covers the pure logic: prepare_worms_input (accepted-genus expansion),
apply_name_resolution (attach acceptedName/AphiaID), run_worms_merge (in_mol
credited on the accepted name, verbatim preserved), genus_homonyms (which
genus homonym is kept), finalize_tax_results (species MOL rejected) and
summarize_name_resolution
(what the TAX tab reports about the MOL names). The cached run_name_resolution
step is exercised via its client contract in test_worms.py.
"""

import inspect

import pandas as pd

from trident.pipelines.tax_pipeline import (
    prepare_worms_input,
    prepare_resolution_input,
    apply_name_resolution,
    genus_homonyms,
    run_worms_merge,
    finalize_tax_results,
    summarize_name_resolution,
)

# run_worms_merge is wrapped by @save_to_db + @preserve_sequence_order; unwrap to
# the pure logic so we can test it without a database.
_merge = inspect.unwrap(run_worms_merge)
_finalize = inspect.unwrap(finalize_tax_results)


def _resolution_df():
    return pd.DataFrame(
        [
            {
                "scientificName": "Gadus ogac",
                "acceptedName": "Gadus macrocephalus",
                "acceptedNameUsageID": 126437,
                "is_synonym": True,
            },
            {
                "scientificName": "Boreogadus saida",
                "acceptedName": "Boreogadus saida",
                "acceptedNameUsageID": 126433,
                "is_synonym": False,
            },
        ]
    )


# --- apply_name_resolution ---


def test_apply_attaches_accepted_and_aphia():
    mol = pd.DataFrame(
        {"seq_id": ["s1", "s2"], "scientificName": ["Gadus ogac", "Boreogadus saida"]}
    )
    out = apply_name_resolution(mol, _resolution_df())
    assert out["scientificName"].tolist() == [
        "Gadus ogac",
        "Boreogadus saida",
    ]  # raw kept
    assert out["acceptedName"].tolist() == ["Gadus macrocephalus", "Boreogadus saida"]
    assert out["acceptedNameUsageID"].tolist() == [126437, 126433]


def test_apply_unresolved_name_falls_back_to_raw():
    mol = pd.DataFrame({"seq_id": ["s1"], "scientificName": ["Ammodytes sp."]})
    out = apply_name_resolution(mol, _resolution_df())
    assert out["acceptedName"].iloc[0] == "Ammodytes sp."  # not in resolution -> raw
    assert pd.isna(out["acceptedNameUsageID"].iloc[0])


def test_apply_empty_resolution_keeps_raw():
    mol = pd.DataFrame({"seq_id": ["s1"], "scientificName": ["Gadus ogac"]})
    out = apply_name_resolution(mol, pd.DataFrame())
    assert out["acceptedName"].iloc[0] == "Gadus ogac"


# --- prepare_worms_input ---


def test_prepare_uses_accepted_genus_cross_genus():
    # Raw genus 'Allocentrotus' but accepted is 'Strongylocentrotus fragilis'.
    mol = pd.DataFrame(
        {
            "genus": ["Allocentrotus"],
            "acceptedName": ["Strongylocentrotus fragilis"],
        }
    )
    assert prepare_worms_input(mol) == ["Strongylocentrotus"]


def test_prepare_falls_back_to_genus_without_resolution():
    mol = pd.DataFrame({"genus": ["Gadus", "Gadus", "Boreogadus"]})
    assert sorted(prepare_worms_input(mol)) == ["Boreogadus", "Gadus"]


# --- run_worms_merge: in_mol on accepted name ---


def test_merge_credits_synonym_in_mol():
    # MOL hit raw 'Gadus ogac' resolved to accepted 'Gadus macrocephalus'.
    mol = pd.DataFrame(
        {
            "seq_id": ["s1"],
            "dna_sequence": ["ACGT"],
            "scientificName": ["Gadus ogac"],
            "acceptedName": ["Gadus macrocephalus"],
            "genus": ["Gadus"],
            "identity_percentage": [99.0],
            "query_cover": [100.0],
        }
    )
    # WoRMS genus expansion of Gadus -> two accepted species.
    worms = pd.DataFrame(
        {
            "scientificName": ["Gadus macrocephalus", "Gadus morhua"],
            "genus": ["Gadus", "Gadus"],
        }
    )
    out = _merge(worms, mol)

    macro = out[out["scientificName"] == "Gadus macrocephalus"].iloc[0]
    morhua = out[out["scientificName"] == "Gadus morhua"].iloc[0]
    # synonym credited against the accepted species
    assert bool(macro["in_mol"]) is True
    assert macro["verbatimIdentification"] == "Gadus ogac"  # synonym surfaced
    assert macro["mol_top_identity_percentage"] == 99.0
    # the other expanded species is not a MOL hit
    assert bool(morhua["in_mol"]) is False
    assert pd.isna(morhua["verbatimIdentification"])


def test_merge_synonym_surfaced_even_when_accepted_also_hit():
    # NCBI returned BOTH the accepted name and the synonym for the same species;
    # the accepted hit is the top score, but the synonym must still surface.
    mol = pd.DataFrame(
        {
            "seq_id": ["s1", "s1"],
            "dna_sequence": ["ACGT", "ACGT"],
            "scientificName": ["Gadus macrocephalus", "Gadus ogac"],
            "acceptedName": ["Gadus macrocephalus", "Gadus macrocephalus"],
            "genus": ["Gadus", "Gadus"],
            "identity_percentage": [99.5, 97.0],  # accepted name is the top hit
            "query_cover": [100.0, 100.0],
        }
    )
    worms = pd.DataFrame(
        {"scientificName": ["Gadus macrocephalus"], "genus": ["Gadus"]}
    )
    out = _merge(worms, mol)
    row = out[out["scientificName"] == "Gadus macrocephalus"].iloc[0]
    assert bool(row["in_mol"]) is True
    assert (
        row["verbatimIdentification"] == "Gadus ogac"
    )  # synonym, not the accepted dup
    assert row["mol_top_identity_percentage"] == 99.5  # top hit score


def test_merge_without_resolution_matches_raw_name():
    # No acceptedName column -> falls back to raw scientificName matching.
    mol = pd.DataFrame(
        {
            "seq_id": ["s1"],
            "dna_sequence": ["ACGT"],
            "scientificName": ["Gadus morhua"],
            "genus": ["Gadus"],
            "identity_percentage": [98.0],
            "query_cover": [100.0],
        }
    )
    worms = pd.DataFrame({"scientificName": ["Gadus morhua"], "genus": ["Gadus"]})
    out = _merge(worms, mol)
    assert bool(out.iloc[0]["in_mol"]) is True
    # No resolution -> raw == accepted -> not a synonym -> blank.
    assert pd.isna(out.iloc[0]["verbatimIdentification"])


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
            "acceptedName": ["Liparis loeselii", "Liparis fabricii", "Liparis sp."],
            "identity_percentage": [99.0, 99.0, 99.0],
            "query_cover": [100.0, 100.0, 100.0],
        }
    )


def test_each_sequence_keeps_the_genus_homonym_its_mol_species_belong_to():
    """Each sequence keeps the homonym its MOL species belong to, by accepted
    name (so a synonym counts too), and all of them when none of its species
    is in any (open nomenclature)."""
    tax = _merge(_liparis_worms(), _liparis_mol())
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
    report = genus_homonyms(worms, _merge(worms, _liparis_mol()))
    assert report["genusID"].tolist() == [126160, 425907]
    assert report["molSpecies"].tolist() == ["Liparis fabricii", "Liparis loeselii"]
    assert report["sequences"].tolist() == ["2/3", "2/3"]


# --- summarize_name_resolution ---


def _resolved_mol():
    """The example's two synonyms, a name WoRMS does not know, a failed lookup."""
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
    resolution = pd.DataFrame(
        {
            "scientificName": [
                "Gadus ogac",
                "Gadus macrocephalus",
                "Myoxocephalus verrucosus",
                "Myoxocephalus scorpius",
                "Liparis sp.",
            ],
            "acceptedName": [
                "Gadus macrocephalus",
                "Gadus macrocephalus",
                "Myoxocephalus scorpius",
                "Myoxocephalus scorpius",
                "Liparis sp.",
            ],
            "acceptedNameUsageID": [254538, 254538, 127203, 127203, None],
            "is_synonym": [True, False, True, False, False],
            "matchedBy": ["name", "name", "name", "name", None],
        }
    )
    # Atherinomorus lacunosus is absent from the resolution: its lookup failed.
    return apply_name_resolution(mol, resolution)


def test_summary_counts_and_lists_the_synonyms_merged():
    """MOL names are counted as NCBI gives them and species as WoRMS accepts
    them; the synonyms WoRMS merges are listed."""
    summary = summarize_name_resolution(_resolved_mol())
    assert (summary.n_names, summary.n_species) == (6, 4)
    assert summary.renamed[["scientificName", "acceptedName"]].values.tolist() == [
        ["Gadus ogac", "Gadus macrocephalus"],
        ["Myoxocephalus verrucosus", "Myoxocephalus scorpius"],
    ]


def test_not_found_names_exclude_failed_lookups_and_tell_their_genus():
    """A failed lookup also has no AphiaID, but it is retried, not absent. A
    name not found whose genus is in TAX lost no candidates."""
    tax = pd.DataFrame({"genus": ["Liparis"]})
    summary = summarize_name_resolution(
        _resolved_mol(), tax_df=tax, failed_names=["Atherinomorus lacunosus"]
    )
    assert summary.not_found.values.tolist() == [["Liparis sp.", True]]


def test_hits_without_a_name_are_neither_resolved_nor_expanded():
    """Also when no hit has a name at all."""
    mol = pd.DataFrame(
        {"scientificName": [None, "Gadus morhua"], "genus": [None, "Gadus"]}
    )
    assert prepare_resolution_input(mol) == ["Gadus morhua"]
    resolved = apply_name_resolution(mol, None)
    assert prepare_worms_input(resolved) == ["Gadus"]
    assert summarize_name_resolution(resolved).n_names == 1

    nameless = apply_name_resolution(mol.iloc[:1], None)
    assert prepare_worms_input(nameless) == []
    assert summarize_name_resolution(nameless).n_names == 0


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
        }
    )
    mol = pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3"],
            "scientificName": ["Gadus morhua", "Gadus chalcogrammus", "Gadus ogac"],
            "acceptedName": [
                "Gadus morhua",
                "Gadus chalcogrammus",
                "Gadus macrocephalus",
            ],
        }
    )

    out = _finalize(tax, ncbi_search_df=ncbi, mol_df=mol)
    assert set(zip(out["seq_id"], out["scientificName"])) == {
        ("s1", "Gadus morhua"),
        ("s1", "Gadus nunavut"),
        ("s2", "Gadus chalcogrammus"),
    }

    kept = _finalize(tax, ncbi_search_df=ncbi, mol_df=mol, drop_mol_rejected=False)
    assert len(kept) == len(tax)
