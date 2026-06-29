"""Tests for TAX pipeline name resolution (R3) and the accepted-name joins.

Covers the pure logic: prepare_worms_input (accepted-genus expansion),
apply_name_resolution (attach acceptedName/AphiaID), and run_worms_merge
(in_mol credited on the accepted name, verbatim preserved). The cached
run_name_resolution step is exercised via its client contract in test_worms.py.
"""

import inspect

import pandas as pd

from trident.pipelines.tax_pipeline import (
    prepare_worms_input,
    apply_name_resolution,
    run_worms_merge,
)

# run_worms_merge is wrapped by @save_to_db + @preserve_sequence_order; unwrap to
# the pure logic so we can test it without a database.
_merge = inspect.unwrap(run_worms_merge)


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
