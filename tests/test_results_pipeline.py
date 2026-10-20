"""Tests for trident.pipelines.results_pipeline — results assembly logic."""

import pandas as pd
import pytest

from trident.pipelines import results_pipeline
from trident.pipelines.names import build_name_table
from trident.pipelines.results_pipeline import (
    EXPORT_COLS,
    build_results_df,
    find_sequence_exclusion_step,
    filter_excluded_results,
    build_gbif_export_df,
    build_species_flow,
    species_flow_table,
)


def _names(accepted: dict) -> pd.DataFrame:
    """Name table of {NCBI name: WoRMS accepted name}."""
    return build_name_table(
        pd.DataFrame(
            {
                "scientificName": list(accepted),
                "acceptedName": list(accepted.values()),
                "acceptedNameUsageID": range(len(accepted)),
                "matchedBy": "name",
            }
        )
    )


def _curated_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "dna_sequence": ["AAA", "AAA", "CCC"],
            "scientificName": ["Gadus morhua", "Gadus ogac", "Salmo salar"],
        }
    )


def test_filter_excluded_noop_when_empty():
    expected = _curated_df()
    assert filter_excluded_results(_curated_df(), set()).equals(expected)
    assert filter_excluded_results(_curated_df(), None).equals(expected)


def test_filter_excluded_drops_selected_species():
    out = filter_excluded_results(_curated_df(), {"s1||Gadus ogac"})
    assert "Gadus ogac" not in out["scientificName"].tolist()
    assert "Gadus morhua" in out["scientificName"].tolist()
    assert len(out) == 2


def test_filter_excluded_keeps_emptied_seq_as_blank_row():
    # Excluding the only species of s2 leaves an empty placeholder row.
    out = filter_excluded_results(_curated_df(), {"s2||Salmo salar"})
    s2 = out[out["seq_id"] == "s2"]
    assert len(s2) == 1
    assert pd.isna(s2["scientificName"].iloc[0])
    assert s2["dna_sequence"].iloc[0] == "CCC"


@pytest.fixture
def sequences_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3"],
            "dna_sequence": ["ATCG", "GCTA", "TTTT"],
        }
    )


@pytest.fixture
def mol_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["SpecA", "SpecB", "SpecC"],
            "identity_percentage": [99.0, 97.0, 95.0],
            "query_cover": [100, 100, 98],
            "hit_url": ["url1", "url2", "url3"],
        }
    )


@pytest.fixture
def geo_df():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["SpecA", "SpecB", "SpecC"],
            "in_mol": [True, True, True],
            "gbif_occurrences": [100, 200, 50],
            "gbif_taxonURL": ["gurl1", "gurl2", "gurl3"],
            "taxonURL": ["wurl1", "wurl2", "wurl3"],
        }
    )


@pytest.fixture
def hypo_df():
    return pd.DataFrame(
        {
            "seq_id": ["s2"],
            "scientificName": ["SpecD"],
            "identity_percentage": [88.0],
            "query_cover": [95],
            "seq_url": ["bold_url"],
            "ncbi_top_identity_percentage": [92.0],
            "ncbi_top_query_cover": [100],
        }
    )


# --- build_results_df ---


class TestBuildResultsDf:
    def test_mol_geo_only(self, sequences_df, geo_df, mol_df):
        result = build_results_df(sequences_df, geo_df, mol_df, name_table=None)
        assert "validation_step" in result.columns
        mol_geo = result[result["validation_step"] == "MOL+GEO"]
        assert len(mol_geo) == 3

    def test_with_hypo(self, sequences_df, geo_df, mol_df, hypo_df):
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
        )
        hypo = result[result["validation_step"] == "HYPO"]
        assert len(hypo) == 1
        assert hypo.iloc[0]["scientificName"] == "SpecD"

    def test_empty_sequences_get_row(self, sequences_df, geo_df, mol_df):
        """s3 has no species → gets an empty row with seq_id and dna_sequence."""
        result = build_results_df(sequences_df, geo_df, mol_df, name_table=None)
        s3_rows = result[result["seq_id"] == "s3"]
        assert len(s3_rows) == 1
        assert pd.isna(s3_rows.iloc[0]["scientificName"])
        assert s3_rows.iloc[0]["dna_sequence"] == "TTTT"

    def test_preserves_sequence_order(self, geo_df, mol_df):
        """Output follows the order of sequences_df."""
        sequences_df = pd.DataFrame(
            {"seq_id": ["s2", "s1"], "dna_sequence": ["G", "A"]}
        )
        result = build_results_df(sequences_df, geo_df, mol_df, name_table=None)
        seq_order = result["seq_id"].unique().tolist()
        assert seq_order == ["s2", "s1"]

    def test_all_export_cols_present(self, sequences_df, geo_df, mol_df, hypo_df):
        """All EXPORT_COLS should be present (possibly as NA)."""
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
        )
        for col in EXPORT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_hypo_renames_columns(self, sequences_df, geo_df, mol_df, hypo_df):
        """HYPO columns renamed: identity_percentage → proxy_identity_percentage."""
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
        )
        hypo = result[result["validation_step"] == "HYPO"]
        assert "proxy_identity_percentage" in result.columns
        assert hypo.iloc[0]["proxy_identity_percentage"] == 88.0

    def test_empty_hypo_df(self, sequences_df, geo_df, mol_df):
        """Empty hypo_df is treated like None."""
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=pd.DataFrame(), name_table=None
        )
        assert (result["validation_step"].dropna() == "MOL+GEO").all()

    def test_ncbi_identity_from_mol(self, sequences_df, geo_df, mol_df):
        """MOL+GEO rows get best identity from mol_df."""
        result = build_results_df(sequences_df, geo_df, mol_df, name_table=None)
        s1_specA = result[
            (result["seq_id"] == "s1") & (result["scientificName"] == "SpecA")
        ]
        assert s1_specA.iloc[0]["ncbi_top_identity_percentage"] == 99.0

    def test_hypo_species_carries_its_mol_link(self, sequences_df, geo_df, mol_df):
        """A HYPO species gets the MOL species its proxy matched and the
        sequence's MOL identity to it, found under the NCBI name or the WoRMS
        accepted name of that species."""
        hypo_df = pd.DataFrame(
            {
                "seq_id": ["s1", "s2"],
                "scientificName": ["SpecX", "SpecY"],
                "scientificName_hit": ["SpecB", "SpecC accepted"],
                "identity_percentage": [97.0, 96.0],
                "query_cover": [90, 90],
            }
        )
        result = build_results_df(
            sequences_df,
            geo_df,
            mol_df,
            hypo_df=hypo_df,
            name_table=_names({"SpecC": "SpecC accepted"}),
        )
        hypo = result[result["validation_step"] == "HYPO"]
        assert hypo["via_mol_species"].tolist() == ["SpecB", "SpecC accepted"]
        assert hypo["via_mol_identity"].tolist() == [97.0, 95.0]


def test_species_mol_found_under_a_synonym_keeps_its_ncbi_hit(sequences_df):
    """A MOL+GEO species that NCBI named by a synonym gets its best NCBI hit,
    and the GBIF export gives the NCBI names as verbatimIdentification."""
    mol_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s1"],
            "scientificName": ["Gadus ogac", "Gadus macrocephalus"],
            "identity_percentage": [99.0, 97.0],
            "query_cover": [100, 100],
            "hit_url": ["url1", "url2"],
        }
    )
    geo_df = pd.DataFrame(
        {
            "seq_id": ["s1"],
            "scientificName": ["Gadus macrocephalus"],
            "in_mol": [True],
        }
    )
    names = _names({"Gadus ogac": "Gadus macrocephalus"})
    results = build_results_df(sequences_df, geo_df, mol_df, name_table=names)
    row = results[results["seq_id"] == "s1"].iloc[0]
    assert (row["ncbi_top_identity_percentage"], row["ncbi_top_hit_url"]) == (
        99.0,
        "url1",
    )
    export = build_gbif_export_df(results).set_index("seq_id")
    assert export.loc["s1", "verbatimIdentification"] == (
        "Gadus macrocephalus, Gadus ogac"
    )


# --- find_sequence_exclusion_step ---


class TestFindExclusionStep:
    def test_basic(self):
        result = find_sequence_exclusion_step(
            ["s1", "s2", "s3"],
            ncbi_search_df=pd.DataFrame({"seq_id": ["s1", "s2", "s3"]}),
            mol_df=pd.DataFrame({"seq_id": ["s1", "s2"]}),
            geo_df=pd.DataFrame({"seq_id": ["s1"]}),
        )
        assert len(result) == 2
        s3 = result[result["seq_id"] == "s3"]
        assert s3.iloc[0]["pipeline_step"] == "MOL Filter"

    def test_no_steps(self):
        result = find_sequence_exclusion_step(["s1"])
        assert result.empty

    def test_all_present(self):
        result = find_sequence_exclusion_step(
            ["s1"],
            ncbi_search_df=pd.DataFrame({"seq_id": ["s1"]}),
            mol_df=pd.DataFrame({"seq_id": ["s1"]}),
        )
        assert result.empty


# --- check_hit flag ---


def test_any_hypo_check_hit_is_flagged(sequences_df, geo_df, mol_df):
    """A hypothetical species has no marker in GenBank, so any check hit flags
    it, whatever its identity or cover; MOL+GEO species and empty sequences
    are never flagged, although they carry an NCBI identity."""
    hypo_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s2"],
            "scientificName": ["SpecX", "SpecY", "SpecZ"],
            "ncbi_top_identity_percentage": [80.0, None, 100.0],
            "ncbi_top_query_cover": [60.0, None, 100.0],
        }
    )
    result = build_results_df(
        sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
    )
    flagged = result.loc[result["check_hit"], "scientificName"].tolist()
    assert sorted(flagged) == ["SpecX", "SpecZ"]


def test_empty_mol_list_flows_to_results(monkeypatch):
    """No NCBI hit at all: every later list is empty but has its columns, and
    Results still lists each sequence, as lost at the MOL step."""
    import requests
    from Bio import Blast

    import trident.pipelines as pipe
    from trident.clients.ncbi import BLAST_HIT_COLUMNS

    def offline(*args, **kwargs):
        raise AssertionError("an empty list must not query any database")

    monkeypatch.setattr(requests.Session, "send", offline)
    monkeypatch.setattr(Blast, "qblast", offline)

    sequences_df = pd.DataFrame({"seq_id": ["s1", "s2"], "dna_sequence": ["A", "C"]})
    ncbi_search_df = pd.DataFrame(columns=BLAST_HIT_COLUMNS)
    filter_df, _ = pipe.run_ncbi_filter(ncbi_search_df, db_path=None)
    mol_df, _ = pipe.finalize_mol_results(filter_df, db_path=None)

    tax_df = pipe.empty_tax_results(mol_df)
    geo_df = pipe.empty_geo_results(tax_df)
    extra_df = pipe.empty_extra_results(geo_df)
    hypo_df = pipe.empty_hypo_results(extra_df)

    for df in (tax_df, geo_df, extra_df, hypo_df):
        assert df.empty
        assert {"seq_id", "scientificName"} <= set(df.columns)
    results = build_results_df(sequences_df, geo_df, mol_df, hypo_df, name_table=None)
    assert sorted(results["seq_id"]) == ["s1", "s2"]
    lost = find_sequence_exclusion_step(
        ["s1", "s2"], ncbi_search_df, mol_df, tax_df, geo_df
    )
    assert set(lost["pipeline_step"]) == {"MOL Search"}


# --- GBIF export of genus homonyms (Liparis: a fish and a plant) ---

_FISH = {
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class": "Teleostei",
    "order": "Perciformes",
    "family": "Liparidae",
}
_PLANT = {
    "kingdom": "Plantae",
    "phylum": "Tracheophyta",
    "class": "Magnoliopsida",
    "order": "Asparagales",
    "family": "Orchidaceae",
}


def _liparis(seq_id, taxonomy, name):
    """A results row of a Liparis species (name without the genus)."""
    return {
        "seq_id": seq_id,
        "dna_sequence": "ACGT",
        **taxonomy,
        "genus": "Liparis",
        "specificEpithet": name.split()[0],
        "scientificName": f"Liparis {name}",
    }


def _worms_genus(aphia_id, kingdom, class_, status="accepted"):
    return {
        "AphiaID": aphia_id,
        "valid_AphiaID": aphia_id,
        "rank": "Genus",
        "status": status,
        "kingdom": kingdom,
        "class": class_,
        "authority": "Author, 1800",
    }


def test_gbif_export_takes_the_worms_record_of_the_rows_group(monkeypatch):
    """A sequence resolved to a genus that is also a genus in another group
    gets the WoRMS record of its own class, even when the other is listed
    first and accepted while its own is not."""
    monkeypatch.setattr(
        results_pipeline,
        "get_aphia_records",
        lambda name: [
            _worms_genus(425907, "Plantae", "Magnoliopsida"),
            _worms_genus(126160, "Animalia", "Teleostei", status="unaccepted"),
        ],
    )
    results = pd.DataFrame(
        [_liparis("s1", _FISH, "liparis"), _liparis("s1", _FISH, "fabricii")]
    )
    row = build_gbif_export_df(results).iloc[0]
    assert row["scientificName"] == "Liparis"
    assert row["taxonID"] == "126160"


def test_gbif_export_common_taxon_is_a_real_taxon(monkeypatch):
    """The lowest common taxon of a sequence's species is shared at every rank
    above it, so homonym genera of two kingdoms have none; a common species
    is named by its binomial."""
    monkeypatch.setattr(results_pipeline, "get_aphia_records", lambda name: [])
    results = pd.DataFrame(
        [
            _liparis("both", _FISH, "liparis"),
            _liparis("both", _PLANT, "loeselii"),
            _liparis("subspecies", _FISH, "fabricii"),
            _liparis("subspecies", _FISH, "fabricii fabricii"),
        ]
    )
    rows = build_gbif_export_df(results).set_index("seq_id")
    assert pd.isna(rows.loc["both", "scientificName"])
    assert rows.loc["both", "identificationRemarks"] == (
        "Liparis liparis, Liparis loeselii"
    )
    assert rows.loc["subspecies", "scientificName"] == "Liparis fabricii"
    assert rows.loc["subspecies", "taxonRank"] == "Species"


def test_resolve_sequences_matches_the_gbif_export(monkeypatch):
    """Each sequence resolves to the taxon the GBIF export writes for it, and is
    at species level when that taxon is a species or a subspecies; only a
    candidate named by its genus alone keeps the sequence at genus."""
    monkeypatch.setattr(results_pipeline, "get_aphia_records", lambda name: [])
    lone_subspecies = _liparis("lone", _FISH, "fabricii fabricii")
    lone_subspecies["taxonRank"] = "Subspecies"
    genus_only = {**_liparis("genus_only", _FISH, "x"), "specificEpithet": None}
    genus_only["scientificName"] = "Liparis"
    no_epithet = {**_liparis("no_epithet", _FISH, "fabricii"), "specificEpithet": None}
    results = pd.DataFrame(
        [
            lone_subspecies,
            _liparis("subspecies", _FISH, "fabricii"),
            _liparis("subspecies", _FISH, "fabricii fabricii"),
            _liparis("genus", _FISH, "liparis"),
            _liparis("genus", _FISH, "fabricii"),
            _liparis("both", _FISH, "liparis"),
            _liparis("both", _PLANT, "loeselii"),
            genus_only,
            _liparis("genus_only", _FISH, "fabricii"),
            no_epithet,
            _liparis("no_epithet", _FISH, "fabricii fabricii"),
            {"seq_id": "empty", "dna_sequence": "ACGT"},
        ]
    )
    resolved = results_pipeline.resolve_sequences(results).set_index("seq_id")
    export = build_gbif_export_df(results).set_index("seq_id")
    for col in ("scientificName", "taxonRank"):
        assert resolved[col].equals(export.loc[resolved.index, col])
    assert resolved.loc["genus_only", "taxonRank"] == "Genus"
    assert resolved["species_level"].to_dict() == {
        "lone": True,
        "subspecies": True,
        "genus": False,
        "both": False,
        "genus_only": False,
        "no_epithet": True,
        "empty": False,
    }
    # A subspecies counts with its species.
    assert resolved.loc["lone", "species"] == resolved.loc["subspecies", "species"]


# --- species flow ---


def test_species_flow_ends_each_candidate_once():
    """Each NCBI name and each species TAX added ends in exactly one place, from
    TAX on under its WoRMS accepted name; other sequences do not leak in."""
    ncbi_search_df = pd.DataFrame(
        {
            "seq_id": ["s1"] * 6 + ["s2"],
            "scientificName": [
                "Gadus ogac",
                "Gadus macrocephalus",
                "Gadus morhua",
                "Salmo salar",
                "Bad fish",
                "Gadus rejected",
                "Other fish",
            ],
            "identity_percentage": [99.0, 85.0, 98.0, 97.0, 80.0, 70.0, 99.0],
        }
    )
    # Gadus ogac passes the filter and WoRMS accepts it as G. macrocephalus,
    # whose own lower hit the filter rejected.
    mol_df = pd.DataFrame(
        {
            "seq_id": ["s1"] * 3,
            "scientificName": ["Gadus ogac", "Gadus morhua", "Salmo salar"],
            "identity_percentage": [99.0, 98.0, 97.0],
        }
    )
    names = _names({"Gadus ogac": "Gadus macrocephalus"})
    # TAX adds Gadus rejected back, as when it keeps MOL-rejected congeners.
    tax_df = pd.DataFrame(
        {
            "seq_id": ["s1"] * 6,
            "scientificName": [
                "Gadus macrocephalus",
                "Gadus morhua",
                "Gadus chalcogrammus",
                "Gadus nobold",
                "Gadus rejected",
                "Gadus extra",
            ],
            "in_mol": [True, True, False, False, False, False],
        }
    )
    geo_df = tax_df.iloc[:5]
    extra_df = pd.DataFrame(
        {"seq_id": ["s1"], "scientificName": ["Gadus chalcogrammus"]}
    )
    hypo_df = extra_df.copy()

    links = build_species_flow(
        "s1",
        ncbi_search_df,
        mol_df,
        tax_df=tax_df,
        geo_df=geo_df,
        extra_df=extra_df,
        hypo_df=hypo_df,
        excluded=["s1||Gadus macrocephalus", "s2||Gadus morhua"],
        name_table=names,
    )
    table = species_flow_table("s1", links, ncbi_search_df, names)

    assert dict(zip(table["species"], table["outcome"])) == {
        "Bad fish": "Removed by the MOL filter",
        "Gadus rejected": "Removed by the MOL filter",
        "Salmo salar": "Not kept at TAX",
        "Gadus extra": "Not in the GEO area",
        "Gadus nobold": "No BOLD proxy (EXTRA)",
        "Gadus macrocephalus": "Excluded by hand",
        "Gadus morhua": "Kept",
        "Gadus chalcogrammus": "Kept",
    }
    assert len(table) == table["species"].nunique()
    added = table.set_index("species")["origin"].eq("Added by TAX")
    assert set(added[added].index) == {
        "Gadus chalcogrammus",
        "Gadus nobold",
        "Gadus extra",
    }
    assert table.set_index("species").loc["Gadus macrocephalus", "identity"] == 99.0
