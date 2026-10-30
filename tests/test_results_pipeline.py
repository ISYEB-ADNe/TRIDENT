"""Tests for trident.pipelines.results_pipeline — results assembly logic."""

import pandas as pd
import pytest

from trident.pipelines import results_pipeline
from trident.pipelines.names import build_name_table
from trident.pipelines.results_pipeline import (
    EXPORT_COLS,
    build_assignments,
    build_results_df,
    CHECK_HIT_NOTE,
    flow_lost_at,
    flows_lost_at,
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


def test_excluded_species_leave_the_results_and_an_emptied_sequence_stays_blank():
    """No exclusion changes nothing; excluding the only species of s2 leaves
    a row with its sequence and no species."""
    assert filter_excluded_results(_curated_df(), set()).equals(_curated_df())
    assert filter_excluded_results(_curated_df(), None).equals(_curated_df())
    out = filter_excluded_results(_curated_df(), {"s1||Gadus ogac", "s2||Salmo salar"})
    assert out["seq_id"].tolist() == ["s1", "s2"]
    assert out["scientificName"].iloc[0] == "Gadus morhua"
    assert pd.isna(out["scientificName"].iloc[1])
    assert out["dna_sequence"].iloc[1] == "CCC"


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
            "mol_identity": [99.0, 97.0, 95.0],
            "mol_query_cover": [100, 100, 98],
            "mol_hit_url": ["url1", "url2", "url3"],
            "geo_occurrences": [100, 200, 50],
            "geo_url": ["gurl1", "gurl2", "gurl3"],
            "taxonURL": ["wurl1", "wurl2", "wurl3"],
        }
    )


@pytest.fixture
def hypo_df():
    return pd.DataFrame(
        {
            "seq_id": ["s2"],
            "scientificName": ["SpecD"],
            "proxy_hit_species": ["SpecC"],
            "proxy_identity": [88.0],
            "proxy_query_cover": [95],
            "proxy_url": ["bold_url"],
            "hypo_check_identity": [92.0],
            "hypo_check_query_cover": [100],
        }
    )


# --- build_results_df ---


class TestBuildResultsDf:
    def test_one_row_per_species_and_per_empty_sequence_in_input_order(
        self, geo_df, mol_df, hypo_df
    ):
        """Every EXPORT_COL is present; a sequence without species gets a row
        with its sequence."""
        sequences_df = pd.DataFrame(
            {"seq_id": ["s3", "s2", "s1"], "dna_sequence": ["TTTT", "GCTA", "ATCG"]}
        )
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
        )
        assert set(EXPORT_COLS) <= set(result.columns)
        assert result["seq_id"].tolist() == ["s3", "s2", "s2", "s1", "s1"]
        s3 = result.iloc[0]
        assert pd.isna(s3["scientificName"]) and s3["dna_sequence"] == "TTTT"

    def test_mol_geo_rows_keep_their_best_mol_hit(self, sequences_df, geo_df, mol_df):
        result = build_results_df(sequences_df, geo_df, mol_df, name_table=None)
        mol_geo = result[result["validation_step"] == "MOL+GEO"]
        assert mol_geo["scientificName"].tolist() == ["SpecA", "SpecB", "SpecC"]
        assert mol_geo["mol_identity"].tolist() == [99.0, 97.0, 95.0]

    @pytest.mark.parametrize("hypo", [None, pd.DataFrame()])
    def test_no_hypo_list_gives_only_mol_geo_rows(
        self, sequences_df, geo_df, mol_df, hypo
    ):
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo, name_table=None
        )
        assert set(result["validation_step"].dropna()) == {"MOL+GEO"}

    def test_hypo_rows_keep_their_proxy_hit(
        self, sequences_df, geo_df, mol_df, hypo_df
    ):
        result = build_results_df(
            sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
        )
        hypo = result[result["validation_step"] == "HYPO"]
        assert hypo["scientificName"].tolist() == ["SpecD"]
        assert hypo["proxy_identity"].tolist() == [88.0]

    def test_hypo_species_carries_its_mol_link(self, sequences_df, geo_df, mol_df):
        """A HYPO species gets the MOL species its proxy matched and the
        sequence's MOL identity to it, found under the NCBI name or the WoRMS
        accepted name of that species."""
        hypo_df = pd.DataFrame(
            {
                "seq_id": ["s1", "s2"],
                "scientificName": ["SpecX", "SpecY"],
                "proxy_hit_species": ["SpecB", "SpecC accepted"],
                "proxy_identity": [97.0, 96.0],
                "proxy_query_cover": [90, 90],
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
        assert hypo["proxy_hit_species"].tolist() == ["SpecB", "SpecC accepted"]
        assert hypo["via_mol_identity"].tolist() == [97.0, 95.0]


def test_export_names_a_species_by_the_ncbi_names_of_its_mol_hits(sequences_df):
    """A MOL+GEO species that NCBI named by a synonym gets its NCBI names as
    verbatimIdentification in the GBIF export."""
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
    export = build_gbif_export_df(results).set_index("seq_id")
    assert export.loc["s1", "verbatimIdentification"] == (
        "Gadus macrocephalus, Gadus ogac"
    )


# --- check_hit flag ---


def test_any_hypo_check_hit_is_flagged(sequences_df, geo_df, mol_df):
    """A hypothetical species has no marker in GenBank, so any check hit flags
    it, whatever its identity or cover; MOL+GEO species and empty sequences
    are never flagged."""
    hypo_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s2"],
            "scientificName": ["SpecX", "SpecY", "SpecZ"],
            "proxy_hit_species": ["SpecA", "SpecC", "SpecC"],
            "hypo_check_identity": [80.0, None, 100.0],
            "hypo_check_query_cover": [60.0, None, 100.0],
        }
    )
    result = build_results_df(
        sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
    )
    flagged = result.loc[result["hypo_check_hit"], "scientificName"].tolist()
    assert sorted(flagged) == ["SpecX", "SpecZ"]


def test_assignment_notes_say_what_to_check_and_where_a_sequence_lost_its_species():
    """An assigned sequence has no note; one below the identity threshold or
    with a HYPO check hit says so; one without a species gets the reason of
    the step where it lost its last one, or "no candidate left"."""
    sequences_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s2", "s3", "s4"],
            "dna_sequence": ["ATCG", "GCTA", "TTTT", "GGGG"],
        }
    )
    species = {
        "seq_id": ["s1", "s2"],
        "scientificName": ["Gadus morhua", "Salmo salar"],
        "taxonRank": "Species",
        "genus": ["Gadus", "Salmo"],
        "specificEpithet": ["morhua", "salar"],
    }
    mol_df = pd.DataFrame(
        {**species, "identity_percentage": [99.0, 95.0], "query_cover": 100}
    )
    geo_df = pd.DataFrame(
        {
            **species,
            "in_mol": True,
            "mol_identity": [99.0, 95.0],
            "mol_query_cover": 100,
        }
    )
    hypo_df = pd.DataFrame(
        {
            "seq_id": ["s2"],
            "scientificName": ["Salmo trutta"],
            "taxonRank": "Species",
            "genus": ["Salmo"],
            "specificEpithet": ["trutta"],
            "proxy_hit_species": ["Salmo salar"],
            "hypo_check_identity": [92.0],
            "hypo_check_query_cover": [100],
        }
    )
    results = build_results_df(
        sequences_df, geo_df, mol_df, hypo_df=hypo_df, name_table=None
    )
    lost = {"s3": ("GEO", {"Gadus ogac"}), "s4": (None, set())}

    def lost_at(seq_ids):
        assert sorted(seq_ids) == ["s3", "s4"]
        return {seq_id: lost[seq_id] for seq_id in seq_ids}

    notes = build_assignments(results, 97, 90, lost_at).set_index("seq_id")["note"]
    assert notes.to_dict() == {
        "s1": "",
        "s2": f"best identity below 97%, {CHECK_HIT_NOTE}",
        "s3": "no candidate recorded in the area (GEO)",
        "s4": "no candidate left",
    }


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
    for seq_id in ("s1", "s2"):
        links = build_species_flow(
            seq_id,
            ncbi_search_df,
            mol_df,
            tax_df,
            geo_df,
            extra_df,
            hypo_df,
            name_table=None,
        )
        assert flow_lost_at(links) == ("MOL Search", set())


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


def _worms_genus(aphia_id, kingdom, class_, status="accepted", accepted=None):
    """A WoRMS genus record; *accepted* is (AphiaID, authority) of the
    accepted record an unaccepted one points to."""
    valid_id, valid_authority = accepted or (aphia_id, "Author, 1800")
    return {
        "AphiaID": aphia_id,
        "valid_AphiaID": valid_id,
        "rank": "Genus",
        "status": status,
        "kingdom": kingdom,
        "class": class_,
        "authority": "Author, 1800",
        "valid_authority": valid_authority,
    }


def test_gbif_export_takes_the_worms_record_of_the_rows_group(monkeypatch):
    """A sequence resolved to a genus that is also a genus in another group
    gets the WoRMS record of its own class, even when the other is listed
    first and accepted while its own is not; an unaccepted record gives the
    ID and authority of the accepted record it points to."""
    monkeypatch.setattr(
        results_pipeline,
        "get_aphia_records",
        lambda name, session=None: [
            _worms_genus(425907, "Plantae", "Magnoliopsida"),
            _worms_genus(
                126160,
                "Animalia",
                "Teleostei",
                status="unaccepted",
                accepted=(126161, "Scopoli, 1777"),
            ),
        ],
    )
    results = pd.DataFrame(
        [_liparis("s1", _FISH, "liparis"), _liparis("s1", _FISH, "fabricii")]
    )
    row = build_gbif_export_df(results).iloc[0]
    assert row["scientificName"] == "Liparis"
    assert row["taxonID"] == "126161"
    assert row["scientificNameAuthorship"] == "Scopoli, 1777"


def test_gbif_export_saves_its_worms_lookups(monkeypatch, tmp_path):
    """The export's WoRMS records are saved in the analysis: the next export
    asks WoRMS only for the names whose request failed, which until then are
    reported and keep their ID empty."""
    requested = []

    def records(name, session=None):
        requested.append(name)
        if name == "Gadus" and requested.count(name) == 1:
            raise ConnectionError("WoRMS unreachable")
        return [
            _worms_genus(
                126160 if name == "Liparis" else 125732, "Animalia", "Teleostei"
            )
        ]

    monkeypatch.setattr(results_pipeline, "get_aphia_records", records)
    results = pd.DataFrame(
        [
            _liparis("s1", _FISH, "liparis"),
            _liparis("s1", _FISH, "fabricii"),
            {
                **_liparis("s2", _FISH, "morhua"),
                "genus": "Gadus",
                "scientificName": "Gadus morhua",
            },
            {
                **_liparis("s2", _FISH, "ogac"),
                "genus": "Gadus",
                "scientificName": "Gadus ogac",
            },
        ]
    )
    db_path = tmp_path / "analysis.db"

    first = build_gbif_export_df(results, db_path=db_path)
    assert first.attrs["failed_items"] == ["Gadus"]
    assert first.set_index("seq_id")["taxonID"].isna().to_dict() == {
        "s1": False,
        "s2": True,
    }

    second = build_gbif_export_df(results, db_path=db_path).set_index("seq_id")
    assert requested == ["Gadus", "Liparis", "Gadus"]
    assert second.attrs["failed_items"] == []
    assert second["taxonID"].to_dict() == {"s1": "126160", "s2": "125732"}


def test_gbif_export_common_taxon_is_a_real_taxon(monkeypatch):
    """The lowest common taxon of a sequence's species is shared at every rank
    above it, so homonym genera of two kingdoms have none; a common species
    is named by its binomial."""
    monkeypatch.setattr(
        results_pipeline, "get_aphia_records", lambda name, session=None: []
    )
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
    monkeypatch.setattr(
        results_pipeline, "get_aphia_records", lambda name, session=None: []
    )
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
    TAX on under its WoRMS accepted name; other sequences do not leak in. A
    species GBIF did not match ends apart from one GBIF counts left out."""
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
            "seq_id": ["s1"] * 7,
            "scientificName": [
                "Gadus macrocephalus",
                "Gadus morhua",
                "Gadus chalcogrammus",
                "Gadus nobold",
                "Gadus rejected",
                "Gadus extra",
                "Gadus unknown",
            ],
            "in_mol": [True, True, False, False, False, False, False],
        }
    )
    geo_df = tax_df.iloc[:5]
    # GBIF counted Gadus extra (too few records), but not Gadus unknown.
    gbif_merge_df = tax_df.iloc[:6]
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
        gbif_merge_df=gbif_merge_df,
    )
    table = species_flow_table("s1", links, ncbi_search_df, names)

    assert dict(zip(table["species"], table["outcome"])) == {
        "Bad fish": "Removed by the MOL filter",
        "Gadus rejected": "Removed by the MOL filter",
        "Salmo salar": "Not kept at TAX",
        "Gadus extra": "Not in the GEO area",
        "Gadus unknown": "Not matched in GBIF",
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
        "Gadus unknown",
    }
    assert table.set_index("species").loc["Gadus macrocephalus", "identity"] == 99.0


def test_lost_at_is_the_flows_last_loss():
    """A sequence without a species lost them where its flow's last loss is:
    a congener BLAST returned and the MOL filter rejected is a filter loss even
    when TAX keeps it, and expanded species wait for EXTRA, then HYPO. At GEO,
    a sequence whose candidates GBIF did not match lost them there, unless
    some were counted and found outside the area; a failed GBIF search
    outranks both."""
    ncbi_search_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["Gadus morhua", "Gadus rejected", "Gadus morhua"],
            "identity_percentage": [98.0, 70.0, 98.0],
        }
    )
    mol_df = ncbi_search_df.iloc[[0, 2]]
    # s1 keeps in the area only the rejected congener, s2 only an added one.
    tax_df = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2", "s2"],
            "scientificName": [
                "Gadus morhua",
                "Gadus rejected",
                "Gadus morhua",
                "Gadus added",
            ],
            "in_mol": [True, False, True, False],
        }
    )
    geo_df = tax_df.iloc[[1, 3]]
    proxied = tax_df.iloc[[3]]
    names = _names({})

    def lost_at(seq_id, extra_df, hypo_df):
        links = build_species_flow(
            seq_id,
            ncbi_search_df,
            mol_df,
            tax_df,
            geo_df,
            extra_df,
            hypo_df,
            name_table=names,
        )
        return flow_lost_at(links)

    assert lost_at("s1", proxied, proxied.iloc[:0]) == ("GEO", {"Gadus morhua"})
    added = {"Gadus added"}
    assert lost_at("s2", None, None) == ("EXTRA not run", added)
    assert lost_at("s2", proxied.iloc[:0], None) == ("EXTRA", added)
    assert lost_at("s2", proxied, None) == ("HYPO not run", added)
    assert lost_at("s2", proxied, proxied.iloc[:0]) == ("HYPO", added)

    # Species excluded by hand outrank a candidate still waiting for a step.
    links = build_species_flow(
        "s2",
        ncbi_search_df,
        mol_df,
        tax_df,
        tax_df.iloc[2:],
        None,
        None,
        excluded=["s2||Gadus morhua"],
        name_table=names,
    )
    assert flow_lost_at(links) == ("Curation", {"Gadus morhua"})

    def lost_at_geo(gbif_merge_df, gbif_failed=()):
        links = build_species_flow(
            "s2",
            ncbi_search_df,
            mol_df,
            tax_df,
            tax_df.iloc[[1]],
            name_table=names,
            gbif_merge_df=gbif_merge_df,
            failed={"gbif_search_failed": gbif_failed},
        )
        return flow_lost_at(links)

    unmatched = {"Gadus morhua", "Gadus added"}
    assert lost_at_geo(tax_df.iloc[[1]]) == ("GEO match", unmatched)
    assert lost_at_geo(tax_df.iloc[[1, 3]]) == ("GEO", {"Gadus added"})
    # A failed search waits for GEO to run again, whatever the others became,
    # also when another area's count worked; EXTRA comes after GEO.
    failed = lost_at_geo(tax_df.iloc[[1, 2, 3]], ["Gadus morhua"])
    assert failed == ("GEO failed", {"Gadus morhua"})
    links = build_species_flow(
        "s2",
        ncbi_search_df,
        mol_df,
        tax_df,
        tax_df.iloc[[3]],
        name_table=names,
        gbif_merge_df=tax_df.iloc[[2, 3]],
        failed={"gbif_search_failed": ["Gadus morhua"]},
    )
    assert flow_lost_at(links) == ("GEO failed", {"Gadus morhua"})


def test_a_candidate_whose_query_failed_ends_at_its_steps_failed_box():
    """Failed queries the user went on without are no results: a species
    whose genus WoRMS could not be asked about, and a candidate whose BOLD
    search, HYPO search or HYPO check failed, end at that step's failed box,
    not at "Not kept at TAX", "No BOLD proxy" or "Not validated by HYPO"; a
    sequence whose MOL BLAST search failed is lost at "MOL failed", not at "no
    NCBI hit"."""
    ncbi_search_df = pd.DataFrame(
        {"seq_id": "s1", "scientificName": ["Gadus morhua", "Salmo salar"]}
    )
    mol_df = ncbi_search_df
    added = ["Gadus nobold", "Gadus unsearched", "Gadus unchecked"]
    tax_df = pd.DataFrame(
        {
            "seq_id": "s1",
            "scientificName": ["Gadus morhua", *added],
            "in_mol": [True, False, False, False],
        }
    )
    extra_df = tax_df.iloc[2:]  # Gadus nobold has no proxy
    failed = {
        "tax_failed": ["Salmo"],
        "bold_search_failed": ["Gadus nobold"],
        "hypo_search_failed": [("s1", "Gadus unsearched")],
        "hypo_check_failed": [("s1", "Gadus unchecked")],
    }
    links = build_species_flow(
        "s1",
        ncbi_search_df,
        mol_df,
        tax_df,
        tax_df,
        extra_df,
        extra_df.iloc[:0],
        name_table=_names({}),
        failed=failed,
    )
    ends = {dst: names for _, dst, names in links if dst != "added_geo"}
    assert ends["tax_failed"] == {"Salmo salar"}
    assert ends["bold_search_failed"] == {"Gadus nobold"}
    assert ends["hypo_search_failed"] == {"Gadus unsearched"}
    assert ends["hypo_check_failed"] == {"Gadus unchecked"}
    assert not {"tax_out", "no_proxy", "hypo_out"} & set(ends)

    no_hits = ncbi_search_df.iloc[:0]
    lost = flows_lost_at(
        ["s2"],
        no_hits,
        no_hits,
        name_table=_names({}),
        failed={"ncbi_search_failed": ["s2"]},
    )
    assert lost == {"s2": ("MOL failed", set())}
