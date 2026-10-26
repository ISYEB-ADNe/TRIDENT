"""Each list (TAX, GEO, EXTRA, HYPO) has its declared columns, in their
order, whether it holds candidates or none."""

import json
from pathlib import Path

import pandas as pd
import pytest

import trident.pipelines as pipe
from trident.clients import bold, ncbi, worms
from trident.pipelines.extra_pipeline import EXTRA_COLUMNS
from trident.pipelines.geo_pipeline import GEO_LIST_COLUMNS
from trident.pipelines.hypo_pipeline import (
    HYPO_COLUMNS,
    HYPO_SEARCH_COLUMNS,
)
from trident.pipelines.tax_pipeline import TAX_COLUMNS

BOLD = json.loads((Path(__file__).parent / "fixtures/bold_records.json").read_text())
SEQUENCE = "ACGTACGTAC"


def _worms_species(name, aphia_id):
    """A WoRMS accepted species record, as AphiaChildrenByAphiaID gives it."""
    return {
        "AphiaID": aphia_id,
        "valid_AphiaID": aphia_id,
        "valid_name": name,
        "valid_authority": "(Linnaeus, 1766)",
        "status": "accepted",
        "rank": "Species",
        "kingdom": "Animalia",
        "phylum": "Chordata",
        "class": "Aves",
        "order": "Charadriiformes",
        "family": "Laridae",
        "genus": "Leucophaeus",
        "url": f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}",
        "genusID": 1,
    }


def _lists():
    """The four lists of one sequence whose MOL hit is Leucophaeus atricilla,
    with L. pipixcan added by TAX, found in the area and in BOLD, and
    confirmed by HYPO through L. atricilla."""
    mol, _ = pipe.finalize_mol_results(
        pd.DataFrame(
            {
                **dict.fromkeys(ncbi.BLAST_HIT_COLUMNS, [None]),
                "seq_id": ["s1"],
                "dna_sequence": [SEQUENCE],
                "genus": ["Leucophaeus"],
                "scientificName": ["Leucophaeus atricilla"],
                "identity_percentage": [99.0],
                "query_cover": [100.0],
                "filter_method": ["similarity"],
            }
        )
    )
    worms_df, _, _ = worms._build_taxonomy_dataframe(
        [
            _worms_species("Leucophaeus atricilla", 1),
            _worms_species("Leucophaeus pipixcan", 2),
        ],
        accepted_only=True,
    )
    merge, _ = pipe.run_worms_merge(worms_df, mol, None, mol_params={})
    tax, _ = pipe.finalize_tax_results(merge, ncbi_search_df=None, name_table=None)

    gbif_search = pd.DataFrame(
        {
            "scientificName": tax["scientificName"],
            "taxonURL": "https://www.gbif.org/taxon/X",
            "occurrences": 12,
            "gbif_extent": "500",
        }
    )
    geo_merge, _ = pipe.run_gbif_merge(
        gbif_search, tax, tax_params={}, gbif_search_params={}
    )
    geo, _ = pipe.run_gbif_filter(geo_merge, extent=500, min_occurrences=3)

    records, _ = bold._clean_records(BOLD["Leucophaeus pipixcan"]["records"])
    bold_filter = records.assign(scientificName="Leucophaeus pipixcan").head(1)
    extra, _ = pipe.run_bold_merge(bold_filter, geo, None, None, geo_params={})

    queries = pipe.hypo_queries(tax, mol, None)
    search = pd.DataFrame(
        {
            "seq_id": extra["proxy_id"],
            "scientificName": "Leucophaeus atricilla",
            "identity_percentage": 98.0,
            "query_cover": 100.0,
            "hit_def": "Leucophaeus atricilla COI",
            "hit_url": "https://www.ncbi.nlm.nih.gov/nucleotide/X",
            "entrez_query": queries["entrez_query"].iloc[0],
        }
    )[HYPO_SEARCH_COLUMNS]
    hypo_merge, _ = pipe.run_hypo_merge(search, extra, queries, extra_params={})
    hypo_filter, _ = pipe.run_hypo_filter(hypo_merge)
    # The check, as the app gets it when nothing was checked
    no_check, _ = pipe.run_hypo_check(hypo_filter.iloc[:0])
    hypo, _ = pipe.finalize_hypo_results(hypo_filter, no_check)
    return mol, tax, geo, extra, hypo


def test_lists_with_candidates_have_their_declared_columns():
    _, tax, geo, extra, hypo = _lists()
    assert not hypo.empty  # the candidate went through every step
    assert list(tax.columns) == TAX_COLUMNS
    assert list(geo.columns) == GEO_LIST_COLUMNS
    assert list(extra.columns) == EXTRA_COLUMNS
    assert list(hypo.columns) == HYPO_COLUMNS


@pytest.mark.parametrize("step", ["tax", "geo", "extra", "hypo"])
def test_empty_lists_have_their_declared_columns(step):
    mol, _ = pipe.finalize_mol_results(ncbi.process_blast_results(pd.DataFrame()))
    tax = pipe.empty_tax_results(mol)
    geo = pipe.empty_geo_results(tax)
    extra = pipe.empty_extra_results(geo)
    hypo = pipe.empty_hypo_results(extra)
    declared = {
        "tax": (tax, TAX_COLUMNS),
        "geo": (geo, GEO_LIST_COLUMNS),
        "extra": (extra, EXTRA_COLUMNS),
        "hypo": (hypo, HYPO_COLUMNS),
    }
    df, columns = declared[step]
    assert list(df.columns) == columns
