<p align="center">
  <img src="logo.png" alt="TRIDENT" width="160" />
</p>

<h1 align="center">TRIDENT</h1>

<p align="center">
  <strong>Taxonomic Resolution and IDentification using Environmental dNa Traces</strong>
</p>

<p align="center">
  <a href="https://trident-nxanjv7pk4z2fnkyulzvwt.streamlit.app/"><img src="https://img.shields.io/badge/live%20demo-Streamlit-22c55e?style=flat-square" alt="Live demo" /></a>
  <a href="https://doi.org/10.64898/2026.06.29.735257"><img src="https://img.shields.io/badge/preprint-bioRxiv-b91c1c?style=flat-square" alt="bioRxiv preprint" /></a>
  <a href="https://doi.org/10.5281/zenodo.21108395"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21108395-0e7490?style=flat-square" alt="Zenodo DOI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-0e7490?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/python-3.12-0e7490?style=flat-square" alt="Python 3.12" />
</p>

<p align="center">
  <a href="#what-it-does">Features</a> •
  <a href="#getting-started">Get started</a> •
  <a href="#how-to-use-the-app">Use</a> •
  <a href="#example">Example</a> •
  <a href="#pipeline-overview">Pipeline</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#how-to-cite">Cite</a> •
  <a href="#license">License</a>
</p>

TRIDENT identifies species from environmental DNA (eDNA) barcode sequences. You upload a FASTA file of MOTU/ASV sequences, and TRIDENT cross-checks them against four biodiversity databases (GenBank through NCBI, WoRMS, GBIF, BOLD) to give each sequence its candidate species and an assignment, ready for export.

It runs as a cross-platform web app (macOS, Windows, Linux), with no coding required once it is installed.

## What it does

TRIDENT takes your sequences through five steps. Each step has its own tab in the app, and you run them in order.

| Step | Tab | What happens | What you get |
|------|-----|--------------|--------------|
| 1 | **MOL** | Your sequences are BLAST-searched against GenBank (NCBI core_nt), and the hits are filtered by query cover and a barcoding gap | The MOL list: the species of the hits kept for each sequence |
| 2 | **TAX** | The MOL names are checked in WoRMS, and each MOL genus is expanded to its accepted species (marine only by default) | The TAX list: MOL species plus the species added by TAX |
| 3 | **GEO** | The TAX species are counted in GBIF occurrences within your study area | The GEO list: species with enough occurrences where you sampled |
| 4 | **EXTRA** | For the GEO species that are not MOL species (their target marker gave no kept hit), CO1 sequences are fetched from BOLD | Proxy sequences for these species |
| 5 | **HYPO** | The proxies are BLAST-searched against GenBank, restricted to the MOL species of each sequence | Hypothetical species, validated through their CO1 proxy |

The **Results** tab gives each sequence its candidate species, from two paths:

- **MOL+GEO**: species with a hit for your target marker, kept by GEO.
- **HYPO**: hypothetical species, kept by GEO and validated through a CO1 proxy.

## Getting started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)
- A contact email address (sent with every API request, required by NCBI, GBIF, WoRMS and BOLD)

### Quick start

```bash
uv sync
uv run trident
```

The app opens in your browser. Set your contact email in the **Settings** panel on first launch: an analysis can only be started or opened once it is set.

<details>
<summary><strong>Detailed installation guide (click to expand)</strong></summary>

#### Step 1: Install uv

Follow the instructions at <https://docs.astral.sh/uv/getting-started/installation/> or run:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Step 2: Get the source code

**Option A**: Clone with git:

```bash
git clone https://github.com/ISYEB-ADNe/TRIDENT.git
cd TRIDENT
```

**Option B**: From a zip file, unzip the archive and open a terminal in the extracted folder.

#### Step 3: Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs everything needed.

#### Step 4: Run the app

```bash
uv run trident
```

Set your contact email in the **Settings** panel on first launch, or pre-fill it in `.streamlit/secrets.toml`:

```toml
CONTACT_EMAIL = "your-email@example.com"
```

Copy `.streamlit/secrets.toml.example` to start; the notebooks and scripts read the same file.

If BLAST searches start failing after a change on NCBI's side, `BLAST_CLIENT = "qblast"` in `secrets.toml` switches to Biopython's BLAST client.

Each analysis is saved as a `.db` file in the `results/` folder of TRIDENT, whatever folder you start it from; the Start tab shows its path. `RESULTS_DIR` in `secrets.toml` (an absolute path) sets another folder.

</details>

## How to use the app

1. **Start Analysis**: Upload a FASTA file of your MOTU/ASV sequences (`.fasta`, `.fa`, `.fas`, `.txt`). Click **Process FASTA**, then **Proceed to MOL**.
2. **Run each step in order**: Open MOL, TAX, GEO, EXTRA, HYPO one at a time. Each tab has its own settings; review them, then run the step. A ✅ on the tab means the step is done.
3. **Set your study area in GEO**: Enter the latitude and longitude of your site and one or more extents (km). Species with too few occurrences in the filtering area are left out.
4. **Results**: The Results tab opens once MOL, TAX and GEO are done; EXTRA and HYPO add the hypothetical species. It shows a summary, the assignment of each sequence (its species, or the lowest taxon its candidate species share), the candidate species of a sequence with the step where each other species was lost, and the curated results, where you can leave out a species by hand.

The **Parameters** tab lists the settings of every step that has run.

### Exports

- **One row per candidate species** (`<analysis>_candidates.csv`): each sequence with its candidate species, its taxonomy, its path (MOL+GEO or HYPO), the identity and query cover of its hits, its GBIF occurrences and links to each source.
- **One row per sequence** (`<analysis>_assignments.csv`): each sequence at its assigned rank, following Darwin Core and the FAIRe taxa table, ready for a GBIF submission.
- **Parameters** (`<analysis>_parameters.txt`): the settings of every step, from the Parameters tab.
- **Analysis file (`.db`)**: the whole analysis, already saved in `results/`. Download it to keep a copy elsewhere or to share it, and open it again from the Start tab. It is a SQLite database, readable with any SQLite tool. To save space, each sequence of the step tables is stored once, in the `sequence_store` table: their `*_sequence` columns hold its `id` there. The `sequences` table (the FASTA) and the `*_inputs` tables hold the sequences themselves, as do rows saved by TRIDENT versions before the store.

### Online demo

A hosted demo is live on Streamlit Cloud: <https://trident-nxanjv7pk4z2fnkyulzvwt.streamlit.app/>. It is meant for trying TRIDENT on small inputs and may be limited. In particular, the EXTRA step (BOLD) may be blocked on the hosted demo, since BOLD can refuse requests from shared cloud IPs; run TRIDENT locally (above) for full analyses.

## Example

`example/example.fasta` holds the 4 Svalbard MOTUs of the paper's worked example (16S, Vert16S primers). Click "Try the example" on the Start tab for a quick end-to-end demo of the pipeline: it also sets the GEO area to the sampling site (latitude 78.5, longitude 14.5, extents 250 and 500 km). If you upload the file instead, enter these values in the GEO step.

<details>
<summary><strong>Run the example as a notebook (advanced)</strong></summary>

`example/example_pipeline.ipynb` runs the full pipeline (Load FASTA -> MOL -> TAX -> GEO -> EXTRA -> HYPO -> Results) step by step in Python on the same file. It calls the same `trident.pipelines` functions and SQLite cache as the app, and exposes every intermediate DataFrame for inspection or for integrating into your own scripts.

Open it in any Jupyter front-end (VS Code, JupyterLab, ...); the project ships the `ipykernel` and `ipywidgets` it needs. Before running, set your contact email and the study-area latitude / longitude / extents in the configuration cell near the top.

</details>

## Pipeline overview

Input MOTU/ASV sequences pass through five steps; the Results tab gives each sequence its candidate species from two paths (MOL+GEO: a hit for the target marker; HYPO: a hit through a CO1 proxy).

```text
MOL  (NCBI BLAST)  ->  TAX (WoRMS)  ->  GEO (GBIF)  ->  EXTRA (BOLD)  ->  HYPO (NCBI BLAST)
```

The settings of each step are detailed below. Defaults are those used by the app (`trident/core/constants.py`); every setting listed is adjustable in the app before running a step, except those marked as fixed.

<details>
<summary><strong>MOL: NCBI BLAST + barcoding-gap filter</strong></summary>

| Setting | Default | Meaning |
|---|---|---|
| BLAST database | core_nt (fixed) | NCBI's Core Nucleotide database, the default of NCBI's BLAST website; it keeps the mitochondrial records. Each GenBank record gives one hit, its best alignment. |
| Max hits per query | 500 | Upper bound on the BLAST hits returned per sequence (up to 1000). |
| E-value exponent | 20 (1e-20) | Hits with a higher e-value are not returned. |
| Query cover (%) | 90 | Minimum query cover of a hit; removes short partial hits. |
| Filtering method | Barcoding gap | See below; a sequence with no barcoding gap falls back to the similarity range. |
| Gap size (%) | 2 | Minimum identity drop between two consecutive hits to count as a barcoding gap, and the width of the similarity range. |
| Gap min top (%) | 97 | The top of the gap must be at or above this identity, so no gap is called among low-identity hits. |
| Low identity warning (%) | 95 | Flags species whose best hit is below this identity. A warning only, unless "Enforce as hard filter" is checked (default: no). |

**Barcoding gap:** within each sequence's hits (sorted by descending identity), find the first drop of at least the gap size among hits at or above the gap min top, and keep only the hits above that gap. A sequence with no such gap falls back to the **similarity range**: keep all hits within the gap size of the best hit's identity.

**Hit limit:** a sequence that reached the max hits with every hit within 3% identity of a top hit at 97% or more may have left other species with a high identity out. The MOL tab names these sequences and offers them as a FASTA file, if you want to search them again with more hits.

</details>

<details>
<summary><strong>TAX: WoRMS genus expansion</strong></summary>

| Setting | Default | Meaning |
|---|---|---|
| Marine species only | Yes | Expand each genus to its marine species only (WoRMS counts brackish species as marine too); uncheck to also add non-marine WoRMS records. |
| Exclude species already rejected by MOL | Yes | For each sequence, leave out the species added by TAX that BLAST returned for it but that the MOL filter did not keep. A species rejected for one sequence is kept for the others. |

</details>

<details>
<summary><strong>GEO: GBIF occurrence filter</strong></summary>

| Setting | Default | Meaning |
|---|---|---|
| Latitude, Longitude | | The sampling site. |
| Search extents (km) | 500 (+ global) | Distances from the site; GBIF occurrences are counted for each extent, and worldwide. |
| Search area | Circle | Circle: the occurrences within each extent of the site. Bounding box: the rectangle around that circle. |
| Filtering area | 500 | The extent whose counts the minimum occurrences apply to. |
| Minimum occurrences | 3 | Minimum GBIF occurrences in the filtering area to keep a species; guards against single stray records. |
| Name-match confidence | 95 (fixed) | GBIF name-match confidence required to accept a name. |

</details>

<details>
<summary><strong>EXTRA: BOLD CO1 proxy retrieval</strong></summary>

| Setting | Default | Meaning |
|---|---|---|
| COI-5P records only | Yes | Keep only BOLD records of the COI-5P barcode region. |
| Include records mined from GenBank | No | BOLD records copied from GenBank are left out by default. |
| Max sequences per species | No limit | Keep only the longest BOLD sequences of each species. |
| Redundancy identity (%) | 98 | Two BOLD sequences of a species at or above this identity are redundant, and only one is kept. |

</details>

<details>
<summary><strong>HYPO: NCBI BLAST validation of proxies</strong></summary>

| Setting | Default | Meaning |
|---|---|---|
| Max hits per query | 500 | As in MOL. |
| E-value exponent | 20 (1e-20) | E-value limit of the proxy BLAST. |
| Min hit identity (%) | 90 | Minimum identity of a proxy hit kept from the search. |
| Hits kept per score | 3 | Per proxy and hit species, the search keeps the hits no other beats on both identity and query cover (a short fragment cannot push out a full-length hit); this is how many hits with the same identity and cover are kept. |
| Query cover (%) | 50 | Minimum query cover of a proxy hit (looser than MOL, since proxy and target marker lengths differ). |
| Proxy identity (%) | 95 | A proxy hit on a MOL species of the sequence at or above this identity makes the species hypothetical. |
| HYPO check E-value exponent | 3 (1e-3) | E-value limit of the HYPO check. |

Each proxy is BLAST-searched against GenBank, restricted to the MOL species of its sequence, and its hits are filtered by identity and query cover. The HYPO check then BLAST-searches each sequence against GenBank restricted to its hypothetical species, looking for their target marker; a species whose marker it finds is flagged.

</details>

<details>
<summary><strong>Reproducibility</strong></summary>

Each cached run records the UTC date it queried each live database (NCBI core_nt, GBIF, BOLD, WoRMS) and the TRIDENT version, shown in the Results "Data sources" panel. These databases change over time, so the query date identifies which version of the source data an analysis used.

</details>

## Project structure

```text
src/trident/
  clients/     API clients (NCBI, BOLD, WoRMS, GBIF, FASTA)
  core/        Shared infrastructure (config, SQLite caching, HTTP, sequence selection)
  pipelines/   The five processing pipelines (mol, tax, geo, extra, hypo) + results
  ui/          Interactive Streamlit web app (one module per step)

tests/         pytest test suite
example/       Sample FASTA input + worked example notebook
```

## How to cite

A preprint describing TRIDENT is available on bioRxiv. Please cite:

> Haderlé R., Jung G., Riou M., Ung V., Jung J.-L. TRIDENT (Taxonomic Resolution and IDentification using Environmental dNa Traces): An Optimized Algorithm for Vertebrate Taxonomic Assignments in eDNA Metabarcoding, Integrating Molecular, Taxonomic, and Ecological Criteria. bioRxiv 2026.06.29.735257 (2026). https://doi.org/10.64898/2026.06.29.735257

## License

TRIDENT is released under the [PolyForm Strict License 1.0.0](https://polyformproject.org/licenses/strict/1.0.0) (see [`LICENSE`](LICENSE)). You may use and run it for any noncommercial purpose, including research and teaching. Redistributing, modifying, forking, or any commercial use requires the authors' written permission: contact <trident-contact@protonmail.com>.
</content>
</invoke>
