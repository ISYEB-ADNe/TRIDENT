<p align="center">
  <img src="logo.png" alt="TRIDENT" width="160" />
</p>

<h1 align="center">TRIDENT</h1>

<p align="center">
  <strong>Taxonomic Resolution and IDentification using Environmental dNa Traces</strong>
</p>

<p align="center">
  <a href="https://trident-nxanjv7pk4z2fnkyulzvwt.streamlit.app/"><img src="https://img.shields.io/badge/live%20demo-Streamlit-22c55e?style=flat-square" alt="Live demo" /></a>
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

TRIDENT identifies species from environmental DNA (eDNA) barcode sequences. You upload a FASTA file of MOTU/ASV sequences, and TRIDENT cross-checks them against four biodiversity databases (NCBI, WoRMS, GBIF, BOLD) to produce a validated species list, ready for export.

It runs as a cross-platform web app (macOS, Windows, Linux), with no coding required once it is installed.

## What it does

TRIDENT takes your sequences through five steps. Each step has its own tab in the app, and you run them in order.

| Step | Tab | What happens | What you get |
|------|-----|--------------|--------------|
| 1 | **MOL** | Your sequences are BLAST-searched against NCBI, then filtered by identity and a barcoding-gap test | Species with a direct molecular match |
| 2 | **TAX** | Every genus found in MOL is expanded to all its accepted marine species via WoRMS | A wider candidate species list |
| 3 | **GEO** | Candidate species are checked against GBIF occurrence records inside your study area | Only species that plausibly occur where you sampled |
| 4 | **EXTRA** | For species with no direct NCBI match, CO1 proxy sequences are fetched from BOLD | Proxy sequences for indirect validation |
| 5 | **HYPO** | The BOLD proxy sequences are BLAST-validated against NCBI | Hypothetical species confirmed by proxy |

The **Results** tab merges the two validation paths into a final table:

- **MOL+GEO**: species with a direct NCBI match for your marker, confirmed geographically.
- **HYPO**: species validated indirectly through a CO1 proxy, confirmed geographically.

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

The app opens in your browser. Set your contact email in the **Settings** panel on first launch.

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

You can also use a `.env` file at the project root:

```env
CONTACT_EMAIL=your-email@example.com
```

</details>

## How to use the app

1. **Start Analysis**: Upload a FASTA file of your MOTU/ASV sequences (`.fasta`, `.fa`, `.fas`, `.txt`). Click **Process FASTA**, then **Proceed to MOL**.
2. **Run each step in order**: Open MOL, TAX, GEO, EXTRA, HYPO one at a time. Each tab has its own settings; review them, then run the step. A ✅ on the tab means the step is done.
3. **Set your study area in GEO**: Enter the latitude, longitude, and extent of your sampling region. Species not recorded there are dropped.
4. **Results**: Once HYPO is done, open the Results tab to view the final species list and download exports.

### Exports

- **Full results CSV**: every species with its validation path and warning flags.
- **GBIF-compatible CSV**: one row per MOTU/ASV; multi-species MOTUs are resolved to their lowest common taxon.
- **SQLite database (`.db`)**: the complete analysis. Re-upload it on the Start tab to restore your work later.

### Online demo

A hosted demo is live on Streamlit Cloud: <https://trident-nxanjv7pk4z2fnkyulzvwt.streamlit.app/>. It is meant for trying TRIDENT on small inputs and may be limited. In particular, the EXTRA step (BOLD) may be blocked on the hosted demo, since BOLD can refuse requests from shared cloud IPs; run TRIDENT locally (above) for full analyses.

## Example

`example/example.fasta` is a small bundled set of 4 MOTU/ASV sequences. Upload it in the app for a quick end-to-end demo of the pipeline.

<details>
<summary><strong>Run the example as a notebook (advanced)</strong></summary>

`example/example_pipeline.ipynb` runs the full pipeline (Load FASTA -> MOL -> TAX -> GEO -> EXTRA -> HYPO -> Results) step by step in Python on the same file. It calls the same `trident.pipelines` functions and SQLite cache as the app, and exposes every intermediate DataFrame for inspection or for integrating into your own scripts.

Open it in any Jupyter front-end (VS Code, JupyterLab, ...); the project ships the `ipykernel` and `ipywidgets` it needs. Before running, set your contact email and the study-area latitude / longitude / extents in the configuration cell near the top.

</details>

## Pipeline overview

Input MOTU/ASV sequences pass through five steps; the Results tab merges the two validation paths (MOL+GEO direct molecular match, HYPO indirect CO1-proxy match).

```text
MOL  (NCBI BLAST)  ->  TAX (WoRMS)  ->  GEO (GBIF)  ->  EXTRA (BOLD)  ->  HYPO (NCBI BLAST)
```

The thresholds applied at each step are detailed below. Defaults are those used by the app (`trident/ui/defaults.py`); every value is adjustable in the UI before running a step.

<details>
<summary><strong>MOL: NCBI BLAST + barcoding-gap filter</strong></summary>

| Parameter | Default | Meaning / rationale |
|---|---|---|
| Max hits per sequence | 500 | Upper bound on BLAST hits returned per query. |
| E-value exponent | 20 (1e-20) | Significance threshold for reporting a hit. |
| Query cover (%) | 90 | Minimum fraction of the query covered by the alignment; removes short partial matches. |
| Filter method | `barcoding_gap` | See below; falls back to `similarity` per sequence when no gap is found. |
| Gap size (%) | 2 | Minimum identity drop between consecutive hits to count as a barcoding gap. |
| Gap minimum top (%) | 97 | The top of the gap must be at or above this identity; prevents calling a gap among low-identity hits. |
| Low-identity threshold (%) | 95 | Flags species whose best hit is below this identity as a warning (not a filter, unless enforced). |

**Barcoding-gap method:** within each sequence's hits (sorted by descending identity), find the first drop >= gap size among hits at or above the gap minimum top, and keep only the hits above that gap. Sequences with no such gap fall back to the **similarity** method: keep all hits within `gap size` percent of the best hit. This keeps the tightest defensible cluster of hits per sequence.

</details>

<details>
<summary><strong>TAX: WoRMS genus expansion</strong></summary>

By default only marine-flagged species are kept ("Marine species only", an advanced option); unchecking it also includes non-marine WoRMS records.

</details>

<details>
<summary><strong>GEO: GBIF occurrence filter</strong></summary>

| Parameter | Default | Meaning / rationale |
|---|---|---|
| Name-match confidence | 95 | GBIF backbone match confidence required to accept a name. |
| Search extent(s) (km) | 500 (+ global) | Radius of the occurrence search box around the study point. |
| Minimum occurrences | 3 | Minimum GBIF records inside the extent to keep a species; guards against single stray records. |

</details>

<details>
<summary><strong>EXTRA: BOLD CO1 proxy retrieval</strong></summary>

| Parameter | Default | Meaning / rationale |
|---|---|---|
| COI-5P records only | Yes | Restrict BOLD records to the COI-5P barcode marker. |
| Include NCBI-mined records | No | Exclude BOLD records that were mined from GenBank. |
| Similarity threshold (%) | 98 | Sequences above this pairwise identity are treated as redundant and collapsed to one representative per cluster. |

</details>

<details>
<summary><strong>HYPO: NCBI BLAST validation of proxies</strong></summary>

| Parameter | Default | Meaning / rationale |
|---|---|---|
| Max hits per query | 500 | As in MOL. |
| E-value exponent (search) | 20 (1e-20) | Significance threshold for the proxy BLAST. |
| Identity cutoff (%) | 90 | Minimum identity to retain a proxy hit. |
| Top N hits | 3 | Keep only the top N hits per species after the identity cutoff. |
| Query cover (%) | 50 | Minimum query cover for the HYPO filter (looser than MOL because proxy/marker lengths differ). |
| Identity (%) | 95 | Minimum identity for the HYPO filter. |
| Check E-value exponent | 3 (1e-3) | Per-species confirmation BLAST, constrained to the candidate species via an Entrez organism query. |

BOLD proxy sequences are BLASTed against NCBI constrained to MOL-validated species, filtered by identity and query cover, then a final per-species check confirms the target marker exists in GenBank for each candidate.

</details>

<details>
<summary><strong>Reproducibility</strong></summary>

Each cached run records the UTC date it queried each live database (NCBI nt, GBIF, BOLD, WoRMS) and the TRIDENT version, shown in the Results "Data sources" panel. These databases change over time, so the query date identifies which version of the source data an analysis used.

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

A peer-reviewed paper describing TRIDENT is in review. Until it is published, please cite:

> Haderlé R., Jung G., Riou M., Ung V., Jung J.-L. TRIDENT (Taxonomic Resolution and IDentification using Environmental dNa Traces): An Optimized Algorithm for Vertebrate Taxonomic Assignments in eDNA Metabarcoding, Integrating Molecular, Taxonomic, and Ecological Criteria. Manuscript in review.

## License

TRIDENT is released under the [PolyForm Strict License 1.0.0](https://polyformproject.org/licenses/strict/1.0.0) (see [`LICENSE`](LICENSE)). You may use and run it for any noncommercial purpose, including research and teaching. Redistributing, modifying, forking, or any commercial use requires the authors' written permission: contact <trident-contact@protonmail.com>.
</content>
</invoke>
