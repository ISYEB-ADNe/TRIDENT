"""NCBI BLAST client — submit queries, parse results, filter by barcoding gap."""

from collections.abc import Callable, Iterable
from functools import wraps

import pandas as pd
from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqRecord import SeqRecord
from joblib import Parallel, delayed
from loguru import logger

from .fasta import records_to_fasta_string
from trident.core import config
from trident.core.constants import (
    NCBI_GAP_SIZE,
    NCBI_GAP_MIN_TOP,
    NCBI_QUERY_COVER,
    NCBI_METHOD,
)
from trident.core.utils import extract_specific_epithet, notify_progress


### CREDENTIALS ###


def set_ncbi_email(email: str | None) -> None:
    """Set ``Entrez.email`` and ``NCBIWWW.email`` (process-level globals)."""
    if not email:
        return
    Entrez.email = email
    NCBIWWW.email = email
    logger.debug(f"NCBI contact email set: {email}")


def setup_ncbi_credentials() -> None:
    """Configure Biopython with the contact email from ``core.config``.

    Sets ``Entrez.email`` (E-utilities) and ``NCBIWWW.email`` (qblast).
    Both are optional but recommended — NCBI may throttle anonymous requests.
    """
    email = config.contact_email()

    if not email:
        logger.debug("CONTACT_EMAIL not set — NCBI requests will be anonymous")
    else:
        set_ncbi_email(email)
        logger.info(f"NCBI contact email set: {email}")


_ncbi_credentials_loaded = False


def reload_ncbi_credentials() -> None:
    """Force re-reading credentials from config (e.g. after Settings change)."""
    global _ncbi_credentials_loaded
    _ncbi_credentials_loaded = False
    setup_ncbi_credentials()
    _ncbi_credentials_loaded = True


def require_ncbi_auth(func: Callable) -> Callable:
    """Decorator that loads NCBI credentials once before the first call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _ncbi_credentials_loaded
        if not _ncbi_credentials_loaded:
            setup_ncbi_credentials()
            _ncbi_credentials_loaded = True
        return func(*args, **kwargs)

    return wrapper


### BATCH UTILITIES ###


def create_sequence_batches(
    sequences: list[SeqRecord],
    batch_size: int | None = None,
    n_batches: int | None = None,
) -> list[list[SeqRecord]]:
    """Split a sequence list into evenly-sized batches for parallel processing.

    Batches are balanced: any remainder is distributed one element at a time
    across the first batches, so sizes differ by at most 1.

    Provide either batch_size OR n_batches, not both. If neither is given,
    defaults to batch_size=10.

    Args:
        sequences: List of sequences to split.
        batch_size: Target size of each batch.
        n_batches: Desired number of batches.

    Returns:
        List of batches, each batch being a sublist of sequences.

    Raises:
        ValueError: If both batch_size and n_batches are provided.
    """
    if batch_size is not None and n_batches is not None:
        raise ValueError("Provide either batch_size or n_batches, not both")

    total = len(sequences)

    if n_batches is None:
        if batch_size is None:
            batch_size = 10
        n_batches = (total + batch_size - 1) // batch_size

    base_size = total // n_batches
    remainder = total % n_batches

    batches = []
    start = 0
    for i in range(n_batches):
        end = start + base_size + (1 if i < remainder else 0)
        batches.append(sequences[start:end])
        start = end

    batch_word = "batch" if len(batches) == 1 else "batches"
    logger.debug(f"Created {len(batches)} {batch_word}: {[len(b) for b in batches]}")
    return batches


### BLAST SUBMISSION ###


def parse_blast_records(records: list) -> pd.DataFrame:
    """Parse a list of BioPython BLAST records into a flat DataFrame.

    Each row corresponds to one HSP (High-scoring Segment Pair). Records with
    no alignments produce no rows.

    Args:
        records: List of BioPython BLAST record objects from NCBIXML.parse().

    Returns:
        DataFrame with columns: seq_id, query_length, hit_def, hit_id,
        hit_accession, score, bits, evalue, align_length, identities,
        gaps, query_start, query_end.
    """
    rows = []

    for record in records:
        seq_id = record.query  # Original FASTA sequence ID
        query_length = record.query_length

        for alignment in record.alignments:
            for hsp in alignment.hsps:
                rows.append(
                    {
                        "seq_id": seq_id,
                        "query_length": query_length,
                        "hit_def": alignment.hit_def,
                        "hit_id": alignment.hit_id,
                        "hit_accession": alignment.accession,
                        "score": hsp.score,
                        "bits": hsp.bits,
                        "evalue": hsp.expect,
                        "align_length": hsp.align_length,
                        "identities": hsp.identities,
                        "gaps": hsp.gaps,
                        "query_start": hsp.query_start,
                        "query_end": hsp.query_end,
                    }
                )

    return pd.DataFrame(rows)


### TAXONOMY ###


def to_binomial(org: str) -> str:
    """Truncate an organism name string to its binomial (genus + species epithet).

    Args:
        org: Raw organism name string, possibly with subspecies or strain suffixes.

    Returns:
        First two words of org joined by a space, or the single word if only one
        exists, or an empty string if org is empty or None.
    """
    parts = (org or "").split()
    return " ".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


@require_ncbi_auth
def fetch_standardized_taxonomy(
    accessions: Iterable[str], chunk_size: int = 200
) -> dict[str, str]:
    """Retrieve official scientific names for nucleotide accessions via NCBI Entrez.

    Fetches GenBank records in chunks and maps each accession (both with and
    without version suffix) to its organism name truncated to binomial form.
    Chunks that fail are skipped and logged at DEBUG level.

    Args:
        accessions: Accession strings to look up (e.g. "MT123456.1").
        chunk_size: Maximum number of IDs per Entrez request. Lower values
            reduce memory use and risk of request failure.

    Returns:
        Dict mapping each accession (versioned and unversioned) to its
        binomial scientific name.
    """
    accession_list = list(accessions)
    taxonomy_map = {}

    logger.debug(
        f"Fetching taxonomy for {len(accession_list)} accessions in chunks of {chunk_size}"
    )

    for i in range(0, len(accession_list), chunk_size):
        chunk = accession_list[i : i + chunk_size]
        try:
            with Entrez.efetch(db="nucleotide", id=chunk, retmode="xml") as handle:
                records = Entrez.read(handle)

            for record in records:
                acc_v = str(record.get("GBSeq_accession-version"))
                org = str(record.get("GBSeq_organism"))

                org = to_binomial(org)

                if acc_v and org:
                    taxonomy_map[acc_v] = org
                    taxonomy_map[acc_v.split(".")[0]] = org

        except Exception as e:
            logger.debug(f"Failed to fetch chunk {i // chunk_size + 1}: {e}")

    return taxonomy_map


@require_ncbi_auth
def submit_blast_batch(
    batch: list[SeqRecord],
    entrez_query: str | None = None,
    evalue: float = 1e-10,
    hitlist_size: int = 500,
    batch_id: int | None = None,
) -> pd.DataFrame | None:
    """Submit a BLASTn query for a single batch of sequences.

    Builds a multi-FASTA string from the batch, submits it to NCBI BLAST via
    qblast, parses the XML response, and returns a raw results DataFrame.
    Returns None on failure so the caller can handle partial results.

    Args:
        batch: List of SeqRecord objects to query.
        entrez_query: Optional Entrez query string to restrict the search space
            (e.g. '"Gadus morhua"[orgn]').
        evalue: E-value threshold for reporting hits.
        hitlist_size: Maximum number of hits to return per query sequence.
        batch_id: Identifier used for logging only.

    Returns:
        DataFrame of raw BLAST hits, or None if the query failed.
    """
    try:
        logger.debug(f"Processing batch {batch_id} with {len(batch)} sequences")

        multi_fasta = records_to_fasta_string(batch)

        handle = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=multi_fasta,
            expect=evalue,
            hitlist_size=hitlist_size,
            entrez_query=entrez_query,
        )

        records = list(NCBIXML.parse(handle))
        df = parse_blast_records(records)

        logger.debug(f"Batch {batch_id} completed successfully")
        return df

    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {e}")
        return None


def run_blast_batches(
    batches: list[list[SeqRecord]],
    n_threads: int = 10,
    evalue: float = 1e-10,
    hitlist_size: int = 500,
    entrez_query: str | None = None,
    progress_handler: dict | object | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Execute multiple NCBI BLAST batches in parallel.

    Runs up to n_threads batches concurrently using joblib threading. Progress
    is reported per batch via progress_handler.

    Args:
        batches: List of sequence batches, each a list of SeqRecord objects.
        n_threads: Maximum number of concurrent threads. Capped at len(batches).
        evalue: E-value significance threshold for BLAST hits.
        hitlist_size: Maximum number of hits per query sequence.
        entrez_query: Optional Entrez query to restrict the BLAST search space.
        progress_handler: Optional progress tracker. Accepts:
            - dict with a 'current' key (incremented by batch size) for
              Streamlit or notebook use.
            - Any object with an .update(n) method (e.g. tqdm) for CLI use.
            - None for silent execution.

    Returns:
        Tuple of (concatenated DataFrame of successful batch results,
        failed_ids). ``failed_ids`` lists the record ids in batches that errored
        (distinct from batches that simply returned no hits), so the caller can
        retry them instead of caching them as empty.
    """
    n_threads = min(n_threads, len(batches))
    n_threads_str = "thread" if n_threads == 1 else "threads"
    logger.debug(f"Starting parallel execution with {n_threads} {n_threads_str}")

    def worker_wrapper(batch, i):
        result = submit_blast_batch(
            batch,
            evalue=evalue,
            hitlist_size=hitlist_size,
            entrez_query=entrez_query,
            batch_id=i + 1,
        )
        notify_progress(progress_handler, len(batch))
        return result

    batch_results = Parallel(n_jobs=n_threads, backend="threading", verbose=0)(
        delayed(worker_wrapper)(batch, i) for i, batch in enumerate(batches)
    )

    successful = [df for df in batch_results if df is not None]
    failed_count = len(batch_results) - len(successful)

    # Record ids in errored batches, so the caller can retry them rather than
    # cache them as empty (a failed batch is not a "no hits" result).
    failed_ids = [
        rec.id
        for batch, res in zip(batches, batch_results)
        if res is None
        for rec in batch
    ]

    if failed_count > 0:
        logger.warning(f"{failed_count}/{len(batch_results)} batches failed")

    if not successful:
        logger.error("All NCBI batches failed — returning empty DataFrame")
        return pd.DataFrame(), failed_ids

    logger.debug(f"Batch success rate: {len(successful)}/{len(batch_results)}")
    return pd.concat(successful, ignore_index=True), failed_ids


### RESULT PROCESSING ###


def process_blast_results(
    df: pd.DataFrame,
    fasta_sequences: list[SeqRecord] | None = None,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Compute derived metrics and standardize taxonomy for raw BLAST results.

    Adds identity_percentage, query_cover, scientificName, genus,
    specificEpithet, hit_url, and dna_sequence columns. Scientific names are
    extracted from hit_def (first two words). Accessions whose name appears
    only once in the results are re-resolved via Entrez to correct malformed
    hit_def entries.

    Args:
        df: Raw BLAST DataFrame from parse_blast_records().
        fasta_sequences: Optional list of SeqRecord objects used to attach
            the original DNA sequence to each hit row.
        round_digits: Decimal places for identity_percentage and query_cover.

    Returns:
        Processed DataFrame with standardized columns, sorted by descending
        identity within each seq_id.
    """
    # No hits (e.g. every batch failed) — nothing to process.
    if df.empty:
        return df

    df["identity_percentage"] = ((df["identities"] / df["align_length"]) * 100).round(
        round_digits
    )
    df["query_cover"] = (
        ((df["query_end"] - df["query_start"] + 1) / df["query_length"]) * 100
    ).round(round_digits)

    # Most hit_def entries have the organism name at the start
    df["scientificName"] = df["hit_def"].str.split().str[:2].str.join(" ")
    # Names that occur exactly once are likely malformed — re-resolve via Entrez
    counts = df["scientificName"].value_counts(dropna=False)
    unique_names = counts[counts == 1].index
    accessions_with_unique_names = (
        df.loc[df["scientificName"].isin(unique_names), "hit_accession"]
        .dropna()
        .unique()
        .tolist()
    )

    if accessions_with_unique_names:
        logger.debug(
            f"Fetching standardized taxonomy for {len(accessions_with_unique_names)} "
            "accessions with unique organism names"
        )
        scientific_names_map = fetch_standardized_taxonomy(accessions_with_unique_names)

        df["scientificName"] = (
            df["hit_accession"].map(scientific_names_map).fillna(df["scientificName"])
        )

    # First token of the name. Vectorised and tolerant of an empty/NaN name
    # (a malformed hit_def): genus becomes NaN rather than raising IndexError.
    df["genus"] = df["scientificName"].str.split().str[0]
    df["specificEpithet"] = df.apply(
        lambda row: extract_specific_epithet(row["scientificName"], row["genus"]),
        axis=1,
    )

    df["hit_url"] = df["hit_id"].apply(
        lambda x: (
            f"https://www.ncbi.nlm.nih.gov/nucleotide/{x.split('|')[-2]}"
            if "|" in x
            else None
        )
    )

    df = (
        df.groupby("seq_id", sort=False)
        .apply(
            lambda x: x.sort_values(by="identity_percentage", ascending=False),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    if fasta_sequences:
        seq_dict = {str(sequence.id): str(sequence.seq) for sequence in fasta_sequences}
        df["dna_sequence"] = df["seq_id"].map(seq_dict)
    else:
        df["dna_sequence"] = None

    column_order = [
        "seq_id",
        "dna_sequence",
        "genus",
        "specificEpithet",
        "scientificName",
        "query_cover",
        "identity_percentage",
        "align_length",
        "identities",
        "gaps",
        "query_start",
        "query_end",
        "hit_def",
        "hit_url",
    ]

    return df[column_order]


### FILTERING ###


def compute_barcoding_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Add an identity_drop column: how much identity falls to the next-ranked hit.

    Within each seq_id group (assumed sorted by descending identity),
    identity_drop is strictly positive when the next hit has lower identity. The last
    row of each group gets NaN (no next hit to compare against).

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.

    Returns:
        Copy of df with an additional identity_drop column.
    """
    df = df.copy()
    df["identity_drop"] = df.groupby("seq_id")["identity_percentage"].diff(-1)
    return df


def filter_by_barcoding_gap(
    df: pd.DataFrame,
    gap_size: float = NCBI_GAP_SIZE,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Split sequences into those with a detectable barcoding gap and those without.

    For each seq_id, searches for the first drop ≥ gap_size in identity
    percentage among hits with identity ≥ gap_min_top. Sequences where such a
    gap is found are kept only above the gap; the rest are returned unfiltered
    for downstream similarity filtering.

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.
        gap_size: Minimum identity drop (percentage points) to qualify as a gap.
        gap_min_top: The top of the gap must be at or above this identity value.

    Returns:
        Tuple of:
        - gap_filtered_df: Rows for sequences where a gap was found (above-gap hits only).
        - no_gap_df: All rows for sequences where no gap was found.
        - query_filter_method: Dict mapping each seq_id to 'barcoding_gap' or 'similarity'.
    """
    queries_with_gaps = []
    queries_without_gaps = []
    query_filter_method = {}

    df = df.copy()
    df = compute_barcoding_gap(df)

    for seq_id, group in df.groupby("seq_id"):
        group = group.reset_index(drop=True)
        gap_rows = group[
            (group["identity_drop"] >= gap_size)
            & (group["identity_percentage"] >= gap_min_top)
        ]

        if not gap_rows.empty:
            first_gap_pos = gap_rows.index[0]
            queries_with_gaps.append(group.iloc[: first_gap_pos + 1])
            query_filter_method[seq_id] = "barcoding_gap"
        else:
            queries_without_gaps.append(group)
            query_filter_method[seq_id] = "similarity"

    gap_filtered_df = (
        pd.concat(queries_with_gaps).reset_index(drop=True)
        if queries_with_gaps
        else pd.DataFrame()
    )
    no_gap_df = (
        pd.concat(queries_without_gaps).reset_index(drop=True)
        if queries_without_gaps
        else pd.DataFrame()
    )

    return gap_filtered_df, no_gap_df, query_filter_method


def filter_by_similarity(
    df: pd.DataFrame, gap_size: float = NCBI_GAP_SIZE
) -> pd.DataFrame:
    """Keep only hits within gap_size percentage points of the top hit per sequence.

    Args:
        df: BLAST results DataFrame with seq_id and identity_percentage columns.
        gap_size: Maximum allowed drop from the best hit's identity percentage.

    Returns:
        Filtered DataFrame containing only hits within the similarity window.
    """
    max_per_query = df.groupby("seq_id")["identity_percentage"].transform("max")
    filtered_df = df[df["identity_percentage"] >= (max_per_query - gap_size)]
    return filtered_df


def filter_blast_results(
    df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    gap_size: float = NCBI_GAP_SIZE,
    method: str = NCBI_METHOD,
    gap_min_top: float = NCBI_GAP_MIN_TOP,
) -> pd.DataFrame:
    """Filter BLAST results by query coverage and identity gap method.

    First applies a query_cover threshold, then filters by either the barcoding
    gap method or the similarity window method. When method='barcoding_gap',
    sequences without a detectable gap fall back to similarity filtering.

    Args:
        df: Processed BLAST DataFrame from process_blast_results().
        query_cover: Minimum query coverage percentage to retain a hit.
        gap_size: Gap size threshold (percentage points) used by both methods.
        method: Either 'barcoding_gap' or 'similarity'.
        gap_min_top: Minimum identity for barcoding gap search. Only used when
            method='barcoding_gap'.

    Returns:
        Filtered DataFrame sorted by descending identity, then ascending species name.

    Raises:
        ValueError: If method is not 'barcoding_gap' or 'similarity'.
    """
    original_rows = len(df)
    df = df[df["query_cover"] >= query_cover]
    logger.info(
        f"Query cover filter (≥{query_cover}%): {original_rows} → {len(df)} rows"
    )

    if method == "barcoding_gap":
        df_gap, df_nogap, _ = filter_by_barcoding_gap(
            df, gap_size=gap_size, gap_min_top=gap_min_top
        )
        logger.debug(
            f"Gap analysis: {len(df_gap)} rows with gaps, {len(df_nogap)} rows need similarity method"
        )

        df_gap["filter_method"] = "barcoding_gap"

        if not df_nogap.empty:
            df_nogap = filter_by_similarity(df_nogap, gap_size=gap_size)
            df_nogap["filter_method"] = "similarity"

        dfs_to_concat = [d for d in [df_gap, df_nogap] if not d.empty]
        df = (
            pd.concat(dfs_to_concat).reset_index(drop=True)
            if dfs_to_concat
            else pd.DataFrame()
        )

        logger.info(
            f"Final result: {len(df)} rows after barcoding gap + similarity filtering"
        )

    elif method == "similarity":
        df = filter_by_similarity(df, gap_size=gap_size)
        df["filter_method"] = "similarity"
        logger.info(f"Similarity filtering: {len(df)} rows retained")

    else:
        raise ValueError(
            f"Unknown filtering method: {method!r}. Expected 'barcoding_gap' or 'similarity'."
        )

    df = df.sort_values(
        by=["identity_percentage", "scientificName"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


### GAP ANALYSIS ###


def check_species_across_gap(
    df: pd.DataFrame, gap_size: float = NCBI_GAP_SIZE
) -> dict[str, dict]:
    """Identify species that appear on both sides of the barcoding gap per sequence.

    For each seq_id, detects the first identity drop ≥ gap_size and checks
    whether any species has hits both above and below that gap, which would indicate
    an ambiguous taxonomic boundary.

    Args:
        df: Filtered BLAST DataFrame with seq_id, scientificName, and
            identity_percentage columns. Should contain a single seq_id or
            a pre-filtered subset.
        gap_size: Minimum identity drop (percentage points) to qualify as a gap.

    Returns:
        Dict mapping seq_id to a result dict with keys:
        - 'gap_range': (identity_above_gap, identity_below_gap) tuple, or
          (identity_above_gap, None) if the gap is at the last hit.
        - 'gap_size': Size of the gap in percentage points, or None.
        - 'species_details': Dict mapping species name to
          {'before_gap': [...], 'after_gap': [...]} identity lists.
          Only populated for species appearing on both sides of the gap.
        Seq IDs with no detectable gap are absent from the result.
    """
    results = {}

    # Compute gaps
    df_with_gaps = compute_barcoding_gap(df)

    for seq_id, group in df_with_gaps.groupby("seq_id"):
        group = group.reset_index(drop=True)
        gap_rows = group[group["identity_drop"] >= gap_size]

        if not gap_rows.empty:
            first_gap_pos = gap_rows.index[0]

            identity_before_gap = group.iloc[first_gap_pos]["identity_percentage"]
            identity_after_gap = (
                group.iloc[first_gap_pos + 1]["identity_percentage"]
                if first_gap_pos + 1 < len(group)
                else None
            )

            result = {
                "gap_range": (identity_before_gap, identity_after_gap)
                if identity_after_gap is not None
                else (identity_before_gap, None),
                "gap_size": (identity_before_gap - identity_after_gap)
                if identity_after_gap is not None
                else None,
                "species_details": {},
            }

            before_gap_data = group.iloc[: first_gap_pos + 1]
            after_gap_data = group.iloc[first_gap_pos + 1 :]

            species_before = set(before_gap_data["scientificName"].unique())
            species_after = set(after_gap_data["scientificName"].unique())
            problematic_species = species_before & species_after

            if problematic_species:
                species_details = {}

                for species in problematic_species:
                    species_data = group[group["scientificName"] == species]
                    before_identities = species_data[
                        species_data.index <= first_gap_pos
                    ]["identity_percentage"].tolist()
                    after_identities = species_data[species_data.index > first_gap_pos][
                        "identity_percentage"
                    ].tolist()

                    species_details[species] = {
                        "before_gap": before_identities,
                        "after_gap": after_identities,
                    }

                result["species_details"] = species_details

            results[seq_id] = result

    return results


def format_species_gap_details(species_details: dict[str, dict]) -> str:
    """Format the species_details dict from check_species_across_gap as a readable string.

    Args:
        species_details: Dict mapping species name to
            {'before_gap': [...], 'after_gap': [...]} identity lists, as
            returned by check_species_across_gap()[seq_id]['species_details'].

    Returns:
        Markdown-formatted string listing each problematic species with its
        identity percentages on each side of the gap.
    """
    lines = []
    for species, identities in species_details.items():
        before_str = ", ".join([f"{x:.2f}%" for x in identities["before_gap"]])
        after_str = ", ".join([f"{x:.2f}%" for x in identities["after_gap"]])
        lines.append(
            f"**{species}**\n  - Before gap: {before_str}\n  - After gap: {after_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _add_species_traces(
    fig,
    df_plot,
    y_col,
    filtered_species,
    hover_kwargs,
    marker_size=10,
    show_filtered_out=True,
):
    """Add one scatter trace per species, highlighting filtered ones.

    Args:
        fig: Plotly Figure to add traces to.
        df_plot: DataFrame with 'scientificName', 'x_position', and *y_col*.
        y_col: Column name for the y-axis values.
        filtered_species: Species names to highlight (or None).
        hover_kwargs: ``(species_name, species_df) -> dict`` returning extra
            kwargs for ``go.Scatter`` (e.g. customdata, hovertemplate).
        marker_size: Marker size for highlighted species (filtered-out species
            are drawn smaller).
        show_filtered_out: Unused; retained for call-site compatibility.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    color_map = {}
    if filtered_species:
        n = len(filtered_species)
        if n <= 10:
            palette = px.colors.qualitative.Plotly
        elif n <= 24:
            palette = px.colors.qualitative.Light24
        else:
            palette = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
        color_map = {
            sp: palette[i % len(palette)]
            for i, sp in enumerate(sorted(filtered_species))
        }

    filtered_out_added = False

    for species in df_plot["scientificName"].unique():
        sp_data = df_plot[df_plot["scientificName"] == species]

        if species in color_map:
            color, name, showlegend = color_map[species], species, True
            opacity, size = 0.9, marker_size
        else:
            color, name = "lightgrey", "Filtered out"
            showlegend = not filtered_out_added
            filtered_out_added = True
            opacity, size = 0.5, max(marker_size - 4, 4)

        fig.add_trace(
            go.Scatter(
                x=sp_data["x_position"],
                y=sp_data[y_col],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0.5, color="white"),
                ),
                name=name,
                showlegend=showlegend,
                legendgroup="filtered_out" if color == "lightgrey" else species,
                **hover_kwargs(species, sp_data),
            )
        )


def _apply_plot_layout(fig, title, **extra):
    """Apply common layout settings to a plot figure."""
    layout = {
        "hovermode": "closest",
        "showlegend": True,
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        "height": 500,
        **extra,
    }
    if title:
        layout["title"] = title
    fig.update_layout(**layout)


def plot_identity_percentage(
    df,
    threshold=2,
    filtered_species=None,
    title="Identity Percentage Distribution",
    marker_size=10,
    show_filtered_out=True,
):
    """Plot identity percentages, highlighting filtered species.

    Points at the same identity level are spread horizontally.
    Shows the barcoding gap location (if detected) or similarity range.

    Args:
        df: DataFrame with 'scientificName' and 'identity_percentage' columns.
        threshold: Minimum gap size or similarity range width.
        filtered_species: Set/list of species names to highlight.
        title: Plot title (empty string to hide).
        marker_size: Size of scatter markers for filtered species.
        show_filtered_out: Forwarded to ``_add_species_traces`` (currently unused).

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    df_plot = df.copy()
    df_plot["x_position"] = df_plot.groupby("identity_percentage").cumcount() * 1.5

    _add_species_traces(
        fig,
        df_plot,
        y_col="identity_percentage",
        filtered_species=filtered_species,
        marker_size=marker_size,
        show_filtered_out=show_filtered_out,
        hover_kwargs=lambda sp, _: {
            "hovertemplate": f"<b>{sp}</b><br>Identity: %{{y:.2f}}%<extra></extra>"
        },
    )

    # Find barcoding gap or fall back to similarity range
    by_identity = df.sort_values("identity_percentage", ascending=False).copy()
    by_identity["drop"] = by_identity["identity_percentage"].diff(-1)
    gap_rows = by_identity[by_identity["drop"] >= threshold]

    if not gap_rows.empty:
        first_gap_idx = gap_rows.index[0]
        bracket_top = by_identity.loc[first_gap_idx, "identity_percentage"]
        next_pos = by_identity.index.get_loc(first_gap_idx) + 1
        bracket_bottom = (
            by_identity.iloc[next_pos]["identity_percentage"]
            if next_pos < len(by_identity)
            else bracket_top
        )
        bracket_label = f"Gap<br>{bracket_top - bracket_bottom:.2f}%"
    else:
        bracket_top = df["identity_percentage"].max()
        bracket_bottom = bracket_top - threshold
        bracket_label = f"Range<br>{threshold}%"

    # Draw bracket (vertical line + horizontal caps + label)
    bracket_x = -1.5
    bracket_style = dict(color="red", width=2)
    fig.add_shape(
        type="line",
        x0=bracket_x,
        y0=bracket_top,
        x1=bracket_x,
        y1=bracket_bottom,
        line=bracket_style,
    )
    fig.add_shape(
        type="line",
        x0=bracket_x - 0.4,
        y0=bracket_top,
        x1=bracket_x + 0.4,
        y1=bracket_top,
        line=bracket_style,
    )
    fig.add_shape(
        type="line",
        x0=bracket_x - 0.4,
        y0=bracket_bottom,
        x1=bracket_x + 0.4,
        y1=bracket_bottom,
        line=bracket_style,
    )
    fig.add_annotation(
        x=bracket_x - 0.7,
        y=(bracket_top + bracket_bottom) / 2,
        text=bracket_label,
        showarrow=False,
        font=dict(color="red", size=10),
        xanchor="right",
        align="center",
    )

    _apply_plot_layout(fig, title, xaxis_title="Hits", yaxis_title="Identity (%)")
    return fig


def plot_barcoding_gap(
    df, threshold=2, filtered_species=None, title="Barcoding Gap Analysis"
):
    """Plot barcoding gap values, highlighting filtered species.

    Points at the same identity level share the same gap value.

    Args:
        df: DataFrame with 'scientificName' and 'identity_percentage' columns.
        threshold: Gap threshold displayed as a horizontal line.
        filtered_species: Set/list of species names to highlight.
        title: Plot title (empty string to hide).

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Compute gap between consecutive identity levels
    identity_levels = df["identity_percentage"].sort_values(ascending=False).unique()
    gap_by_identity = {
        identity_levels[i]: identity_levels[i] - identity_levels[i + 1]
        for i in range(len(identity_levels) - 1)
    }
    if len(identity_levels) > 0:
        gap_by_identity[identity_levels[-1]] = 0.0

    df_plot = df.copy()
    df_plot["gap"] = df_plot["identity_percentage"].map(gap_by_identity)
    df_plot["x_position"] = df_plot.groupby("gap").cumcount() * 1.5

    _add_species_traces(
        fig,
        df_plot,
        y_col="gap",
        filtered_species=filtered_species,
        hover_kwargs=lambda _, sp_data: {
            "customdata": sp_data[["scientificName", "identity_percentage"]].values,
            "hovertemplate": (
                "<b>%{customdata[0]}</b><br>"
                "Identity: %{customdata[1]:.2f}%<br>"
                "Gap: %{y:.2f}%<extra></extra>"
            ),
        },
    )

    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2)
    fig.add_annotation(
        x=df_plot["x_position"].max() * 0.95,
        y=threshold,
        text=f"Threshold: {threshold}%",
        showarrow=False,
        font=dict(color="red", size=11),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="red",
        borderwidth=1,
        xanchor="right",
        yanchor="middle",
    )

    _apply_plot_layout(
        fig, title, xaxis_title="Hits", yaxis_title="Identity Difference (%)"
    )
    return fig
