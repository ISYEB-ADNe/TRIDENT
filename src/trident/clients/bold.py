"""Client for the BOLD (Barcode of Life Data Systems) API."""

import json
from collections.abc import Callable
from urllib.parse import quote

import pandas as pd
import requests
from loguru import logger

from trident.core import stop
from trident.core.http import create_session, with_optional_session
from trident.core.utils import (
    count_phrase,
    StepLog,
    extract_specific_epithet,
    name_list,
    notify_progress,
    pluralize,
)


### BOLD API ###

BOLD_BASE_URL = "https://portal.boldsystems.org"
API_BASE_URL = f"{BOLD_BASE_URL}/api"


def bold_search_url(species: str) -> str:
    """BOLD portal search for a species, to look at it by hand. The name is
    quoted: unquoted, the portal reads each word as a separate term and finds
    nothing."""
    return f"{BOLD_BASE_URL}/result?query={quote(f'"{species}"[tax]')}"


@with_optional_session(retries=5, backoff_factor=0.5)
def _preprocess_query(
    query: str,
    session: requests.Session | None = None,
) -> dict:
    """Validate and resolve a query against BOLD's controlled taxonomy.

    Args:
        query: BOLD query string (e.g. ``"tax:species:Caranx caballus"``).
        session: Optional requests session for connection reuse.

    Returns:
        Parsed JSON with ``successful_terms`` and ``failed_terms`` lists.
    """
    preprocessor_url = f"{API_BASE_URL}/query/preprocessor"
    params = {"query": query}

    resp = session.get(preprocessor_url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


@with_optional_session(retries=5, backoff_factor=0.5)
def _get_query_id(
    query: str,
    session: requests.Session | None = None,
) -> str | None:
    """Submit a query to BOLD and retrieve the query ID.

    Args:
        query: BOLD query string (e.g. ``"tax:species:Caranx caballus"``).
        session: Optional requests session for connection reuse.

    Returns:
        Query ID string, or None on failure.
    """
    query_url = f"{API_BASE_URL}/query"
    query_params = {"query": query, "extent": "full"}

    response = session.get(query_url, params=query_params, timeout=10)
    response.raise_for_status()
    query_id = response.json().get("query_id")

    if not query_id:
        logger.debug(f"No query_id received for query: {query}")
        return None

    logger.debug(f"Query ID obtained: {query_id}")
    return query_id


@with_optional_session(retries=5, backoff_factor=0.5)
def _download_results(
    query_id: str,
    session: requests.Session | None = None,
) -> str:
    """Download BOLD query results as JSONL text.

    Args:
        query_id: BOLD query identifier from ``_get_query_id``.
        session: Optional requests session for connection reuse.

    Returns:
        Raw JSONL response text.
    """
    download_url = f"{API_BASE_URL}/documents/{query_id}/download"
    download_params = {"format": "json"}

    response = session.get(download_url, params=download_params, timeout=10)
    response.raise_for_status()
    logger.debug(f"Download successful, length: {len(response.text)}")
    return response.text


def _parse_jsonl(response_text: str) -> list[dict]:
    """Parse JSONL (newline-delimited JSON) response into record dicts.

    Args:
        response_text: Raw JSONL text from BOLD API download endpoint.

    Returns:
        List of parsed record dicts. Invalid lines are skipped.
    """
    data = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSONL line: {e}")
    return data


# Why a queried species has no BOLD sequence, or SEQUENCES when it has some
# (bold_outcome column of the search).
SEQUENCES = "sequences"
NOT_IN_BOLD = "not in BOLD"
NO_RECORD = "no record"
OTHER_MARKERS = "only other markers"
GENBANK_ONLY = "only GenBank copies"
NO_SEQUENCE = "no DNA sequence"

# In the order the search filters them out.
OUTCOMES_WITHOUT_SEQUENCES = [
    NOT_IN_BOLD,
    NO_RECORD,
    OTHER_MARKERS,
    GENBANK_ONLY,
    NO_SEQUENCE,
]

GENBANK_INST = "Mined from GenBank, NCBI"

# Columns of _clean_records, also present when there are no records.
BOLD_RECORD_COLUMNS = [
    "seq_id",
    "dna_sequence",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specificEpithet",
    "scientificName",
    "taxonRank",
    "taxonID",
    "taxonID_db",
    "seq_url",
]

# Columns of the search: scientificName is the queried species (the name the
# EXTRA merge joins on), bold_scientificName the name BOLD files the records
# under, bold_matched_by the name that found them (SPECIES_NAME, or the source
# of another name of the species, e.g. "gbif"). A species without sequences
# has one row, with its bold_outcome, and the name BOLD knows it by if any.
BOLD_SEARCH_COLUMNS = BOLD_RECORD_COLUMNS + [
    "bold_scientificName",
    "bold_matched_by",
    "bold_outcome",
]

# bold_matched_by of a species found under its own name.
SPECIES_NAME = "species name"


def _filter_records(
    records: list[dict], keep_only_COI5P: bool = True, keep_ncbi: bool = False
) -> tuple[list[dict], str | None]:
    """The records the search keeps, and the outcome of the filter that left
    none (None when some are kept).

    Args:
        records: Raw record dicts from the BOLD API.
        keep_only_COI5P: If True, only retain records with marker_code ``"COI-5P"``.
        keep_ncbi: If True, keep the records BOLD mined from GenBank.
    """
    if not records:
        return [], NO_RECORD
    checks = []
    if keep_only_COI5P:
        checks.append((OTHER_MARKERS, lambda r: r.get("marker_code") == "COI-5P"))
    if not keep_ncbi:
        checks.append((GENBANK_ONLY, lambda r: r.get("inst") != GENBANK_INST))
    checks.append((NO_SEQUENCE, lambda r: bool(r.get("nuc"))))
    for outcome, keep in checks:
        records = [r for r in records if keep(r)]
        if not records:
            return [], outcome
    return records, None


def _clean_records(
    records: list[dict], keep_only_COI5P: bool = True, keep_ncbi: bool = False
) -> tuple[pd.DataFrame, str | None]:
    """The raw BOLD API records the search keeps (_filter_records) as a
    DataFrame of BOLD_RECORD_COLUMNS, and the outcome of the filter that left
    none (None when some are kept)."""
    kept, outcome = _filter_records(records, keep_only_COI5P, keep_ncbi)
    cleaned_data = []
    for record in kept:
        processid = record.get("processid")
        taxid = record.get("taxid")
        cleaned_record = {
            "seq_id": processid,
            "dna_sequence": record.get("nuc"),
            "kingdom": record.get("kingdom"),
            "phylum": record.get("phylum"),
            "class": record.get("class"),
            "order": record.get("order"),
            "family": record.get("family"),
            "genus": record.get("genus"),
            "specificEpithet": extract_specific_epithet(
                record.get("species"), record.get("genus")
            ),
            "scientificName": record.get("species"),
            "taxonRank": "species",
            "taxonID": str(taxid) if taxid is not None else None,
            "taxonID_db": "BOLD",
            "seq_url": f"{BOLD_BASE_URL}/record/{processid}",
        }
        cleaned_data.append(cleaned_record)
    return pd.DataFrame(cleaned_data, columns=BOLD_RECORD_COLUMNS), outcome


def query_species(
    species_name: str,
    query_prefix: str = "tax:species:",
    session: requests.Session | None = None,
) -> tuple[bool, list[dict]] | None:
    """Query BOLD for barcode sequences of a species.

    BOLD's preprocessor checks the name against BOLD's taxonomy (exact names
    only, no synonyms). A name it does not know comes back as a search of the
    record IDs (``ids:processid:...``), which never finds records: the name is
    reported as not in BOLD without querying. A name it rejects (in
    ``failed_terms``) is queried as written. An answer with neither is a
    failure.

    Args:
        species_name: Species name (e.g. ``"Caranx caballus"``).
        query_prefix: Query type prefix.
        session: Optional requests session for connection reuse.

    Returns:
        (in_bold, records): whether BOLD knows the name, and its records
        (empty when it has none). None if any request fails.
    """
    query = f"{query_prefix}{species_name}"

    try:
        preprocess_response = _preprocess_query(query, session=session)
        successful = preprocess_response.get("successful_terms", [])
        matched = successful[0]["matched"] if successful else ""
        if not successful and (failed := preprocess_response.get("failed_terms")):
            logger.warning(
                f"BOLD {species_name}: the preprocessor rejected the name "
                f"({failed}), queried as written "
                f"(check: {bold_search_url(species_name)})"
            )
            matched = query
        if matched.startswith("ids:"):
            return False, []
        if not matched.startswith("tax:"):
            raise ValueError(f"unexpected preprocessor answer: {preprocess_response}")

        query_id = _get_query_id(matched, session=session)
        if not query_id:
            logger.warning(
                f"BOLD {species_name} failed: BOLD gave no query ID "
                f"(check: {bold_search_url(species_name)})"
            )
            return None

        response_text = _download_results(query_id, session=session)
        data = _parse_jsonl(response_text)
        logger.debug(f"Retrieved {len(data)} records for {species_name}")
        return True, data

    except Exception as e:
        logger.warning(
            f"BOLD {species_name} failed: {e} (check: {bold_search_url(species_name)})"
        )
        return None


def _species_rows(
    species: str,
    name: str,
    matched_by: str,
    in_bold: bool,
    records: list[dict],
    keep_only_COI5P: bool,
    keep_ncbi: bool,
) -> pd.DataFrame:
    """Search rows of one queried species under one of its names: its usable
    sequences, or one row with the reason it has none."""
    if not in_bold:
        # Typed object like the record columns: the first save of the step
        # sets the SQLite column types, and a float taxonID column would read
        # later IDs back as floats.
        return pd.DataFrame(
            [{"scientificName": species, "bold_outcome": NOT_IN_BOLD}],
            columns=BOLD_SEARCH_COLUMNS,
            dtype=object,
        )
    cleaned, outcome = _clean_records(records, keep_only_COI5P, keep_ncbi)
    if outcome:
        return pd.DataFrame(
            [
                {
                    "scientificName": species,
                    "bold_scientificName": name,
                    "bold_matched_by": matched_by,
                    "bold_outcome": outcome,
                }
            ],
            columns=BOLD_SEARCH_COLUMNS,
            dtype=object,
        )
    return cleaned.assign(
        bold_scientificName=cleaned["scientificName"],
        bold_matched_by=matched_by,
        scientificName=species,
        bold_outcome=SEQUENCES,
    )[BOLD_SEARCH_COLUMNS]


def _search_species(
    species: str,
    other_names: list[tuple[str, str]],
    session: requests.Session,
    keep_only_COI5P: bool,
    keep_ncbi: bool,
    rate_limit_delay: float,
) -> pd.DataFrame | None:
    """Search rows of one species (see _species_rows). Without a usable
    sequence under the species name, its other names (name, source) are tried
    in turn until one gives some; a species none gives any keeps the outcome
    of the name that got furthest through the filters. None if a query fails.
    """
    best = None
    names = [(species, SPECIES_NAME), *other_names]
    for i, (name, matched_by) in enumerate(names):
        if i > 0 and rate_limit_delay > 0:
            stop.wait(rate_limit_delay)
        result = query_species(name, session=session)
        if result is None:
            return None
        rows = _species_rows(
            species, name, matched_by, *result, keep_only_COI5P, keep_ncbi
        )
        outcome = rows["bold_outcome"].iloc[0]
        if outcome == SEQUENCES:
            return rows
        if best is None or _progress(outcome) > _progress(best["bold_outcome"].iloc[0]):
            best = rows
    return best


def _progress(outcome: str) -> int:
    """How far through the filters a name got (OUTCOMES_WITHOUT_SEQUENCES
    order)."""
    return OUTCOMES_WITHOUT_SEQUENCES.index(outcome)


def under_other_name(bold_search_df: pd.DataFrame) -> pd.Series:
    """Which rows of the search come from another name of their species than
    its own (all False in a search saved before BOLD names were kept)."""
    if "bold_matched_by" not in bold_search_df.columns:
        return pd.Series(False, index=bold_search_df.index)
    matched_by = bold_search_df["bold_matched_by"]
    return matched_by.notna() & (matched_by != SPECIES_NAME)


def names_in_bold(bold_search_df: pd.DataFrame) -> pd.Series:
    """bold_scientificName of the rows from another name of their species
    (see under_other_name), empty (NaN) elsewhere."""
    other = under_other_name(bold_search_df)
    if not other.any():
        return pd.Series(pd.NA, index=bold_search_df.index, dtype=object)
    return bold_search_df["bold_scientificName"].where(other)


def _found_as(rows: pd.DataFrame) -> str:
    """The note "(as <name>, <source> name)" when the rows of one species come
    from another of its names, empty otherwise."""
    rows = rows[under_other_name(rows)]
    names = ", ".join(rows["bold_scientificName"].dropna().unique())
    if not names:
        return ""
    return f" (as {names}, {rows['bold_matched_by'].iloc[0].upper()} name)"


# Columns only the search needs: bold_outcome is always SEQUENCES on a
# sequence row, and bold_scientificName and bold_matched_by (the name BOLD
# files a record under, and which name found it) are for looking at the
# search itself.
SEARCH_ONLY_COLUMNS = ["bold_scientificName", "bold_matched_by", "bold_outcome"]


def with_sequences(bold_search_df: pd.DataFrame) -> pd.DataFrame:
    """The rows of the BOLD search that hold a sequence, without
    SEARCH_ONLY_COLUMNS. A search saved with no column at all (no species had
    records) gives an empty frame."""
    if "dna_sequence" not in bold_search_df.columns:
        return pd.DataFrame(columns=BOLD_RECORD_COLUMNS)
    return bold_search_df.dropna(subset=["dna_sequence"]).drop(
        columns=SEARCH_ONLY_COLUMNS, errors="ignore"
    )


def outcomes_by_species(bold_search_df: pd.DataFrame) -> pd.Series:
    """bold_outcome of each queried species, indexed by scientificName. A
    search saved before outcomes were recorded only has species with
    sequences."""
    if "scientificName" not in bold_search_df.columns:
        return pd.Series(dtype=object, name="bold_outcome")
    if "bold_outcome" not in bold_search_df.columns:
        bold_search_df = bold_search_df.assign(bold_outcome=SEQUENCES)
    return (
        bold_search_df.dropna(subset=["scientificName"])
        .drop_duplicates("scientificName")
        .set_index("scientificName")["bold_outcome"]
        .fillna(SEQUENCES)
    )


def get_records_from_species_list(
    species_list: list[str],
    other_names: dict[str, list[tuple[str, str]]] | None = None,
    rate_limit_delay: float = 10.0,
    session: requests.Session | None = None,
    keep_only_COI5P: bool = True,
    keep_ncbi: bool = False,
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    result_sink: Callable | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Retrieve BOLD records for a list of species.

    Creates a temporary session if none is provided. A delay is applied
    between queries to respect rate limits.

    Args:
        species_list: Species names to query.
        other_names: Other names of each species, as (name, source), tried
            in that order when the species name gives no usable sequence
            (see _search_species).
        rate_limit_delay: Seconds to wait between queries (rate limiting).
        session: Optional requests session for connection reuse.
        keep_only_COI5P: If True, only retain COI-5P marker records.
        keep_ncbi: If True, keep the records BOLD mined from GenBank.
        progress_handler: Optional progress tracker (a dict with "current", or
            tqdm-like).
        user_agent: User-Agent header for HTTP requests.
        result_sink: Called with the rows of each species and the species once
            its query is done (see save_to_db), unless it errored.

    Returns:
        Tuple of (bold_df, failed_species): bold_df has BOLD_SEARCH_COLUMNS,
        the usable sequences of each species, or one row with the reason it
        has none (bold_outcome).
    """
    active_session = session or create_session(
        retries=5, backoff_factor=0.5, user_agent=user_agent
    )

    all_records = []
    failed_species = []
    log = StepLog("BOLD search", len(species_list))

    for i, species in enumerate(species_list):
        try:
            with log.query(species):
                rows = _search_species(
                    species,
                    (other_names or {}).get(species, []),
                    active_session,
                    keep_only_COI5P,
                    keep_ncbi,
                    rate_limit_delay,
                )

            if rows is None:
                # None means the query errored (network/403/parse), as opposed
                # to a genuine empty result. Only errors count as failures so
                # the cache layer retries them instead of caching them as
                # empty. Keep query_species returning None on error.
                failed_species.append(species)
                log.item(species, "failed", notable=True)  # its error is logged above
            else:
                all_records.append(rows)
                if result_sink is not None:
                    result_sink(rows, [species])
                outcome = rows["bold_outcome"].iloc[0]
                found_as = _found_as(rows)
                if outcome == SEQUENCES:
                    n = len(rows)
                    log.item(
                        species,
                        f"{n} BOLD {pluralize(n, 'sequence')}{found_as}",
                        notable=bool(found_as),
                    )
                else:
                    log.item(
                        species, f"{outcome}{found_as}", notable=outcome == NOT_IN_BOLD
                    )

        except Exception as e:
            log.failed(species, e)
            failed_species.append(species)

        finally:
            # Ensure progress bar always moves
            notify_progress(progress_handler)

        # Rate limiting
        if i < len(species_list) - 1 and rate_limit_delay > 0:
            stop.wait(rate_limit_delay)

    # Only close if created internally
    if session is None:
        active_session.close()

    bold_df = (
        pd.concat(all_records, ignore_index=True)
        if all_records
        else pd.DataFrame(columns=BOLD_SEARCH_COLUMNS)
    )
    _log_outcomes(log, bold_df, failed_species)
    return bold_df, failed_species


def _log_outcomes(log: StepLog, bold_search_df: pd.DataFrame, failed: list[str]):
    """End the step's log: the sequences found and the species of each
    outcome without one, then one line naming them per outcome."""
    sequences = with_sequences(bold_search_df)
    n_species = sequences["scientificName"].nunique()
    summary = (
        f"{len(sequences)} BOLD {pluralize(len(sequences), 'sequence')} for "
        f"{n_species} {pluralize(n_species, 'species')}"
    )
    found_as = _found_under_other_names(bold_search_df)
    if found_as:
        summary += f" ({len(found_as)} under another name)"
    outcomes = outcomes_by_species(bold_search_df)
    counts = count_phrase(outcomes, "species", OUTCOMES_WITHOUT_SEQUENCES)
    log.done(f"{summary}, {counts}" if counts else summary, failed, "species")
    if found_as:
        logger.info(f"BOLD search, found under another name: {name_list(found_as)}")
    for outcome in OUTCOMES_WITHOUT_SEQUENCES:
        names = sorted(outcomes.index[outcomes == outcome])
        if names:
            logger.info(f"BOLD search, {outcome}: {name_list(names)}")


def _found_under_other_names(bold_search_df: pd.DataFrame) -> list[str]:
    """Each species whose sequences come from another of its names, with the
    note of _found_as."""
    sequences = bold_search_df[
        under_other_name(bold_search_df) & (bold_search_df["bold_outcome"] == SEQUENCES)
    ]
    found = (
        (species, _found_as(rows))
        for species, rows in sequences.groupby("scientificName", sort=True)
    )
    return [f"{species}{note}" for species, note in found if note]
