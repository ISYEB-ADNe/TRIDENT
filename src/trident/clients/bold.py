"""Client for the BOLD (Barcode of Life Data Systems) API."""

import json
import time

import pandas as pd
import requests
from loguru import logger

from trident.core.http import create_session, with_optional_session
from trident.core.utils import extract_specific_epithet, notify_progress


### BOLD API ###

BOLD_BASE_URL = "https://portal.boldsystems.org"
API_BASE_URL = f"{BOLD_BASE_URL}/api"


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
        query_id: BOLD query identifier from ``get_query_id``.
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


def _clean_records(
    records: list[dict], keep_only_COI5P: bool = True, keep_ncbi: bool = False
) -> pd.DataFrame:
    """Transform raw BOLD API records into a standardized DataFrame.

    Args:
        records: Raw record dicts from the BOLD API.
        keep_only_COI5P: If True, only retain records with marker_code ``"COI-5P"``.
        keep_ncbi: If True, retain records also found in NCBI.

    Returns:
        DataFrame with taxonomy, sequence, and URL columns.
    """
    cleaned_data = []
    for record in records:
        if keep_only_COI5P and record.get("marker_code") != "COI-5P":
            continue
        if not keep_ncbi and record.get("inst") == "Mined from GenBank, NCBI":
            continue
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
    return pd.DataFrame(cleaned_data)


def query_species(
    species_name: str,
    query_prefix: str = "tax:species:",
    session: requests.Session | None = None,
) -> list[dict] | None:
    """Query BOLD for barcode sequences of a species.

    Preprocesses the query, obtains a query ID, downloads results, and
    parses the JSONL response.

    Args:
        species_name: Species name (e.g. ``"Caranx caballus"``).
        query_prefix: Query type prefix.
        session: Optional requests session for connection reuse.

    Returns:
        List of record dicts if successful, empty list if no results,
        or None if any step fails.
    """
    query = f"{query_prefix}{species_name}"

    try:
        # Step 1: Preprocess — validate and resolve query against BOLD taxonomy
        preprocess_response = _preprocess_query(query, session=session)

        # Use the matched term from the preprocessor (handles synonyms/canonical forms)
        successful = preprocess_response.get("successful_terms", [])
        resolved_query = successful[0]["matched"] if successful else query

        # Step 2: Get query ID
        query_id = _get_query_id(resolved_query, session=session)
        if not query_id:
            logger.warning(f"Failed to obtain query ID for {species_name}")
            return None

        # Step 3: Download and parse results
        response_text = _download_results(query_id, session=session)
        data = _parse_jsonl(response_text)
        logger.debug(f"Retrieved {len(data)} records for {species_name}")
        return data

    except Exception as e:
        logger.error(f"Query failed for {species_name}: {e}")
        return None


def get_records_from_species_list(
    species_list: list[str],
    rate_limit_delay: float = 10.0,
    session: requests.Session | None = None,
    keep_only_COI5P: bool = True,
    keep_ncbi: bool = False,
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Retrieve BOLD records for a list of species.

    Creates a temporary session if none is provided. A delay is applied
    between queries to respect rate limits.

    Args:
        species_list: Species names to query.
        rate_limit_delay: Seconds to wait between queries (rate limiting).
        session: Optional requests session for connection reuse.
        keep_only_COI5P: If True, only retain COI-5P marker records.
        keep_ncbi: If True, retain records also found in NCBI.
        progress_handler: Optional progress tracker (dict or tqdm-like).
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Tuple of (bold_df, failed_species).
    """
    active_session = session or create_session(
        retries=5, backoff_factor=0.5, user_agent=user_agent
    )

    all_records = []
    failed_species = []

    for i, species in enumerate(species_list):
        try:
            logger.debug(f"Querying species {i + 1}/{len(species_list)}: {species}")
            records = query_species(species, session=active_session)

            if records:
                cleaned = _clean_records(
                    records, keep_only_COI5P=keep_only_COI5P, keep_ncbi=keep_ncbi
                )
                if not cleaned.empty:
                    all_records.append(cleaned)
            elif records is None:
                # None means the query errored (network/403/parse), as opposed
                # to [] which is a genuine empty result. Only errors count as
                # failures so the cache layer retries them instead of caching
                # them as empty. Keep query_species returning None on error.
                failed_species.append(species)

        except Exception as e:
            logger.error(f"Error processing {species}: {e}")
            failed_species.append(species)

        finally:
            # Ensure progress bar always moves
            notify_progress(progress_handler)

        # Rate limiting
        if i < len(species_list) - 1 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    # Only close if created internally
    if session is None:
        active_session.close()

    bold_df = (
        pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    )
    record_word = "record" if len(bold_df) == 1 else "records"
    logger.info(
        f"BOLD retrieval complete: {len(bold_df)} {record_word}, {len(failed_species)} failed"
    )
    return bold_df, failed_species
