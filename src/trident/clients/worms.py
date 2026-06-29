"""Client for the WoRMS (World Register of Marine Species) REST API."""

import json
import time

import pandas as pd
import requests
from loguru import logger

from trident.core.http import create_session, with_optional_session
from trident.core.utils import (
    extract_specific_epithet,
    normalize_name,
    notify_progress,
)


### WORMS API ###

WORMS_API_BASE = "https://www.marinespecies.org/rest"


@with_optional_session(retries=5, backoff_factor=0.5)
def get_aphia_record(
    taxon_name: str,
    session: requests.Session | None = None,
) -> dict | None:
    """Retrieve the first WoRMS record for a taxon name.

    Args:
        taxon_name: Taxon name to look up.
        session: Optional requests session for connection reuse.

    Returns:
        Raw record dict if found, otherwise None.
    """
    url = f"{WORMS_API_BASE}/AphiaRecordsByName/{taxon_name}"
    params = {"like": "false", "marine_only": "false"}

    response = session.get(url, params=params, timeout=10)
    # WoRMS returns 404 for an unknown name (e.g. open nomenclature like
    # "Genus sp."); treat it as a valid "not found", not a request failure.
    if response.status_code == 404:
        logger.debug(f"No WoRMS record for taxon '{taxon_name}' (404).")
        return None
    response.raise_for_status()

    try:
        records = response.json()
    except json.JSONDecodeError:
        logger.debug(f"No JSON response for taxon '{taxon_name}' (204?)")
        return None

    if records:
        logger.debug(f"Found AphiaID {records[0]['AphiaID']} for taxon '{taxon_name}'")
        return records[0]
    else:
        logger.debug(f"No records found for taxon '{taxon_name}'")
        return None


def get_aphia_id(
    taxon_name: str,
    session: requests.Session | None = None,
) -> int | None:
    """Retrieve the AphiaID for a taxon name from WoRMS.

    Args:
        taxon_name: Taxon name to look up.
        session: Optional requests session for connection reuse.

    Returns:
        AphiaID integer if found, otherwise None.
    """
    record = get_aphia_record(taxon_name, session=session)
    return record["AphiaID"] if record else None


@with_optional_session(retries=5, backoff_factor=0.5)
def resolve_accepted_name(
    taxon_name: str,
    session: requests.Session | None = None,
) -> dict:
    """Resolve a taxon name to its WoRMS accepted name and AphiaID.

    Maps synonyms / misspellings / reassignments to the currently accepted
    taxon (e.g. 'Gadus ogac' -> 'Gadus macrocephalus'). Used to canonicalise
    MOL (NCBI) names before cross-source matching so synonyms are not treated
    as distinct species.

    Args:
        taxon_name: The name to resolve (typically an NCBI scientific name).
        session: Optional requests session for connection reuse.

    Returns:
        Dict with keys:
            input: the queried name.
            accepted_name: the WoRMS accepted name, or the input unchanged when
                not found in WoRMS.
            aphia_id: the accepted AphiaID (int), or None when not found.
            status: the WoRMS record status (e.g. 'accepted', 'unaccepted'),
                or None when not found.
            is_synonym: True when accepted_name differs from the input.

    Raises:
        Network/HTTP errors (other than 404, which means "not found") propagate
        so the caller can route them to a failure_sink and retry.
    """
    record = get_aphia_record(taxon_name, session=session)
    if record is None:
        return {
            "input": taxon_name,
            "accepted_name": taxon_name,
            "aphia_id": None,
            "status": None,
            "is_synonym": False,
        }

    accepted = record.get("valid_name") or record.get("scientificname") or taxon_name
    aphia_id = record.get("valid_AphiaID") or record.get("AphiaID")
    return {
        "input": taxon_name,
        "accepted_name": accepted,
        "aphia_id": int(aphia_id) if aphia_id is not None else None,
        "status": record.get("status"),
        "is_synonym": normalize_name(accepted) != normalize_name(taxon_name),
    }


@with_optional_session(retries=5, backoff_factor=0.5)
def get_children_by_aphia_id(
    aphia_id: int,
    session: requests.Session | None = None,
    timeout: int = 20,
    marine_only: bool = True,
) -> list[dict]:
    """Retrieve all child taxa under an AphiaID with pagination.

    Returns records at all ranks (species, subspecies, etc.). Use
    _build_taxonomy_dataframe downstream to filter to accepted species.

    Args:
        aphia_id: AphiaID of the parent taxon (e.g. a genus).
        session: Optional requests.Session.
        timeout: Request timeout in seconds.
        marine_only: If True, WoRMS returns only marine-flagged children.

    Returns:
        List of raw taxon records as dictionaries. Empty list if none found.
    """
    all_children = []
    offset = 1
    page = 1
    limit = 50  # WoRMS standard page size

    while True:
        url = f"{WORMS_API_BASE}/AphiaChildrenByAphiaID/{aphia_id}"
        # WoRMS API offset is 1-based
        params = {"marine_only": str(marine_only).lower(), "offset": offset}

        try:
            response = session.get(url, params=params, timeout=timeout)

            # 204 No Content is common for "end of results" in some REST APIs
            if response.status_code == 204:
                break

            response.raise_for_status()

            # Handle empty strings or whitespace-only responses
            if not response.text or not response.text.strip():
                break

            page_records = response.json()

            # WoRMS usually returns a list; if not a list, it's likely an error msg or empty
            if not isinstance(page_records, list) or not page_records:
                break

            all_children.extend(page_records)

            # If we got fewer than 50 results, we've reached the final page
            if len(page_records) < limit:
                break

            offset += limit
            page += 1

        except requests.exceptions.HTTPError as e:
            # WoRMS returns 404 if no children are found; this is a valid "empty" result
            if e.response is not None and e.response.status_code == 404:
                logger.debug(f"AphiaID {aphia_id} has no children (404).")
            else:
                logger.error(
                    f"HTTP error for AphiaID {aphia_id} at offset {offset}: {e}"
                )
            break

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Network error for AphiaID {aphia_id} at offset {offset}: {e}"
            )
            break

    if all_children:
        logger.debug(
            f"AphiaID {aphia_id}: retrieved {len(all_children)} children ({page} pages)"
        )

    return all_children


def clean_authorships(authorships: pd.Series) -> pd.Series:
    """Remove parentheses and extra whitespace from WoRMS authorship strings.

    Args:
        authorships: Series of raw authorship strings.

    Returns:
        Series with parentheses removed and whitespace normalised.
    """
    return (
        authorships.str.replace(r"[()]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _build_taxonomy_dataframe(
    raw_records: list[dict], accepted_only: bool
) -> tuple[pd.DataFrame, int, int]:
    """Filter raw WoRMS records and build a standardised taxonomy DataFrame.

    When accepted_only is True, keeps only accepted species, extracts
    standardised fields, renames columns to Darwin Core terms, and
    deduplicates by scientificName.

    Args:
        raw_records: Raw taxon records from the WoRMS API.
        accepted_only: If True, keep only accepted species and standardise columns.

    Returns:
        Tuple of (taxonomy_df, kept_count, skipped_count).
    """
    # 1. Filter and extract fields
    filtered = []
    skipped = 0

    for sp in raw_records:
        if not sp.get("valid_name") or (
            accepted_only
            and (sp.get("status") != "accepted" or sp.get("rank") != "Species")
        ):
            skipped += 1
            continue

        if accepted_only:
            record = {
                key: sp[key]
                for key in [
                    "kingdom",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "valid_name",
                    "valid_authority",
                    "rank",
                    "valid_AphiaID",
                    "url",
                ]
                if key in sp
            }
            if sp.get("rank") == "Species":
                specific_epithet = extract_specific_epithet(
                    sp.get("valid_name"), sp.get("genus")
                )
                if specific_epithet:
                    record["specificEpithet"] = specific_epithet
        else:
            record = sp

        filtered.append(record)

    kept = len(filtered)

    # 2. Build DataFrame
    df = pd.DataFrame(filtered)

    if accepted_only and not df.empty:
        df = df.rename(
            columns={
                "valid_name": "scientificName",
                "valid_authority": "scientificNameAuthorship",
                "valid_AphiaID": "taxonID",
                "rank": "taxonRank",
                "url": "taxonURL",
            }
        )
        df["scientificNameAuthorship"] = clean_authorships(
            df["scientificNameAuthorship"]
        )
        df["taxonID_db"] = "WORMS"

        desired_cols = [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "specificEpithet",
            "scientificName",
            "scientificNameAuthorship",
            "taxonRank",
            "taxonID",
            "taxonID_db",
            "taxonURL",
        ]
        existing_cols = [col for col in desired_cols if col in df.columns]
        df = df[existing_cols]
        df = df.drop_duplicates(subset=["scientificName"], keep="first")

    if "taxonID" in df.columns:
        df["taxonID"] = df["taxonID"].astype(str)

    return df, kept, skipped


def get_species_from_genus_list(
    genus_list: list[str],
    delay: float = 0.5,
    accepted_only: bool = True,
    progress_handler: dict | object | None = None,
    user_agent: str | None = None,
    marine_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Retrieve and consolidate WoRMS species for a list of genera.

    For each genus, resolves the AphiaID then fetches all child taxa with
    pagination. Results are filtered to accepted species by default and
    normalised into a standardised taxonomy DataFrame.

    Args:
        genus_list: List of genus names to query.
        delay: Delay in seconds between requests to avoid rate limiting.
        accepted_only: If True, keep only species with status 'accepted'.
        progress_handler: Optional progress tracker (dict with 'current' key,
            or object with .update() method).
        user_agent: User-Agent header for HTTP requests.
        marine_only: If True (default), WoRMS returns only marine-flagged
            children; uncheck to include non-marine records.

    Returns:
        Tuple of (taxonomy DataFrame, failed_genera). A genus is "failed" only
        when its query errored (network/HTTP); a genus with no AphiaID or no
        children is a valid empty, not a failure.
    """
    logger.debug(f"Starting to process {len(genus_list)} genera")
    all_records: list[dict] = []
    failed_genera: list[str] = []
    session = create_session(user_agent=user_agent)

    for genus in genus_list:
        try:
            # 1. Resolve AphiaID for the genus
            aphia_id = get_aphia_id(genus, session=session)
            if aphia_id is None:
                logger.warning(f"No AphiaID found for genus '{genus}'")
                continue

            # 2. Retrieve all children with internal pagination
            children = get_children_by_aphia_id(
                aphia_id, session=session, marine_only=marine_only
            )
            time.sleep(delay)

            if not children:
                logger.warning(f"No children found for genus '{genus}'")
                continue

            all_records.extend(children)
            record_word = "record" if len(children) == 1 else "records"
            logger.info(f"Retrieved {len(children)} {record_word} for genus '{genus}'")

        except Exception as e:
            # An error (not a valid empty) — track so the caller can retry it
            # instead of caching the genus as empty.
            logger.error(f"Unexpected error processing genus '{genus}': {e}")
            failed_genera.append(genus)
        finally:
            notify_progress(progress_handler)

    session.close()

    # 3. Filter and build taxonomy DataFrame
    df, kept, skipped = _build_taxonomy_dataframe(all_records, accepted_only)
    record_word = "record" if len(all_records) == 1 else "records"
    logger.info(
        f"Completed: {kept} species kept, {skipped} skipped from {len(all_records)} {record_word}"
    )
    return df, failed_genera
