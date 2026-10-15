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
    to_binomial,
)


### WORMS API ###

WORMS_API_BASE = "https://www.marinespecies.org/rest"


def _get_json(session: requests.Session, url: str, params: dict):
    """GET a WoRMS lookup and return its JSON body, or None when nothing matches.

    WoRMS answers an unknown name or id with 204 (no content) or 404; both are
    a valid "not found", not a request failure. Other HTTP errors raise.
    """
    response = session.get(url, params=params, timeout=10)
    if response.status_code in (204, 404):
        return None
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


@with_optional_session(retries=5, backoff_factor=0.5)
def get_aphia_records(
    taxon_name: str,
    session: requests.Session | None = None,
) -> list[dict]:
    """Retrieve every WoRMS record matching a taxon name exactly.

    A name can match several records: homonyms in other groups, and
    unaccepted records of the same name.

    Args:
        taxon_name: Taxon name to look up.
        session: Optional requests session for connection reuse.

    Returns:
        Raw record dicts in WoRMS order, empty if the name is unknown.
    """
    url = f"{WORMS_API_BASE}/AphiaRecordsByName/{taxon_name}"
    params = {"like": "false", "marine_only": "false"}

    records = _get_json(session, url, params)
    if not records:
        logger.debug(f"No WoRMS record for taxon '{taxon_name}'")
    return records or []


def get_aphia_record(
    taxon_name: str,
    session: requests.Session | None = None,
) -> dict | None:
    """Retrieve the first WoRMS record for a taxon name, or None if unknown."""
    records = get_aphia_records(taxon_name, session=session)
    if not records:
        return None
    logger.debug(f"Found AphiaID {records[0]['AphiaID']} for taxon '{taxon_name}'")
    return records[0]


def pick_genus_record(records: list[dict], marine_only: bool = True) -> dict | None:
    """Choose the genus to expand among the WoRMS records of a genus name.

    The accepted genus record is preferred: WoRMS can list an unaccepted
    homonym first (e.g. Delphinapterus, Myersina), whose children are not the
    genus meant. Among several accepted homonyms (e.g. Liparis, a fish and a
    plant), a marine one is preferred when marine_only is set; otherwise WoRMS
    order decides.

    Args:
        records: Records of one name, from get_aphia_records().
        marine_only: Whether the expansion is restricted to marine taxa.

    Returns:
        The chosen record, or None if no record has rank Genus.
    """
    genera = [r for r in records if r.get("rank") == "Genus"]
    accepted = [r for r in genera if r.get("status") == "accepted"]
    if marine_only:
        accepted = [r for r in accepted if r.get("isMarine")] or accepted
    if accepted:
        return accepted[0]
    return genera[0] if genera else None


def get_genus_aphia_id(
    genus: str,
    session: requests.Session | None = None,
    marine_only: bool = True,
) -> int | None:
    """Retrieve the AphiaID of the genus to expand (see pick_genus_record).

    The chosen record is expanded itself, even when unaccepted: WoRMS can list
    accepted species under a genus record marked unaccepted (e.g. Aphaniops,
    Turrum), and its accepted genus does not hold them.

    Returns:
        AphiaID integer if found, otherwise None.
    """
    record = pick_genus_record(
        get_aphia_records(genus, session=session), marine_only=marine_only
    )
    return record["AphiaID"] if record else None


@with_optional_session(retries=5, backoff_factor=0.5)
def get_aphia_record_by_ncbi_taxid(
    ncbi_taxid: int,
    session: requests.Session | None = None,
) -> dict | None:
    """Retrieve the WoRMS record linked to an NCBI taxonomy id, or None.

    WoRMS links only part of its records to NCBI, and a link can point to a
    subspecies or to an older record than the name lookup finds.
    """
    url = f"{WORMS_API_BASE}/AphiaRecordByExternalID/{int(ncbi_taxid)}"
    return _get_json(session, url, {"type": "ncbi"}) or None


# Ranks below species: a record of these ranks is resolved to its species.
_INFRASPECIFIC_RANKS = {"Subspecies", "Variety", "Forma", "Natio"}


def _species_of(record: dict) -> tuple[str, int] | None:
    """Accepted species name and AphiaID of an infraspecific record.

    An unaccepted record whose accepted name is a species resolves to that
    species. An accepted record resolves to its parent, the species of the
    subspecies. None otherwise.
    """
    valid_name = record.get("valid_name") or ""
    if record.get("status") != "accepted":
        # The accepted name is a species when it has two words besides a subgenus.
        words = [w for w in valid_name.split() if not w.startswith("(")]
        if record.get("valid_AphiaID") and len(words) == 2:
            return " ".join(words), int(record["valid_AphiaID"])
        return None
    if not record.get("parentNameUsageID"):
        return None
    name = valid_name or record.get("scientificname") or ""
    return to_binomial(name), int(record["parentNameUsageID"])


@with_optional_session(retries=5, backoff_factor=0.5)
def resolve_accepted_name(
    taxon_name: str,
    ncbi_taxid: int | None = None,
    session: requests.Session | None = None,
) -> dict:
    """Resolve a taxon name to its WoRMS accepted name and AphiaID.

    Maps synonyms / misspellings / reassignments to the currently accepted
    taxon (e.g. 'Gadus ogac' -> 'Gadus macrocephalus'). Used to canonicalise
    MOL (NCBI) names before cross-source matching so synonyms are not treated
    as distinct species.

    The name is looked up first. Only when WoRMS has no record for it, the
    NCBI taxid (if given) is looked up through the WoRMS link to NCBI, which
    finds names WoRMS knows under another genus (e.g. 'Microcarbo niger' ->
    'Phalacrocorax niger'). A taxid that leads to a subspecies resolves to its
    species.

    Args:
        taxon_name: The name to resolve (typically an NCBI scientific name).
        ncbi_taxid: Optional NCBI taxonomy id of the name, used when the name
            is not found.
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
            matched_by: 'name', 'ncbi_taxid', or None when not found.

    Raises:
        Network/HTTP errors (other than 404, which means "not found") propagate
        so the caller can route them to a failure_sink and retry.
    """
    not_found = {
        "input": taxon_name,
        "accepted_name": taxon_name,
        "aphia_id": None,
        "status": None,
        "is_synonym": False,
        "matched_by": None,
    }
    matched_by = "name"
    record = get_aphia_record(taxon_name, session=session)
    if record is None and ncbi_taxid is not None and not pd.isna(ncbi_taxid):
        record = get_aphia_record_by_ncbi_taxid(ncbi_taxid, session=session)
        matched_by = "ncbi_taxid"
    if record is None:
        return not_found

    accepted = record.get("valid_name") or record.get("scientificname") or taxon_name
    aphia_id = record.get("valid_AphiaID") or record.get("AphiaID")
    if matched_by == "ncbi_taxid" and record.get("rank") in _INFRASPECIFIC_RANKS:
        species = _species_of(record)
        if species is None:
            return not_found
        accepted, aphia_id = species
    return {
        "input": taxon_name,
        "accepted_name": accepted,
        "aphia_id": int(aphia_id) if aphia_id is not None else None,
        "status": record.get("status"),
        "is_synonym": normalize_name(accepted) != normalize_name(taxon_name),
        "matched_by": matched_by,
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

    Raises:
        Network/HTTP errors (other than 404, which means "no more children")
        propagate, on any page: a partial list must not pass for the complete
        one, so the caller reports the genus as failed and it is retried.
    """
    all_children = []
    offset = 1
    page = 1
    limit = 50  # WoRMS standard page size

    while True:
        url = f"{WORMS_API_BASE}/AphiaChildrenByAphiaID/{aphia_id}"
        # WoRMS API offset is 1-based
        params = {"marine_only": str(marine_only).lower(), "offset": offset}

        response = session.get(url, params=params, timeout=timeout)

        # 204 No Content and 404 both mean there are no (more) children.
        if response.status_code in (204, 404):
            if response.status_code == 404 and page == 1:
                logger.debug(f"AphiaID {aphia_id} has no children (404).")
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


# Output columns of the accepted-species taxonomy, also present when no species
# is found.
WORMS_TAXON_COLUMNS = [
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

        df = df.reindex(columns=WORMS_TAXON_COLUMNS)
        df = df.drop_duplicates(subset=["scientificName"], keep="first")
    elif accepted_only:
        df = pd.DataFrame(columns=WORMS_TAXON_COLUMNS)

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
            aphia_id = get_genus_aphia_id(
                genus, session=session, marine_only=marine_only
            )
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
