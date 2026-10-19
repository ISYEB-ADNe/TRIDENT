"""Client for the WoRMS (World Register of Marine Species) REST API."""

import json
from urllib.parse import quote

import pandas as pd
import requests
from loguru import logger

from trident.core import stop
from trident.core.http import create_session, with_optional_session
from trident.core.utils import (
    StepLog,
    pluralize,
    extract_specific_epithet,
    normalize_name,
    notify_progress,
    to_binomial,
)


### WORMS API ###

WORMS_API_BASE = "https://www.marinespecies.org/rest"


def worms_search_url(name: str) -> str:
    """WoRMS web search for a taxon name, for log lines to check it by hand."""
    return f"https://www.marinespecies.org/aphia.php?p=taxlist&tName={quote(name)}"


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


def rank_records(
    records: list[dict], rank: str | None = None, group: dict | None = None
) -> list[dict]:
    """The records of a name to choose from, in WoRMS order (see pick_record).

    The records are narrowed in three steps:

    1. rank: the records of that rank (any case), when given; none left
       means the name has no record of that rank;
    2. group: the records in the same group as the taxon meant, trying each
       of its ranks in turn (e.g. class, then kingdom), when given; skipped
       when no record is in the group;
    3. status: the accepted records; skipped when none is accepted.
    """
    if rank is not None:
        records = [r for r in records if (r.get("rank") or "").lower() == rank.lower()]
    for col, value in (group or {}).items():
        same_group = [r for r in records if pd.notna(value) and r.get(col) == value]
        if same_group:
            records = same_group
            break
    accepted = [r for r in records if r.get("status") == "accepted"]
    return accepted or records


def pick_record(
    records: list[dict], rank: str | None = None, group: dict | None = None
) -> dict | None:
    """Choose the taxon a name means among its WoRMS records.

    The accepted record is preferred, because WoRMS can list first an
    unaccepted record of the same name that belongs to another taxon, and
    following it leads to the wrong taxon. For example, the first record of
    the species Epinephelus undulosus points to Mycteroperca acutirostris,
    and the first record of the genus Delphinapterus is an unaccepted
    homonym that does not hold the beluga.

    A name can also be a taxon in several groups (e.g. the genus Liparis, a
    fish and a plant). When the group of the taxon meant is known, it is
    applied before the status, so a record of that group wins even if an
    accepted record of another group exists. Otherwise WoRMS order decides
    (see rank_records for the exact steps).

    Args:
        records: Records of one name, from get_aphia_records().
        rank: If given, only records of this rank are considered (any case).
        group: Optional ranks of the taxon meant, most specific first (e.g.
            {"class": ..., "kingdom": ...}); missing values are ignored.

    Returns:
        The chosen record, or None if there is no record (of that rank).
    """
    candidates = rank_records(records, rank, group)
    return candidates[0] if candidates else None


def get_aphia_record(
    taxon_name: str,
    session: requests.Session | None = None,
) -> dict | None:
    """Retrieve the WoRMS record of a taxon name (see pick_record), or None."""
    record = pick_record(get_aphia_records(taxon_name, session=session))
    if record:
        logger.debug(f"Found AphiaID {record['AphiaID']} for taxon '{taxon_name}'")
    return record


def get_genus_aphia_ids(
    genus: str,
    session: requests.Session | None = None,
) -> list[int]:
    """Retrieve the AphiaIDs of the genus records to expand for a genus name.

    Every accepted genus record is expanded: a name can be an accepted genus
    in several groups (e.g. Liparis, a fish and a plant), and the MOL species
    tell them apart later (see run_worms_merge in the TAX pipeline). Without an
    accepted record, the first genus record is expanded, even unaccepted:
    WoRMS can list accepted species under a genus record marked unaccepted
    (e.g. Aphaniops, Turrum), and its accepted genus does not hold them.

    Returns:
        AphiaIDs in WoRMS order, empty if the name has no genus record.
    """
    genera = rank_records(get_aphia_records(genus, session=session), "Genus")
    if not any(r.get("status") == "accepted" for r in genera):
        genera = genera[:1]
    return [r["AphiaID"] for r in genera]


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
INFRASPECIFIC_RANKS = {"Subspecies", "Variety", "Forma", "Natio"}


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
    if matched_by == "ncbi_taxid" and record.get("rank") in INFRASPECIFIC_RANKS:
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
    "genusID",  # AphiaID of the genus record the species was expanded from
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
                    "genusID",
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

    For each genus, resolves the AphiaIDs to expand (every accepted homonym,
    see get_genus_aphia_ids) then fetches all their child taxa with
    pagination; each species records its genus record as genusID. Results
    are filtered to accepted species by default and normalised into a
    standardised taxonomy DataFrame.

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
    all_records: list[dict] = []
    failed_genera: list[str] = []
    session = create_session(user_agent=user_agent)
    log = StepLog("WoRMS search", len(genus_list))

    for genus in genus_list:
        try:
            # 1. Resolve the AphiaIDs of the genus (several for homonyms)
            with log.query(genus):
                aphia_ids = get_genus_aphia_ids(genus, session=session)
            if not aphia_ids:
                log.item(
                    genus,
                    f"not in WoRMS (check: {worms_search_url(genus)})",
                    notable=True,
                )
                continue

            # 2. Retrieve all children with internal pagination. An error on
            # any homonym fails the whole genus, so no partial list is cached.
            children = []
            for aphia_id in aphia_ids:
                with log.query(genus):
                    children += [
                        {**child, "genusID": aphia_id}
                        for child in get_children_by_aphia_id(
                            aphia_id, session=session, marine_only=marine_only
                        )
                    ]
                stop.wait(delay)

            all_records.extend(children)
            if children:
                log.item(genus, f"{len(children)} {pluralize(len(children), 'record')}")
            else:
                marine = "marine " if marine_only else ""
                log.item(
                    genus,
                    f"no {marine}species in WoRMS (check: {worms_search_url(genus)})",
                    notable=True,
                )

        except Exception as e:
            # An error (not a valid empty) — track so the caller can retry it
            # instead of caching the genus as empty.
            log.failed(genus, f"{e} (check: {worms_search_url(genus)})")
            failed_genera.append(genus)
        finally:
            notify_progress(progress_handler)

    session.close()

    # 3. Filter and build taxonomy DataFrame
    df, kept, skipped = _build_taxonomy_dataframe(all_records, accepted_only)
    log.done(
        f"{kept} species kept, {skipped} skipped (not accepted species)",
        failed_genera,
        "genus",
    )
    return df, failed_genera
