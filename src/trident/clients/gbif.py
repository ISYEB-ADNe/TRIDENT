"""Client for the GBIF (Global Biodiversity Information Facility) API
and geographic bounding box calculations."""

import re
from collections.abc import Callable
from typing import Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from loguru import logger


from trident.core.constants import GBIF_CONFIDENCE
from trident.core import stop
from trident.core.http import create_session, request_label
from trident.core.utils import (
    StepLog,
    as_ids,
    ensure_columns,
    extract_specific_epithet,
    name_list,
    pluralize,
)


### GEOGRAPHIC CALCULATIONS ###

# Compile regex once
_DMS_PATTERN = re.compile(
    r'(\d+)[°ºd]?\s*\'?[\'′]?\s*(\d{1,2})\'?[\'′m]?\s*"?["″]?\s*(\d+(?:\.\d+)?)"?["″s]?\s*([NSEW])',
    re.IGNORECASE,
)


def _validate_coordinate_range(value: float, coord_type: Literal["lat", "lon"]) -> None:
    """Validate coordinate is within acceptable bounds."""
    if coord_type == "lat":
        if not -90 <= value <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {value}")
    elif coord_type == "lon":
        if not -180 <= value <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {value}")
    else:
        raise ValueError(f"coord_type must be 'lat' or 'lon', got '{coord_type}'")


def _validate_direction(direction: str, coord_type: Literal["lat", "lon"]) -> None:
    """Validate direction letter is appropriate for coordinate type."""
    valid_dirs = {"lat": {"N", "S"}, "lon": {"E", "W"}}
    if direction not in valid_dirs[coord_type]:
        raise ValueError(
            f"{coord_type.title()} direction must be "
            f"{valid_dirs[coord_type]}, got {direction}"
        )


def parse_coordinate(
    coord: str | float | int, coord_type: Literal["lat", "lon"] = "lat"
) -> float:
    """Parse a coordinate value from a number or string.

    Supports numeric values, decimal-degree strings, and symbolic DMS
    (e.g. ``45°30'15"N``, ``45d30m15sN``).

    Args:
        coord: Coordinate as a number, decimal-degree string, or DMS string.
        coord_type: ``"lat"`` or ``"lon"`` — used for range and direction validation.

    Returns:
        Decimal-degree float.
    """
    # 1. Numeric inputs (fast path)
    if isinstance(coord, (int, float)):
        value = float(coord)
        _validate_coordinate_range(value, coord_type)
        return value

    # 2. Validate string input
    if not isinstance(coord, str):
        raise TypeError(f"coord must be str, int, or float, got {type(coord).__name__}")

    coord = coord.strip()
    if not coord:
        raise ValueError("Empty coordinate string")

    # 3. Try Decimal Degrees string
    try:
        value = float(coord)
        _validate_coordinate_range(value, coord_type)
        return value
    except ValueError:
        pass

    # 4. Try Full DMS ONLY
    match = _DMS_PATTERN.search(coord)
    if not match:
        raise ValueError(
            f"❌ '{coord}' not valid.\n✅ Use: '45°30'15\"N' or '45d30m15sN'"
        )

    # 5. Parse DMS (always 4 groups)
    degrees_str, minutes_str, seconds_str, direction = match.groups()

    degrees = float(degrees_str)
    minutes = float(minutes_str)
    seconds = float(seconds_str)
    direction = direction.upper()

    # 6. Validate components
    if not (0 <= degrees <= 180):
        raise ValueError(f"Degrees must be 0-180, got {degrees}")
    if not (0 <= minutes < 60):
        raise ValueError(f"Minutes must be 0-59, got {minutes}")
    if not (0 <= seconds < 60):
        raise ValueError(f"Seconds must be 0-59.999, got {seconds}")

    _validate_direction(direction, coord_type)

    # 7. Convert to decimal degrees
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if direction in ("S", "W"):
        value = -value

    _validate_coordinate_range(value, coord_type)
    return value


def get_bounding_box(
    latitude: float | str,
    longitude: float | str,
    extent_km: float,
    return_circle: bool = False,
) -> tuple[float, float, float, float] | tuple[np.ndarray, np.ndarray]:
    """Generate bounding box coordinates or a circular boundary around a point.

    Uses spherical geometry to compute boundary points at ``extent_km`` radius
    from the center. Returns either the enclosing bounding box or the full
    circle of sampled points.

    Args:
        latitude: Center latitude (parsed via ``parse_coordinate``).
        longitude: Center longitude (parsed via ``parse_coordinate``).
        extent_km: Radius in kilometres.
        return_circle: If True, return arrays of circle lat/lon points instead
            of a bounding box.

    Returns:
        If ``return_circle`` is False: ``(min_lat, max_lat, min_lon, max_lon)``.
        If ``return_circle`` is True: ``(lats_array, lons_array)``.
    """
    # 1. Parse and Validate
    latitude = parse_coordinate(latitude, "lat")
    longitude = parse_coordinate(longitude, "lon")

    if extent_km <= 0:
        raise ValueError(f"Search extent must be positive, got {extent_km}")

    # 2. Spherical Geometry Constants
    R = 6371.0  # Earth radius in km
    angular_distance = extent_km / R
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)

    # 3. Sample points on the boundary circle
    angles = np.linspace(0, 2 * np.pi, 3600)

    # Vectorized latitude calculation
    lat_rad = np.arcsin(
        np.sin(latitude_rad) * np.cos(angular_distance)
        + np.cos(latitude_rad) * np.sin(angular_distance) * np.cos(angles)
    )

    # Vectorized longitude calculation
    lon_rad = longitude_rad + np.arctan2(
        np.sin(angles) * np.sin(angular_distance) * np.cos(latitude_rad),
        np.cos(angular_distance) - np.sin(latitude_rad) * np.sin(lat_rad),
    )

    lats = np.degrees(lat_rad)
    lons = np.degrees(lon_rad)

    # 4. Return format
    if return_circle:
        return lats, lons

    # The min/max of the circle points effectively creates the
    # 'smallest bounding box' that contains the circle.
    return float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max())


### GBIF API ###

# GBIF's current taxonomy, the Catalogue of Life eXtended Release (COL XR).
# Occurrence searches send it too: without it, GBIF reads the keys as keys of
# the GBIF Backbone, frozen since 2023.
COL_XR_CHECKLIST = "7ddf754f-d193-4cc9-b351-99906754a03b"

# Ranks of the classification columns a GBIF match carries
_TAXON_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


def worms_lsid(aphia_id: str | int) -> str:
    """WoRMS identifier of a record, as GBIF's matching service reads it."""
    return f"urn:lsid:marinespecies.org:taxname:{aphia_id}"


def _match_error(name: str, error: Exception) -> dict:
    """Match result of a name whose request failed (network, 403): retried on
    the next run, unlike a name GBIF does not know."""
    logger.warning(
        f"GBIF {name} failed: {str(error)[:100]} (check: {gbif_search_url(name)})"
    )
    return {
        "query": name,
        "confidence": 0,
        "matchType": "ERROR",
        "needs_review": True,
        "taxonID": None,
        "matched_by": None,
        "left_out": None,
    }


def _match_single_taxon_name(
    name: str,
    worms_id: str | None = None,
    session: requests.Session | None = None,
    user_agent: str | None = None,
) -> dict:
    """Match a single taxon to GBIF's current taxonomy (COL XR) via the v2
    species/match API.

    With a WoRMS ID, GBIF matches the ID first and checks it against the name.
    It matches the name alone when its ID index lacks the ID (WoRMS records
    newer than the index). An ID match can return another name for the same
    species, e.g. Larus saundersi gives Chroicocephalus saundersi.

    Args:
        name: Species or taxon name to match.
        worms_id: WoRMS AphiaID of the name, if known.
        session: Optional requests session for connection reuse.
        user_agent: User-Agent header, used only when creating an own session.

    Returns:
        Dict with the match type and confidence, ``matched_by`` ("WoRMS ID" or
        "name"), the GBIF species counted for the name (``taxonID``,
        ``speciesID``, ``canonicalName``, ``note``, ``left_out``: see
        _counted_species), its classification, and the 'needs_review' flag
        for low-confidence matches.
    """
    params = {"scientificName": name, "checklistKey": COL_XR_CHECKLIST}
    if worms_id:
        params["scientificNameID"] = worms_lsid(worms_id)

    own_session = session is None
    try:
        stop.check_stop()  # a stopped step builds no session for nothing
        if own_session:
            session = create_session(user_agent=user_agent)
        response = session.get(
            "https://api.gbif.org/v2/species/match", params=params, timeout=15
        )
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage") or {}
        accepted = data.get("acceptedUsage") or usage
        diagnostics = data.get("diagnostics") or {}
        ranks = {
            str(taxon.get("rank", "")).lower(): taxon.get("name")
            for taxon in data.get("classification") or []
        }
        match_type = diagnostics.get("matchType", "NONE")
        confidence = diagnostics.get("confidence", 0)
        matched_by = "WoRMS ID" if diagnostics.get("matchedID") else "name"

        return {
            "query": name,
            "worms_id": worms_id,
            "confidence": confidence,
            "matchType": match_type,
            "status": usage.get("status"),
            "rank": usage.get("rank"),
            "matched_by": matched_by,
            **_counted_species(
                name,
                usage,
                accepted,
                matched_by,
                match_type,
                data.get("classification"),
            ),
            **{rank: ranks.get(rank) for rank in _TAXON_RANKS},
            "needs_review": confidence < GBIF_CONFIDENCE or match_type == "NONE",
        }

    except Exception as e:
        return _match_error(name, e)

    finally:
        if own_session and session is not None:
            session.close()


def gbif_search_url(name: str) -> str:
    """GBIF web search for a taxon name, for log lines to check it by hand."""
    return f"https://www.gbif.org/species/search?q={quote(name)}"


# Columns of match_taxon_names for an empty names list. Every match result,
# success or error, already carries them; matched names carry more API fields.
GBIF_MATCH_COLUMNS = ["query", "confidence", "matchType", "needs_review", "taxonID"]


def match_taxon_names(
    names: list[str],
    worms_ids: dict[str, str] | None = None,
    n_jobs: int = 10,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Match species names to GBIF's current taxonomy (COL XR), by their WoRMS
    ID when known, using parallel requests.

    Args:
        names: List of scientific names to match.
        worms_ids: WoRMS AphiaID of each name, matched first.
        n_jobs: Number of parallel threads.
        user_agent: User-Agent header for HTTP requests.

    Each call creates its own session: requests.Session is not thread-safe, so
    sharing one across the threading backend can corrupt the pool state.

    Returns:
        DataFrame with one row per name containing GBIF match metadata.
    """
    worms_ids = worms_ids or {}
    if len(names) == 0:
        logger.debug("Empty names list")
        return pd.DataFrame(columns=GBIF_MATCH_COLUMNS)

    log = StepLog("GBIF name match", len(names))

    def match(name):
        with log.query(name):
            result = _match_single_taxon_name(
                name, worms_ids.get(name), user_agent=user_agent
            )
        match_type = str(result.get("matchType", "")).lower()
        check = f"check: {gbif_search_url(name)}"
        if match_type == "error":
            log.item(name, "failed", notable=True)  # its error is logged by the match
        elif match_type == "none":
            log.item(name, f"not in GBIF ({check})", notable=True)
        elif result["needs_review"]:
            confidence = result.get("confidence")
            log.item(
                name,
                f"needs review ({match_type} match, confidence {confidence}; {check})",
                notable=True,
            )
        elif result["taxonID"] is None:  # e.g. matched to its genus only
            rank = str(result.get("rank", "")).lower()
            log.item(
                name, f"no species in GBIF, matched as {rank} ({check})", notable=True
            )
        elif reason := result.get("left_out"):
            log.item(name, f"left out, {reason} ({check})", notable=True)
        else:
            by = result.get("matched_by")
            note = result.get("note")
            log.item(
                name,
                f"{match_type} match"
                + (f" by {by}" if by else "")
                + (f", {note}" if note else ""),
                notable=bool(note),
            )
        return result

    results = stop.run_threads(match, [(name,) for name in names], n_jobs)

    results_df = pd.DataFrame(results)
    errored = results_df["matchType"] == "ERROR"
    matched = (
        ~results_df["needs_review"]
        & results_df["taxonID"].notna()
        & results_df["left_out"].isna()
    )
    n_failed = int(errored.sum())
    n_other = len(names) - int(matched.sum()) - n_failed
    failed = f", {n_failed} failed" if n_failed else ""
    n_by_name = int((matched & (results_df["matched_by"] == "name")).sum())
    by_name = f" ({n_by_name} by name, not by WoRMS ID)" if n_by_name else ""
    # The failed names are listed by the GBIF search, with its failed counts.
    log.done(
        f"{int(matched.sum())}/{len(names)} species matched{by_name}, {n_other} not "
        f"matched to a species or left out{failed}"
    )

    return results_df


# Standardised taxonomy columns emitted by filter_taxon_matches. Shared by the
# empty-result returns and the column-ensure step so an empty result keeps the
# same schema as a populated one (downstream merges stay stable).
_GBIF_TAXON_COLS = [
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
    "taxonURL",
    "gbif_scientificName",
    "gbif_matched_by",
    "aphia_id",
    "gbif_note",
    "gbif_speciesID",
]

# Name match types that mean the queried species: the same name, or a variant
# of it (e.g. a gender ending, Monocentris neozelanica -> neozelanicus). A fuzzy
# match may be another species.
_SAME_NAME_MATCHES = ["EXACT", "VARIANT"]


# Latin gender endings of a species epithet (flexuosus, flexuosa, flexuosum;
# niger, nigra, nigrum), longest first
_GENDER_ENDINGS = ("rum", "um", "us", "is", "ra", "er", "a", "e")


def _epithet_stem(name: str | None) -> str | None:
    """The last epithet of a name (the species epithet of a binomial, the
    subspecies one of a trinomial) without its gender ending."""
    parts = (name or "").lower().split()
    if len(parts) < 2:
        return None
    epithet = parts[-1]
    for ending in _GENDER_ENDINGS:
        if epithet.endswith(ending) and len(epithet) > len(ending) + 1:
            return epithet[: -len(ending)]
    return epithet


def _counted_species(
    query: str,
    usage: dict,
    accepted: dict,
    matched_by: str,
    match_type: str,
    classification: list[dict] | None = None,
) -> dict:
    """The GBIF species whose occurrences count for the queried name.

    A name match must give the same name or a variant of it: a fuzzy one may
    be another species and is left out. WoRMS is the reference. A name GBIF
    holds as a synonym of another species (Lestrolepis philippina, a species of
    its own in WoRMS, a synonym of L. japonica in GBIF) is counted under its
    own GBIF key, which holds the records published under that name; the
    accepted key would add all those of L. japonica. A synonym that only moves
    the species to another genus (same epithet, up to its gender ending) is the
    same species, counted under the accepted key, which also holds the records
    under the new genus.

    A WoRMS species GBIF holds as a subspecies follows the same rule. Its
    nominate subspecies (Larus atricilla -> Leucophaeus atricilla atricilla) is
    the species itself, counted under the GBIF species, which also holds the
    records named at species level only. Another subspecies (Larus thayeri ->
    Larus glaucoides thayeri) is counted under its own key: the species key
    would add all the records of L. glaucoides.

    Args:
        query: The queried (TAX) name.
        usage: GBIF's usage of the match (``usage`` of the v2 answer).
        accepted: Its accepted usage (``acceptedUsage``, or the usage itself).
        matched_by: "WoRMS ID" or "name".
        match_type: GBIF's match type.
        classification: GBIF's classification of the match, from the top
            rank down to the match.

    Returns:
        Dict with ``taxonID`` (the key whose occurrences are counted; None
        when GBIF matched a higher rank only), ``speciesID`` (the key of
        GBIF's species for the name: the accepted species of a synonym, the
        species of a subspecies), ``canonicalName`` (GBIF's name
        of the species, a subspecies name when GBIF holds it as one),
        ``note`` (how a synonym or subspecies is counted, else None) and
        ``left_out`` (why the match is not counted, else None).
        A nominate subspecies keeps its own name: its name searches in other
        sources (BOLD, the HYPO check) then miss the species' other
        subspecies.
    """
    counted = {
        "taxonID": accepted.get("key"),
        "speciesID": None,
        "canonicalName": accepted.get("canonicalName"),
        "note": None,
        "left_out": None,
    }
    species = accepted
    if accepted.get("rank") != "SPECIES":
        species = next(
            (taxon for taxon in classification or [] if taxon.get("rank") == "SPECIES"),
            None,
        )
        if species is None:  # matched to a genus or higher
            return counted | {"taxonID": None}
    counted["speciesID"] = species.get("key")
    if matched_by != "WoRMS ID" and match_type not in _SAME_NAME_MATCHES:
        return counted | {
            "left_out": f"GBIF found only {accepted.get('canonicalName')}, by a "
            f"{match_type.lower()} name match"
        }
    status = usage.get("status") or ""
    synonym = "SYNONYM" in status or status == "MISAPPLIED"
    if synonym and _epithet_stem(query) != _epithet_stem(accepted.get("canonicalName")):
        return counted | {
            "taxonID": usage.get("key"),
            "canonicalName": usage.get("canonicalName"),
            "note": "counted under its own name, a synonym in GBIF of "
            f"{accepted.get('canonicalName')}",
        }
    if species is not accepted:
        # The nominate subspecies repeats the species epithet
        nominate = _epithet_stem(accepted.get("canonicalName")) == _epithet_stem(
            species.get("name")
        )
        if nominate and _epithet_stem(query) == _epithet_stem(species.get("name")):
            return counted | {
                "taxonID": species.get("key"),
                "note": f"GBIF subspecies {accepted.get('canonicalName')}, counted "
                f"with all records of {species.get('name')} (its other subspecies "
                "included)",
            }
        return counted | {
            "note": f"GBIF subspecies {accepted.get('canonicalName')}, counted with "
            "its own records only"
        }
    return counted


def filter_taxon_matches(
    matches_df: pd.DataFrame,
    confidence_threshold: int = GBIF_CONFIDENCE,
) -> pd.DataFrame:
    """Keep the GBIF matches whose occurrences are those of the queried species.

    A match is kept when GBIF found a species with enough confidence, through
    the WoRMS ID or by name, unless the match has a ``left_out`` reason (the
    match log names them). A synonym or subspecies in GBIF counts under the
    key _counted_species chose. Several queried names can share one GBIF
    species (e.g. a species WoRMS lists twice): each name keeps its row, with
    the same occurrence counts.

    Args:
        matches_df: DataFrame from match_taxon_names()
        confidence_threshold: Min confidence score

    Returns:
        One row per kept name: ``scientificName`` is the queried name (the TAX
        name the GEO merge joins on), ``gbif_scientificName`` GBIF's name of
        the species (a subspecies name when GBIF holds it as one; taxonID is
        then the key its occurrences are counted under), ``gbif_matched_by``
        "WoRMS ID" or "name", ``aphia_id`` the WoRMS ID it was queried with,
        ``gbif_note`` how a synonym or subspecies in GBIF is counted,
        ``gbif_speciesID`` GBIF's species for the name (see _counted_species).
        The classification columns and specificEpithet are those of GBIF's
        species. Always carries the standardised taxonomy columns, even when
        empty.
    """
    if matches_df.empty:
        logger.debug("Empty DataFrame provided to filter_taxon_matches")
        return pd.DataFrame(columns=_GBIF_TAXON_COLS)

    # Columns a match carries only on success: absent when every request failed
    matches_df = ensure_columns(
        matches_df.copy(),
        [
            "canonicalName",
            "speciesID",
            "matched_by",
            "worms_id",
            "note",
            "left_out",
            "rank",
            *_TAXON_RANKS,
        ],
    )
    species = (
        pd.to_numeric(matches_df["confidence"], errors="coerce") >= confidence_threshold
    ) & matches_df["taxonID"].notna()
    kept = matches_df[species & matches_df["left_out"].isna()]

    if kept.empty:
        logger.debug("No matches passed quality filters")
        return pd.DataFrame(columns=_GBIF_TAXON_COLS)

    filtered_df = kept[_TAXON_RANKS].reset_index(drop=True)
    filtered_df["scientificName"] = kept["query"].to_numpy()
    filtered_df["taxonRank"] = kept["rank"].to_numpy()
    filtered_df["taxonID"] = kept["taxonID"].astype(str).to_numpy()
    filtered_df["taxonID_db"] = "gbif"
    filtered_df["taxonURL"] = "https://www.gbif.org/taxon/" + filtered_df["taxonID"]
    filtered_df["gbif_scientificName"] = kept["canonicalName"].to_numpy()
    filtered_df["gbif_matched_by"] = kept["matched_by"].to_numpy()
    filtered_df["aphia_id"] = as_ids(kept["worms_id"]).to_numpy()
    filtered_df["gbif_note"] = kept["note"].to_numpy()
    filtered_df["gbif_speciesID"] = kept["speciesID"].astype("string").to_numpy()
    filtered_df["specificEpithet"] = filtered_df.apply(
        lambda row: extract_specific_epithet(row["gbif_scientificName"], row["genus"]),
        axis=1,
    )
    filtered_df = filtered_df[_GBIF_TAXON_COLS]

    shared = filtered_df[filtered_df.duplicated("taxonID", keep=False)]
    same_species_names = [
        " = ".join(names)
        for names in shared.groupby("taxonID")["scientificName"].agg(list)
    ]
    shared_message = (
        f"; {len(same_species_names)} GBIF "
        f"{pluralize(len(same_species_names), 'species')} under several names, "
        f"counted for each ({name_list(same_species_names)})"
        if same_species_names
        else ""
    )
    logger.info(
        f"GBIF search: {len(filtered_df)}/{len(matches_df)} species kept for the "
        f"occurrence counts{shared_message}"
    )

    return filtered_df


SEARCH_AREAS = ["circle", "box"]


def area_params(
    latitude: float | str | None,
    longitude: float | str | None,
    extent: int | float | str,
    area: str = "circle",
) -> dict:
    """Occurrence search filter for one search area.

    Worldwide for the "global" extent. Otherwise the occurrences within
    *extent* km of the site ("circle", GBIF's geoDistance), or within the
    bounding box of that circle ("box").

    Args:
        latitude: Center latitude.
        longitude: Center longitude.
        extent: Search radius in km, or ``"global"``.
        area: ``"circle"`` or ``"box"``.

    Returns:
        GBIF occurrence search params: none, ``geoDistance`` or a ``geometry``
        WKT polygon.
    """
    if isinstance(extent, str) and extent.lower() == "global":
        return {}
    if area == "circle":
        lat = parse_coordinate(latitude, "lat")
        lon = parse_coordinate(longitude, "lon")
        return {"geoDistance": f"{lat},{lon},{float(extent):g}km"}
    if area == "box":
        min_lat, max_lat, min_lon, max_lon = get_bounding_box(
            latitude, longitude, float(extent)
        )
        return {
            "geometry": f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
            f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        }
    raise ValueError(f"area must be one of {SEARCH_AREAS}, got {area!r}")


def search_taxon_occurrences(
    taxon_keys: list[str],
    area_filter: dict | None = None,
    batch_size: int = 100,
    user_agent: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Get GBIF occurrence counts for taxon keys, optionally within an area.

    Queries the GBIF occurrence facet API in batches. The facet lists every
    key the records fall under (each species, its ancestors and the taxa
    below it), most records first, so a full list may have cut off the rarest
    species: such a batch is split in two and asked again. A single species
    comes right after its ancestors, so its own list never cuts it off. Keys
    not found in any response are filled with count=0. Keys whose batch
    request errored are returned separately as failed keys (not counted) so
    the caller can retry them rather than treat them as absent.

    Args:
        taxon_keys: List of GBIF taxon key strings.
        area_filter: Search params of the area (see ``area_params``); none
            for worldwide counts.
        batch_size: Number of taxon keys per facet request.
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Tuple of (DataFrame with columns taxonID and count, list of taxon keys
        whose batch request failed).
    """
    taxon_keys = list(dict.fromkeys(taxon_keys))  # Remove duplicates, keep order

    if len(taxon_keys) == 0:
        return pd.DataFrame(columns=["taxonID", "count"]), []

    facet_limit = 10 * batch_size  # Room for the ancestors and taxa below
    params_base = {
        "hasCoordinate": "true",
        **(area_filter or {}),
        "facet": "taxonKey",
        "facetLimit": facet_limit,
        "limit": 0,
        "checklistKey": COL_XR_CHECKLIST,
    }

    # Split into 100-species batches
    batches = [
        taxon_keys[i : i + batch_size] for i in range(0, len(taxon_keys), batch_size)
    ]

    all_results = []
    failed_keys: list[str] = []
    session = create_session(user_agent=user_agent)
    expected_keys = set(taxon_keys)

    while batches:
        batch = batches.pop(0)
        params = params_base.copy()
        params["taxonKey"] = batch

        try:
            with request_label(f"GBIF occurrences of {len(batch)} species"):
                resp = session.get(
                    "https://api.gbif.org/v1/occurrence/search",
                    params=params,
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()

            facets = data.get("facets", [])
            counts = facets[0].get("counts", []) if facets else []
            if len(counts) >= facet_limit and len(batch) > 1:
                half = len(batch) // 2
                batches[:0] = [batch[:half], batch[half:]]
                continue
            # Only the batch's own keys: a key outside it counts just the
            # records it shares with the batch (a species asked for through
            # one of its subspecies).
            asked = set(batch)
            for facet in counts:
                taxon_key = facet["name"]

                if taxon_key in asked and taxon_key in expected_keys:
                    all_results.append(
                        {
                            "taxonID": taxon_key,
                            "count": facet["count"],
                        }
                    )
                    expected_keys.discard(taxon_key)

        except Exception as e:
            logger.warning(
                f"GBIF occurrences of {len(batch)} species failed: {str(e)[:100]}"
            )
            # Report the batch's keys as failed (not as a count) so the caller
            # retries them instead of caching them as absent. Drop them from
            # expected_keys so the zero-fill below does not emit a misleading
            # count=0 row for them.
            failed_keys.extend(batch)
            for key in batch:
                expected_keys.discard(key)

    session.close()

    # Fill zeros
    for key in expected_keys:
        all_results.append(
            {
                "taxonID": key,
                "count": 0,
            }
        )

    df = pd.DataFrame(all_results, columns=["taxonID", "count"])
    return df, failed_keys


def area_label(extent: int | float | str, area: str | None) -> str:
    """A search area as the log names it: "within 250 km", "in the box around
    250 km" or "worldwide".

    Args:
        extent: Search radius in km, or ``"global"``.
        area: ``"circle"`` or ``"box"``; None for an analysis saved before the
            area setting, which searched in boxes.
    """
    name = get_extent_column_name(extent)
    if name == "global":
        return "worldwide"
    return f"within {name}" if area == "circle" else f"in the box around {name}"


def get_extent_column_name(extent: int | float | str) -> str:
    """Standardize the column name for GBIF occurrence counts.

    Converts numeric extents to ``"<int> km"`` and string extents to lowercase.

    Args:
        extent: Extent value — numeric (km) or string (e.g. ``"global"``).

    Returns:
        Standardized column name string (e.g. ``"100 km"``, ``"global"``).
    """
    try:
        numeric_value = float(extent)
        return f"{int(numeric_value)} km"
    except (ValueError, TypeError):
        if isinstance(extent, str):
            return extent.lower()

    return str(extent)


def counts_for_area(
    taxon_keys: list[str],
    latitude: float | None = None,
    longitude: float | None = None,
    extent: int | float | str = "global",
    area: str = "circle",
    progress_handler: Callable[[float, str], None] | None = None,
    user_agent: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return GBIF occurrence counts for taxon keys within a geographic area.

    Queries either the full global dataset or the area around a point (see
    ``area_params``).

    Args:
        taxon_keys: List of GBIF taxon key strings.
        latitude: Center latitude (required for non-global extents).
        longitude: Center longitude (required for non-global extents).
        extent: Search radius in km, or ``"global"`` for worldwide counts.
        area: ``"circle"`` or ``"box"`` around the point.
        progress_handler: Optional callable ``(fraction, message)`` for UI
            progress updates. Receives values in [0, 1].
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Tuple of (DataFrame with columns ``taxonID``, ``occurrences``,
        ``gbif_extent``; list of taxon keys whose occurrence query failed).
    """

    col_name = get_extent_column_name(extent)

    # No keys to query (e.g. no species cleared the match filter): skip the
    # request and return an empty, correctly-columned frame so the caller's
    # reindex/merge does not fail.
    if not taxon_keys:
        return pd.DataFrame(columns=["taxonID", "occurrences", "gbif_extent"]), []

    if progress_handler:
        progress_handler(0.1, f"Querying {col_name}...")

    counts, failed_keys = search_taxon_occurrences(
        taxon_keys,
        area_params(latitude, longitude, extent, area),
        user_agent=user_agent,
    )

    # Failed keys are reported separately, not counted; exclude them so they are
    # not zero-filled and mistaken for "absent". One row per key, even when
    # several names share it.
    failed = set(failed_keys)
    kept_keys = [k for k in dict.fromkeys(taxon_keys) if k not in failed]
    df = counts.set_index("taxonID")["count"].reindex(kept_keys).fillna(0).reset_index()
    df.columns = ["taxonID", "occurrences"]
    df["gbif_extent"] = extent

    n_present = int((df["occurrences"] > 0).sum())
    n_absent = len(df) - n_present
    logger.info(
        f"GBIF occurrences {area_label(extent, area)}: {n_present} species present, "
        f"{n_absent} absent"
    )

    if progress_handler:
        progress_handler(1.0, f"Finished {col_name}")

    return df, failed_keys


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_bounding_boxes(
    latitude,
    longitude,
    extents,
    show_circles=True,
    title=None,
    padding=0.5,
    show_boxes=True,
):
    """Plot the search areas of geographic extents: circles and bounding boxes.

    Args:
        latitude: Center latitude in decimal degrees.
        longitude: Center longitude in decimal degrees.
        extents: List of distances from center in kilometers.
        show_circles: If True, draw the circular perimeter for each extent.
        show_boxes: If True, draw the bounding box of each circle.
        title: Optional plot title (empty/None to hide).
        padding: Padding around zones as fraction of range (default 0.5).

    Returns:
        Plotly Figure.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    fig = go.Figure()
    n_edge_points = 50

    palette = px.colors.qualitative.Plotly
    color_map = {ext: palette[i % len(palette)] for i, ext in enumerate(extents)}

    # Track outermost bounds across all extents for viewport
    view_min_lat = view_min_lon = float("inf")
    view_max_lat = view_max_lon = float("-inf")

    for extent_km in extents:
        color = color_map[extent_km]
        label = f"{extent_km}km"

        min_lat, max_lat, min_lon, max_lon = get_bounding_box(
            latitude, longitude, extent_km, return_circle=False
        )
        view_min_lat = min(view_min_lat, min_lat)
        view_max_lat = max(view_max_lat, max_lat)
        view_min_lon = min(view_min_lon, min_lon)
        view_max_lon = max(view_max_lon, max_lon)

        if show_circles:
            circle_lats, circle_lons = get_bounding_box(
                latitude, longitude, extent_km, return_circle=True
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=circle_lons,
                    lat=circle_lats,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>Circle {label}</b><br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
                )
            )

        if show_boxes:
            # Build closed rectangle with interpolated edges
            n = n_edge_points
            bbox_lons = np.concatenate(
                [
                    np.linspace(min_lon, max_lon, n),  # north
                    np.full(n, max_lon),  # east
                    np.linspace(max_lon, min_lon, n),  # south
                    np.full(n, min_lon),  # west
                    [min_lon],  # close
                ]
            )
            bbox_lats = np.concatenate(
                [
                    np.full(n, max_lat),  # north
                    np.linspace(max_lat, min_lat, n),  # east
                    np.full(n, min_lat),  # south
                    np.linspace(min_lat, max_lat, n),  # west
                    [max_lat],  # close
                ]
            )

            fig.add_trace(
                go.Scattergeo(
                    lon=bbox_lons,
                    lat=bbox_lats,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=f"<b>BBox {label}</b><br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
                )
            )

    # Center point
    fig.add_trace(
        go.Scattergeo(
            lon=[longitude],
            lat=[latitude],
            mode="markers",
            name="Center",
            marker=dict(size=12, color="black", symbol="cross"),
            showlegend=False,
            hovertemplate="<b>Center</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        )
    )

    # Compute projection scale from the largest extent so the view
    # is consistent regardless of latitude
    max_extent_km = max(extents)
    # projection_scale ~1.0 shows the whole globe; higher = more zoomed in
    # 20000 km ≈ half Earth circumference at scale 1.0
    proj_scale = 20000 / (max_extent_km * (1 + 2 * padding))

    layout_kwargs = {
        "geo": dict(
            center=dict(lon=longitude, lat=latitude),
            projection_type="azimuthal equal area",
            projection_scale=proj_scale,
            resolution=50,
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcoastlines=True,
            coastlinecolor="rgb(100, 100, 100)",
            showcountries=True,
            showocean=True,
            oceancolor="rgb(204, 229, 255)",
            showlakes=True,
            lakecolor="rgb(204, 229, 255)",
        ),
        "height": 500,
        "margin": dict(l=0, r=0, t=30, b=0),
        "legend": dict(
            x=0.5, y=1.02, xanchor="center", yanchor="bottom", orientation="h"
        ),
        "hovermode": "closest",
    }
    if title:
        layout_kwargs["title"] = title

    fig.update_layout(**layout_kwargs)
    return fig
