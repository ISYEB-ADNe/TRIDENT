"""Client for the GBIF (Global Biodiversity Information Facility) API
and geographic bounding box calculations."""

import re
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from loguru import logger


from trident.core.constants import GBIF_CONFIDENCE
from trident.core.http import create_session
from trident.core.utils import ensure_columns, extract_specific_epithet


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


def _match_single_taxon_name(
    name: str,
    session: requests.Session | None = None,
    user_agent: str | None = None,
) -> dict:
    """Match a single taxon name to GBIF taxon keys via the species/match API.

    Args:
        name: Species or taxon name to match.
        session: Optional requests session for connection reuse.
        user_agent: User-Agent header, used only when creating an own session.

    Returns:
        Dict with GBIF match metadata and taxon keys.
        Includes 'needs_review' flag for low-confidence matches.
    """
    key_columns = [
        "kingdomKey",
        "phylumKey",
        "classKey",
        "orderKey",
        "familyKey",
        "genusKey",
        "speciesKey",
        "usageKey",
        "acceptedUsageKey",
    ]

    own_session = session is None
    if own_session:
        session = create_session(user_agent=user_agent)
    try:
        response = session.get(
            "https://api.gbif.org/v1/species/match", params={"name": name}, timeout=15
        )
        response.raise_for_status()
        data = response.json()

        data["query"] = name

        # Force key columns to string (handle float/None/empty)
        for key_col in key_columns:
            if key_col in data and data[key_col] not in [None, ""]:
                data[key_col] = str(int(float(data[key_col])))
            else:
                data[key_col] = None

        # Main taxonID as string
        if data.get("status") == "SYNONYM" and data.get("acceptedUsageKey"):
            data["taxonID"] = str(int(float(data["acceptedUsageKey"])))
        elif data.get("speciesKey"):
            data["taxonID"] = str(int(float(data["speciesKey"])))
        else:
            data["taxonID"] = None

        data["needs_review"] = (
            data.get("confidence", 0) < GBIF_CONFIDENCE
            or data.get("matchType") == "NONE"
        )

        return data

    except Exception as e:
        logger.debug(f"Match failed for '{name}': {str(e)[:100]}")
        error_result = {
            "query": name,
            "confidence": 0,
            "matchType": "ERROR",
            "needs_review": True,
            "taxonID": None,
        }
        for key_col in key_columns:
            error_result[key_col] = None
        return error_result

    finally:
        if own_session:
            session.close()


def match_taxon_names(
    names: list[str], n_jobs: int = 10, user_agent: str | None = None
) -> pd.DataFrame:
    """Match species/taxon names to GBIF taxon keys using parallel requests.

    Args:
        names: List of scientific names to match.
        n_jobs: Number of parallel threads.
        user_agent: User-Agent header for HTTP requests.

    Returns:
        DataFrame with one row per name containing GBIF match metadata.
    """
    if len(names) == 0:
        logger.warning("Empty names list")
        return pd.DataFrame()

    # Each worker creates its own session: requests.Session is not thread-safe,
    # so sharing one across the threading backend can corrupt the pool state.
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_match_single_taxon_name)(name, user_agent=user_agent) for name in names
    )

    results_df = pd.DataFrame(results)
    high_conf = len(results_df[results_df["confidence"] >= GBIF_CONFIDENCE])
    review = len(results_df[results_df["needs_review"]])

    logger.info(
        f"Matched {high_conf}/{len(names)} species to GBIF backbone ({review} need review)"
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
]


def filter_taxon_matches(
    matches_df: pd.DataFrame,
    confidence_threshold: int = GBIF_CONFIDENCE,
    strict_name_match: bool = True,
) -> pd.DataFrame:
    """Filter GBIF matches by confidence and name quality.

    Args:
        matches_df: DataFrame from match_taxon_names()
        confidence_threshold: Min confidence score
        strict_name_match: Require query == canonicalName

    Returns:
        Filtered DataFrame with high-quality taxon matches only. Always carries
        the standardised taxonomy columns, even when empty.
    """
    if matches_df.empty:
        logger.warning("Empty DataFrame provided to filter_taxon_matches")
        return pd.DataFrame(columns=_GBIF_TAXON_COLS)

    # Quality filters
    mask = (
        (
            pd.to_numeric(matches_df["confidence"], errors="coerce")
            >= confidence_threshold
        )
        & (matches_df["matchType"] != "NONE")
        & matches_df["taxonID"].notna()
    )

    if strict_name_match:
        if "canonicalName" in matches_df.columns:
            mask &= (
                matches_df["query"].str.lower()
                == matches_df["canonicalName"].str.lower()
            )
        else:
            # No row carried a canonicalName (every match was NONE/ERROR), so
            # none can satisfy the strict name check.
            mask &= False

    filtered_df = matches_df[mask].copy()

    # Nothing passed: return an empty, correctly-columned frame before the
    # row-wise steps below (apply(axis=1) on an empty frame yields a DataFrame,
    # which cannot be assigned back to a single column).
    if filtered_df.empty:
        logger.warning("No matches passed quality filters")
        return pd.DataFrame(columns=_GBIF_TAXON_COLS)

    # Drop redundant/internal columns (keep useful taxonomy!)
    drop_cols = [
        "query",  # Replaced by canonicalName
        "scientificName",  # canonicalName is cleaner
        "confidence",  # Already filtered
        "matchType",  # Already filtered
        "synonym",  # Status tells us this
        "usageKey",  # taxonID is what we need
        "status",  # Resolved via taxonID
        "needs_review",  # Already filtered
        "species",  # Redundant with canonicalName
    ]

    # Safe drop (only existing columns)
    existing_drops = [col for col in drop_cols if col in filtered_df.columns]
    filtered_df = filtered_df.drop(columns=existing_drops).reset_index(drop=True)
    filtered_df = filtered_df.rename(
        columns={
            "canonicalName": "scientificName",
            "rank": "taxonRank",
        }
    )

    # Remove duplicate taxonIDs (keep first)
    before_dedup = len(filtered_df)
    filtered_df = filtered_df.drop_duplicates(subset=["taxonID"], keep="first")

    filtered_df["taxonID_db"] = "gbif"
    filtered_df["taxonURL"] = filtered_df["taxonID"].apply(
        lambda x: f"https://www.gbif.org/species/{x}" if pd.notna(x) else None
    )
    filtered_df["specificEpithet"] = filtered_df.apply(
        lambda row: extract_specific_epithet(row["scientificName"], row["genus"]),
        axis=1,
    )

    ensure_columns(filtered_df, _GBIF_TAXON_COLS)
    filtered_df = filtered_df[_GBIF_TAXON_COLS]

    dups_removed = before_dedup - len(filtered_df)
    dups_str = "duplicates" if dups_removed != 1 else "duplicate"
    dups_message = f" ({dups_removed} {dups_str} removed)" if dups_removed > 0 else ""
    logger.info(
        f"Retained {len(filtered_df)}/{len(matches_df)} taxon keys after quality filtering{dups_message}"
    )

    return filtered_df


def _build_occurrence_params(
    min_lat: float | None,
    max_lat: float | None,
    min_lon: float | None,
    max_lon: float | None,
) -> dict:
    """Build GBIF occurrence search params with optional bounding box geometry.

    Args:
        min_lat: Southern bound of the bounding box.
        max_lat: Northern bound of the bounding box.
        min_lon: Western bound of the bounding box.
        max_lon: Eastern bound of the bounding box.

    Returns:
        GBIF API query params dict, including a ``geometry`` WKT polygon
        when all bounds are provided.
    """
    params = {"hasCoordinate": "true", "limit": 10000}

    if all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
        geometry = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        params["geometry"] = geometry
        logger.debug(f"Geometry: {min_lon},{min_lat} to {max_lon},{max_lat}")

    return params


def search_taxon_occurrences(
    taxon_keys: list[str],
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    batch_size: int = 100,
    user_agent: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Get GBIF occurrence counts for taxon keys with optional bounding box.

    Queries the GBIF occurrence facet API in batches. Keys not found in any
    response are filled with count=0. Keys whose batch request errored are
    returned separately as failed keys (not counted) so the caller can retry
    them rather than treat them as absent.

    Args:
        taxon_keys: List of GBIF taxon key strings.
        min_lat: Southern bound of the bounding box.
        max_lat: Northern bound of the bounding box.
        min_lon: Western bound of the bounding box.
        max_lon: Eastern bound of the bounding box.
        batch_size: Number of taxon keys per facet request.
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Tuple of (DataFrame with columns taxonID and count, list of taxon keys
        whose batch request failed).
    """
    taxon_keys = list(set(taxon_keys))  # Remove duplicates

    if len(taxon_keys) == 0:
        return pd.DataFrame(columns=["taxonID", "count"]), []

    params_base = _build_occurrence_params(min_lat, max_lat, min_lon, max_lon)
    params_base.update(
        {
            "facet": "taxonKey",
            "facetLimit": 10 * batch_size,  # Get more facets to avoid missing keys
            "limit": 0,
        }
    )

    # Split into 100-species batches
    batches = [
        taxon_keys[i : i + batch_size] for i in range(0, len(taxon_keys), batch_size)
    ]

    all_results = []
    failed_keys: list[str] = []
    session = create_session(user_agent=user_agent)
    expected_keys = set(taxon_keys)

    for batch in batches:
        params = params_base.copy()
        params["taxonKey"] = batch

        try:
            resp = session.get(
                "https://api.gbif.org/v1/occurrence/search", params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            facets = data.get("facets", [])
            counts = facets[0].get("counts", []) if facets else []
            for facet in counts:
                taxon_key = facet["name"]

                if taxon_key in expected_keys:
                    all_results.append(
                        {
                            "taxonID": taxon_key,
                            "count": facet["count"],
                        }
                    )
                    expected_keys.discard(taxon_key)

        except Exception as e:
            logger.error(f"Batch failed: {str(e)[:100]}")
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
    progress_handler: Callable[[float, str], None] | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Return GBIF occurrence counts for taxon keys within a geographic area.

    Queries either the full global dataset or a bounding box around a point.

    Args:
        taxon_keys: List of GBIF taxon key strings.
        latitude: Center latitude (required for non-global extents).
        longitude: Center longitude (required for non-global extents).
        extent: Search radius in km, or ``"global"`` for worldwide counts.
        progress_handler: Optional callable ``(fraction, message)`` for UI
            progress updates. Receives values in [0, 1].
        user_agent: User-Agent header for HTTP requests.

    Returns:
        Tuple of (DataFrame with columns ``taxonID``, ``occurrences``,
        ``gbif_extent``; list of taxon keys whose occurrence query failed).
    """

    col_name = get_extent_column_name(extent)

    # No keys to query (e.g. no species cleared the backbone filter): skip the
    # request and return an empty, correctly-columned frame so the caller's
    # reindex/merge does not fail.
    if not taxon_keys:
        return pd.DataFrame(columns=["taxonID", "occurrences", "gbif_extent"]), []

    if progress_handler:
        progress_handler(0.1, f"Querying {col_name}...")

    if isinstance(extent, str) and extent.lower() == "global":
        counts, failed_keys = search_taxon_occurrences(
            taxon_keys, user_agent=user_agent
        )
    else:
        min_lat, max_lat, min_lon, max_lon = get_bounding_box(
            latitude, longitude, extent
        )
        counts, failed_keys = search_taxon_occurrences(
            taxon_keys, min_lat, max_lat, min_lon, max_lon, user_agent=user_agent
        )

    # Failed keys are reported separately, not counted; exclude them so they are
    # not zero-filled and mistaken for "absent".
    failed = set(failed_keys)
    kept_keys = [k for k in taxon_keys if k not in failed]
    df = counts.set_index("taxonID")["count"].reindex(kept_keys).fillna(0).reset_index()
    df.columns = ["taxonID", "occurrences"]
    df["gbif_extent"] = extent

    n_present = int((df["occurrences"] > 0).sum())
    n_absent = len(df) - n_present
    logger.info(f"Queried {col_name}: {n_present} present, {n_absent} absent")

    if progress_handler:
        progress_handler(1.0, f"Finished {col_name}")

    return df, failed_keys


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_bounding_boxes(
    latitude, longitude, extents, show_circles=True, title=None, padding=0.5
):
    """Plot bounding boxes and optional circles for geographic extents.

    Args:
        latitude: Center latitude in decimal degrees.
        longitude: Center longitude in decimal degrees.
        extents: List of distances from center in kilometers.
        show_circles: If True, draw the circular perimeter for each extent.
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
                    line=dict(color=color, width=1),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.1,
                    hovertemplate=f"<b>Circle {label}</b><br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
                )
            )

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
