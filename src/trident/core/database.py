import functools
import hashlib
import inspect
import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from loguru import logger

from trident.core import config


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(
    db_path: str | Path = "./results/trident_data.db",
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, creating the file if needed.

    Commits on clean exit, rolls back on exception, and always closes the
    connection so file handles are not leaked across cache calls.
    """
    db_path = Path(db_path).absolute()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _params_to_sql_values(params: dict) -> dict:
    """Convert param values to SQL-safe strings (JSON for dicts/lists)."""
    return {
        k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
        for k, v in params.items()
    }


# ---------------------------------------------------------------------------
# Schema version
#
# Stored in SQLite's built-in PRAGMA user_version (a header int, no table).
# Bump SCHEMA_VERSION when a change breaks reading an older database. So far
# all changes have been additive (new columns added lazily, new provenance
# table read-tolerantly), so older databases stay readable.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def _set_schema_version(conn: sqlite3.Connection) -> None:
    """Stamp the current schema version into the database header."""
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def check_schema_version(db_path: str | Path) -> tuple[bool, str]:
    """Whether *db_path* can be read by this TRIDENT version.

    Returns (compatible, message). A database written by a newer TRIDENT
    (user_version > SCHEMA_VERSION) is refused; equal or older is accepted
    (older databases re-run any steps whose columns changed).
    """
    with get_connection(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        return False, (
            f"This database was created by a newer version of TRIDENT "
            f"(schema v{version} > v{SCHEMA_VERSION}). Update TRIDENT to open it."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Provenance
#
# Each cached run_id records the UTC date it was queried from the external
# database and the trident version that produced it. The date is the
# de-facto version of the live source (NCBI nt, GBIF, BOLD, WoRMS), which is
# what makes a result reproducible-describable.
# ---------------------------------------------------------------------------

_PROVENANCE_TABLE = "provenance"


def _trident_version() -> str:
    """Installed trident version (single source: config.app_version)."""
    return config.app_version()


def _today_utc() -> str:
    """Current UTC date as an ISO string (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).date().isoformat()


def _ensure_provenance_table(conn: sqlite3.Connection) -> None:
    """Create the shared provenance table if it does not yet exist."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_PROVENANCE_TABLE} ("
        "step TEXT NOT NULL, "
        "run_id INTEGER NOT NULL, "
        "queried_on TEXT, "
        "trident_version TEXT, "
        "PRIMARY KEY (step, run_id))"
    )


def _write_provenance(
    conn: sqlite3.Connection,
    step: str,
    run_ids: list[int],
    queried_on: str,
    trident_version: str,
) -> None:
    """Record provenance for newly executed run_ids of *step*."""
    if not run_ids:
        return
    _ensure_provenance_table(conn)
    conn.executemany(
        f"INSERT OR REPLACE INTO {_PROVENANCE_TABLE} "
        "(step, run_id, queried_on, trident_version) VALUES (?, ?, ?, ?)",
        [(step, run_id, queried_on, trident_version) for run_id in run_ids],
    )


def _delete_provenance(
    conn: sqlite3.Connection, step: str, run_ids: list[int] | None = None
) -> None:
    """Drop provenance rows for *step* (optionally only specific run_ids)."""
    if not _table_exists(conn, _PROVENANCE_TABLE):
        return
    if run_ids is None:
        conn.execute(f"DELETE FROM {_PROVENANCE_TABLE} WHERE step = ?", (step,))
    elif run_ids:
        placeholders = ",".join("?" * len(run_ids))
        conn.execute(
            f"DELETE FROM {_PROVENANCE_TABLE} "
            f"WHERE step = ? AND run_id IN ({placeholders})",
            (step, *run_ids),
        )


def load_provenance(db_path: str | Path, table_name: str | None = None) -> pd.DataFrame:
    """Return recorded provenance, optionally filtered to one step.

    Tolerates databases written before provenance existed (returns an empty
    frame with the expected columns).
    """
    cols = ["step", "run_id", "queried_on", "trident_version"]
    with get_connection(db_path) as conn:
        if not _table_exists(conn, _PROVENANCE_TABLE):
            return pd.DataFrame(columns=cols)
        if table_name is not None:
            return pd.read_sql_query(
                f"SELECT * FROM {_PROVENANCE_TABLE} WHERE step = ? ORDER BY run_id",
                conn,
                params=(table_name,),
            )
        return pd.read_sql_query(
            f"SELECT * FROM {_PROVENANCE_TABLE} ORDER BY step, run_id", conn
        )


# ---------------------------------------------------------------------------
# Input fingerprinting
#
# A cached step that inherits only upstream *parameters* will not re-run when
# the upstream *data* changes (e.g. a previously-failed item now returns rows).
# Steps that aggregate an upstream DataFrame include a fingerprint of that
# DataFrame in their cache key, so a content change forces a re-run.
# ---------------------------------------------------------------------------


def fingerprint(df: "pd.DataFrame | None") -> str:
    """Stable, row-order-independent content hash of a DataFrame.

    Returns ``"empty"`` for an empty or missing frame. Sensitive to values,
    columns, and dtypes; insensitive to row order.
    """
    if df is None or len(df) == 0:
        return "empty"
    # Row-order-independent: sort the per-row hashes (vectorised) and hash the
    # raw bytes, avoiding a per-row str()/join over large frames.
    row_hashes = np.sort(pd.util.hash_pandas_object(df, index=False).to_numpy())
    return hashlib.sha1(row_hashes.tobytes()).hexdigest()[:16]


def _fingerprint_params(kwargs: dict, fingerprint_on: list[str] | None) -> dict:
    """Build ``{_fp_<kwarg>: fingerprint(df)}`` for each named upstream frame."""
    if not fingerprint_on:
        return {}
    return {f"_fp_{name}": fingerprint(kwargs.get(name)) for name in fingerprint_on}


def _failed_item_filter(
    failure_sink: list, match_map: dict[str, str] | None
) -> Callable[[dict], bool]:
    """Build a predicate that returns True for cache items that failed.

    *failure_sink* holds the identifying values (e.g. species names) the
    pipeline function reported as failed. They are matched against the first
    ``match_map`` key of each cache item. With no failures or no match_map the
    predicate is always False (nothing is treated as failed).

    The first match_map key defines the failure granularity. This is
    intentional: gbif failures are per-species but cached per (species,
    extent), so matching the first key (species) drops every extent for a
    failed species. The reporter must push values for that first key.
    """
    failed_values = set(failure_sink or [])
    if not failed_values or not match_map:
        return lambda item: False
    key_field = next(iter(match_map.keys()))
    return lambda item: item.get(key_field) in failed_values


def _find_cached_run_id(
    conn: sqlite3.Connection, table_name: str, item: dict
) -> int | None:
    """Return the run_id matching *item*, or None."""
    inputs_table = f"{table_name}_inputs"

    try:
        conditions = " AND ".join([f"{k} IS ?" for k in item.keys()])
        values = tuple(_params_to_sql_values(item).values())
        cursor = conn.execute(
            f"SELECT run_id FROM {inputs_table} WHERE {conditions}", values
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Whether *table_name* exists.

    Results tables are created lazily by ``to_sql`` on the first run that
    produces rows, so a 0-row result leaves only its ``_inputs`` entry and no
    results table. Every reader treats a missing results table as "0 rows".
    """
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    )
    return cursor.fetchone() is not None


def _load_cached_df(
    conn: sqlite3.Connection, table_name: str, run_id: int
) -> pd.DataFrame:
    """Load cached results for *run_id* (without the run_id column)."""
    if not _table_exists(conn, table_name):
        return pd.DataFrame()
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE run_id = ?", conn, params=(run_id,)
    )
    if "run_id" in df.columns:
        df = df.drop("run_id", axis=1)
    return df


def _delete_run(conn: sqlite3.Connection, table_name: str, run_id: int) -> None:
    """Delete all rows for *run_id* from *table_name*."""
    if not _table_exists(conn, table_name):
        return
    conn.execute(f"DELETE FROM {table_name} WHERE run_id = ?", (run_id,))
    conn.commit()
    logger.debug(f"Deleted data for run_id={run_id} in table={table_name}")


def _add_missing_columns(
    conn: sqlite3.Connection, table_name: str, df: pd.DataFrame
) -> None:
    """Add columns present in *df* but missing from the table schema."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()
    if not rows:
        return  # Table doesn't exist yet; to_sql will create it

    existing_cols = {row[1] for row in rows}
    for col in df.columns:
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE {table_name} ADD COLUMN "{col}"')
            logger.debug(f"Added column '{col}' to table '{table_name}'")
    conn.commit()


def _next_run_id(conn: sqlite3.Connection, table_name: str) -> int:
    """Return max(run_id) + 1 for the inputs table, or 1 if empty."""
    inputs_table = f"{table_name}_inputs"
    try:
        cursor = conn.execute(f"SELECT MAX(run_id) FROM {inputs_table}")
        max_id = cursor.fetchone()[0]
        return (max_id or 0) + 1
    except sqlite3.OperationalError:
        return 1


def warn_empty_params(params: dict, name: str) -> None:
    """Warn when an upstream params dict is empty (weakens cache key)."""
    if not params:
        logger.warning(
            f"'{name}' is empty! "
            "The cache will not be able to distinguish between different settings."
        )


# ---------------------------------------------------------------------------
# Cache strategies
# ---------------------------------------------------------------------------


class CacheStrategy(ABC):
    """Interface for ``save_to_db`` caching behaviour."""

    @abstractmethod
    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        """Return ``(cache_items, params)`` from the function's bound kwargs.

        *cache_items*: one dict per independently-cacheable unit.
        *params*: dict passed downstream as the step's output params.
        """

    def rebuild(self, uncached: list[dict], bound_kwargs: dict) -> dict:
        """Narrow *bound_kwargs* to only the uncached items.

        Default returns *bound_kwargs* unchanged (whole-step caching).
        """
        return bound_kwargs

    @property
    def match_map(self) -> dict[str, str] | None:
        """Map from cache-dict keys to output DataFrame columns.

        Used to assign run_ids to result rows.
        Return ``None`` for single-entry caching (FullCache).
        """
        return None


class FullCache(CacheStrategy):
    """Whole-step caching — one entry keyed by merged upstream + local params.

    Use for filter / merge / finalize steps where the result is either
    fully cached or fully recomputed.
    """

    def __init__(
        self,
        inherit_from: list[str] | None = None,
        local: dict[str, str] | None = None,
        fingerprint_on: list[str] | None = None,
    ):
        self.inherit_from = inherit_from or []
        self.local = local or {}
        self.fingerprint_on = fingerprint_on or []

    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        params: dict = {}
        for name in self.inherit_from:
            upstream = kwargs.get(name) or {}
            warn_empty_params(upstream, name)
            params |= upstream
        for cache_key, kwarg_name in self.local.items():
            params[cache_key] = kwargs[kwarg_name]
        # Re-run when the upstream data changed, not only when params changed.
        params |= _fingerprint_params(kwargs, self.fingerprint_on)
        return [params], params


class PartialCache(CacheStrategy):
    """Per-item caching — each item cached independently, only missing
    items are re-executed.

    Use for search steps where each item (sequence, species, genus)
    can be fetched independently.

    No ``fingerprint_on``: per-item caches already re-run only changed/new
    items, so whole-batch content fingerprinting does not apply here.
    """

    def __init__(
        self,
        items_kwarg: str,
        item_key: str,
        extract: Callable | None = None,
        params: dict[str, str] | None = None,
        output_col: str | None = None,
    ):
        self.items_kwarg = items_kwarg
        self.item_key = item_key
        self.extract = extract
        self.params = params or {}
        self.output_col = output_col

    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        items = kwargs[self.items_kwarg]
        global_params = {
            cache_key: kwargs[kwarg_name]
            for cache_key, kwarg_name in self.params.items()
        }
        cache_items = [
            {
                self.item_key: (self.extract(item) if self.extract else item),
                **global_params,
            }
            for item in items
        ]
        return cache_items, global_params

    def rebuild(self, uncached: list[dict], bound_kwargs: dict) -> dict:
        keys_to_keep = {item[self.item_key] for item in uncached}
        items = bound_kwargs[self.items_kwarg]
        if self.extract:
            filtered = [item for item in items if self.extract(item) in keys_to_keep]
        else:
            filtered = [item for item in items if item in keys_to_keep]
        return bound_kwargs | {self.items_kwarg: filtered}

    @property
    def match_map(self) -> dict[str, str]:
        output = self.output_col or self.item_key
        return {self.item_key: output}


class CustomCache(CacheStrategy):
    """Delegate to raw callables for patterns too complex for
    FullCache/PartialCache."""

    def __init__(
        self,
        prepare_fn: Callable,
        rebuild_fn: Callable | None = None,
        match_map_dict: dict[str, str] | None = None,
        fingerprint_on: list[str] | None = None,
    ):
        self._prepare_fn = prepare_fn
        self._rebuild_fn = rebuild_fn
        self._match_map = match_map_dict
        self.fingerprint_on = fingerprint_on or []

    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        inputs, params = self._prepare_fn(**kwargs)
        fp = _fingerprint_params(kwargs, self.fingerprint_on)
        if fp:
            # Add the upstream-data fingerprint to the shared params and to each
            # item so a content change invalidates the cache entries.
            params = {**params, **fp}
            inputs = [{**item, **fp} for item in inputs]
        return inputs, params

    def rebuild(self, uncached: list[dict], bound_kwargs: dict) -> dict:
        if self._rebuild_fn:
            return self._rebuild_fn(uncached, bound_kwargs)
        return bound_kwargs

    @property
    def match_map(self) -> dict[str, str] | None:
        return self._match_map


# ---------------------------------------------------------------------------
# save_to_db decorator
# ---------------------------------------------------------------------------


def save_to_db(table_name: str, cache: CacheStrategy) -> Callable:
    """Decorator that adds SQLite caching to a pipeline function.

    Always returns ``(DataFrame, params_dict)``.  When ``db_path`` is
    ``None`` caching is bypassed but the return shape is preserved.

    The wrapped function gains three keyword arguments:
    ``db_path`` (str | Path | None), ``force_rerun`` (bool),
    and ``retry_empty`` (bool).

    If the wrapped function declares a ``failure_sink`` parameter, it receives a
    list to append the identifying values of items whose query failed; those
    items are excluded from caching and retried on the next run (see
    ``_failed_item_filter``).
    """
    match_map = cache.match_map

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            *args, db_path=None, force_rerun=False, retry_empty=False, **kwargs
        ):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            bound_kwargs = dict(bound.arguments)
            cache_items, params = cache.prepare(**bound_kwargs)

            # No db_path: bypass caching entirely
            if db_path is None:
                return func(*args, **kwargs), params

            with get_connection(db_path) as conn:
                # 0. Clear empty cache entries so failed items get retried
                if retry_empty:
                    clear_empty_cache(db_path, table_name)

                cached_dfs = []
                uncached_items = []

                # 1. Classify each item as cached or uncached
                for idx, item in enumerate(cache_items):
                    run_id = _find_cached_run_id(conn, table_name, item)

                    if run_id is not None and not force_rerun:
                        logger.debug(
                            f"Item {idx + 1}/{len(cache_items)}: cached (run_id={run_id})"
                        )
                        cached_dfs.append(_load_cached_df(conn, table_name, run_id))
                    else:
                        if run_id is not None:
                            logger.debug(
                                f"Item {idx + 1}: will replace run_id={run_id}"
                            )
                            _delete_run(conn, table_name, run_id)
                            _delete_run(conn, f"{table_name}_inputs", run_id)
                            _delete_provenance(conn, table_name, [run_id])
                        uncached_items.append(item)

                logger.info(
                    f"{func.__name__}: {len(cached_dfs)} cached, "
                    f"{len(uncached_items)} to execute"
                )

                # 2. Execute uncached items. A function that accepts a
                # ``failure_sink`` can report items whose query failed (e.g.
                # network/403); those are excluded from caching below so they
                # are retried on the next run instead of cached as empty.
                failure_sink: list = []
                if uncached_items:
                    run_kwargs = cache.rebuild(uncached_items, bound_kwargs)
                    if "failure_sink" in sig.parameters:
                        run_kwargs["failure_sink"] = failure_sink
                    new_df = func(**run_kwargs)
                else:
                    new_df = None

                # 3. Save new results and cache entries
                if uncached_items:
                    _set_schema_version(conn)
                    base_run_id = _next_run_id(conn, table_name)

                    # 3a. Save result rows (if any)
                    if new_df is not None and len(new_df) > 0:
                        if not isinstance(new_df, pd.DataFrame):
                            raise TypeError(
                                f"{func.__name__} must return a pandas DataFrame, "
                                f"got {type(new_df)}"
                            )

                        if not match_map:
                            if len(uncached_items) != 1:
                                raise ValueError(
                                    f"{func.__name__}: match_map is empty but "
                                    f"{len(uncached_items)} uncached items found. "
                                    "Cannot assign run_ids without mapping."
                                )
                            new_df["run_id"] = base_run_id
                        else:
                            item_to_run_id = {
                                tuple(
                                    str(item[k]) for k in match_map.keys()
                                ): base_run_id + i
                                for i, item in enumerate(uncached_items)
                            }

                            def assign_run_id(row):
                                key = tuple(str(row[col]) for col in match_map.values())
                                return item_to_run_id.get(key)

                            new_df["run_id"] = new_df.apply(assign_run_id, axis=1)

                        unmatched = new_df["run_id"].isna().sum()
                        if unmatched > 0:
                            logger.warning(
                                f"{unmatched} rows could not be matched to a cache entry"
                            )
                            new_df = new_df.dropna(subset=["run_id"])

                        new_df["run_id"] = new_df["run_id"].astype(int)

                        _add_missing_columns(conn, table_name, new_df)
                        new_df.to_sql(
                            name=table_name, con=conn, if_exists="append", index=False
                        )
                        logger.debug(
                            f"Saved {len(new_df)} rows across {len(uncached_items)} run_ids"
                        )
                        cached_dfs.append(new_df.drop("run_id", axis=1))

                    # 3b. Cache input entries (even for 0-row results) so
                    # legitimately empty queries are not re-executed. Items the
                    # function reported as failed are skipped, so a transient
                    # failure is retried next run rather than cached as empty.
                    failed = _failed_item_filter(failure_sink, match_map)
                    kept = [
                        (i, item)
                        for i, item in enumerate(uncached_items)
                        if not failed(item)
                    ]
                    n_failed = len(uncached_items) - len(kept)
                    if n_failed:
                        logger.warning(
                            f"{func.__name__}: {n_failed} item(s) failed and were "
                            "not cached (will retry on next run)"
                        )

                    if kept:
                        inputs_rows = [
                            {**_params_to_sql_values(item), "run_id": base_run_id + i}
                            for i, item in kept
                        ]
                        inputs_df = pd.DataFrame(inputs_rows)
                        _add_missing_columns(conn, f"{table_name}_inputs", inputs_df)
                        inputs_df.to_sql(
                            name=f"{table_name}_inputs",
                            con=conn,
                            if_exists="append",
                            index=False,
                        )
                        logger.debug(f"Saved {len(inputs_rows)} input rows")

                        # 3c. Record provenance: the UTC date these items were
                        # queried from the live source, shared across the batch.
                        _write_provenance(
                            conn,
                            table_name,
                            [base_run_id + i for i, _ in kept],
                            _today_utc(),
                            _trident_version(),
                        )

                # 4. Combine and return
                non_empty = [df for df in cached_dfs if not df.empty]
                result_df = (
                    pd.concat(non_empty, ignore_index=True)
                    if non_empty
                    else pd.DataFrame()
                )
                return result_df, params

        return wrapper

    return decorator


def view_cached_runs(db_path: str | Path, table_name: str) -> pd.DataFrame:
    """Return a summary of cached runs (params + row counts per run_id)."""
    inputs_table = f"{table_name}_inputs"
    with get_connection(db_path) as conn:
        if not _table_exists(conn, inputs_table):
            logger.warning(f"No cache found for '{table_name}'")
            return pd.DataFrame()

        inputs_df = pd.read_sql_query(
            f"SELECT * FROM {inputs_table} ORDER BY run_id", conn
        )

        # A missing results table means every run was empty (row_count 0).
        if _table_exists(conn, table_name):
            counts = pd.read_sql_query(
                f"SELECT run_id, COUNT(*) as row_count FROM {table_name} GROUP BY run_id",
                conn,
            )
        else:
            counts = pd.DataFrame(columns=["run_id", "row_count"])

        result = inputs_df.merge(counts, on="run_id", how="left")
        result["row_count"] = result["row_count"].fillna(0).astype(int)

        # Attach the queried-on date when provenance is available.
        if _table_exists(conn, _PROVENANCE_TABLE):
            prov = pd.read_sql_query(
                f"SELECT run_id, queried_on FROM {_PROVENANCE_TABLE} WHERE step = ?",
                conn,
                params=(table_name,),
            )
            result = result.merge(prov, on="run_id", how="left")
        return result


def clear_cache(
    db_path: str | Path, table_name: str, run_id: int | None = None
) -> None:
    """Drop a specific run or the entire table (+ its inputs table)."""
    inputs_table = f"{table_name}_inputs"
    with get_connection(db_path) as conn:
        if run_id is not None:
            _delete_run(conn, table_name, run_id)
            conn.execute(f"DELETE FROM {inputs_table} WHERE run_id = ?", (run_id,))
            _delete_provenance(conn, table_name, [run_id])
            conn.commit()
            logger.info(f"Cleared run_id={run_id} from '{table_name}'")
        else:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"DROP TABLE IF EXISTS {inputs_table}")
            _delete_provenance(conn, table_name)
            conn.commit()
            logger.info(f"Cleared all cache for '{table_name}'")


def _empty_run_ids(conn: sqlite3.Connection, table_name: str) -> list[int]:
    """run_ids cached as input rows but with no matching results rows."""
    inputs_table = f"{table_name}_inputs"
    if not _table_exists(conn, inputs_table):
        return []
    if _table_exists(conn, table_name):
        cursor = conn.execute(
            f"SELECT i.run_id FROM {inputs_table} i "
            f"LEFT JOIN {table_name} r ON i.run_id = r.run_id "
            f"WHERE r.run_id IS NULL"
        )
    else:
        # No results table at all: every cached input run is empty.
        cursor = conn.execute(f"SELECT run_id FROM {inputs_table}")
    return [row[0] for row in cursor.fetchall()]


def clear_empty_cache(db_path: str | Path, table_name: str) -> int:
    """Remove cached entries that produced zero result rows.

    Useful for retrying items that failed (e.g. network error) without
    re-running items that succeeded.

    Returns the number of entries cleared.
    """
    inputs_table = f"{table_name}_inputs"
    with get_connection(db_path) as conn:
        empty_ids = _empty_run_ids(conn, table_name)

        if empty_ids:
            placeholders = ",".join("?" * len(empty_ids))
            conn.execute(
                f"DELETE FROM {inputs_table} WHERE run_id IN ({placeholders})",
                empty_ids,
            )
            _delete_provenance(conn, table_name, empty_ids)
            conn.commit()
            logger.info(
                f"Cleared {len(empty_ids)} empty cache entries from '{table_name}'"
            )
        return len(empty_ids)
