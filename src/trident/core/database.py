import functools
import hashlib
import inspect
import itertools
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from loguru import logger

from trident.core import config
from trident.core.utils import (
    describe_settings,
    ensure_columns,
    pluralize,
    set_progress_total,
)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(
    db_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, creating the file if needed (by default
    trident_data.db in the results folder).

    Commits on clean exit, rolls back on exception, and always closes the
    connection so file handles are not leaked across cache calls.
    """
    if db_path is None:
        db_path = config.results_dir() / "trident_data.db"
    db_path = Path(db_path).absolute()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        # A new database can give back the space of deleted rows (see
        # _after_deleting). Set on an empty one only (no page yet, even when
        # an earlier connection left the file): on an existing one it changes
        # nothing but still writes the header, so reading a saved analysis
        # would change its bytes.
        if conn.execute("PRAGMA page_count").fetchone()[0] == 0:
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _normalize_number(value):
    """Write whole floats as ints, recursively: 90.0 and 90 make the same key.

    The notebook passes floats where the app's widgets pass ints; without this
    the two never share a cache entry.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_number(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_number(v) for k, v in value.items()}
    return value


def _params_to_sql_values(params: dict) -> dict:
    """Convert param values to SQL-safe strings (JSON for dicts/lists)."""
    values = {k: _normalize_number(v) for k, v in params.items()}
    return {
        k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
        for k, v in values.items()
    }


# ---------------------------------------------------------------------------
# Schema version
#
# Stored in SQLite's built-in PRAGMA user_version (a header int, no table).
# Bump SCHEMA_VERSION when a change breaks reading an older database. A
# change that only adds columns or tables needs no bump: columns are added
# when first written, and the provenance and step_columns tables are optional
# when reading.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2  # 2: sequences in the sequence store


def _set_schema_version(conn: sqlite3.Connection) -> None:
    """Stamp the current schema version into the database header."""
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def check_schema_version(db_path: str | Path) -> tuple[bool, str]:
    """Whether *db_path* can be read by this TRIDENT version.

    Returns (compatible, message). A file that is not SQLite, or a database
    written by a newer TRIDENT (user_version > SCHEMA_VERSION), is refused;
    equal or older is accepted (older databases re-run any steps whose
    columns changed).
    """
    try:
        with get_connection(db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError:
        return False, "This file is not a TRIDENT analysis file."
    if version > SCHEMA_VERSION:
        return False, (
            f"This analysis file was created by a newer version of TRIDENT "
            f"(schema v{version} > v{SCHEMA_VERSION}). Update TRIDENT to open it."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Provenance
#
# Each cached run_id records the UTC date it was queried from the external
# database and the trident version that produced it. The date is the
# de-facto version of the live source (NCBI core_nt, GBIF, BOLD, WoRMS): the
# version a methods section can cite.
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


def _delete_provenance(conn: sqlite3.Connection, step: str, run_ids: list[int]) -> None:
    """Drop the provenance rows of *run_ids* for *step*."""
    if not table_exists(conn, _PROVENANCE_TABLE):
        return
    if run_ids:
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
        if not table_exists(conn, _PROVENANCE_TABLE):
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
# Step columns
#
# An empty result stores no rows, so on a cache hit it would come back without
# columns, and any consumer reading a column would fail; nor do the rows of an
# item saved alone carry the columns only other items fill (a HYPO check
# without a hit has no hit columns). Each step records every column its
# function has returned, and a result read back is given those columns. A step
# whose call makes its whole result (see CacheStrategy.whole_result) records
# only that call's columns, so a column it no longer returns is left out.
# ---------------------------------------------------------------------------

_COLUMNS_TABLE = "step_columns"


def _load_step_columns(conn: sqlite3.Connection, step: str) -> list[str]:
    """Return the columns recorded for *step*, or [] (older databases)."""
    if not table_exists(conn, _COLUMNS_TABLE):
        return []
    row = conn.execute(
        f"SELECT columns FROM {_COLUMNS_TABLE} WHERE step = ?", (step,)
    ).fetchone()
    return json.loads(row[0]) if row else []


def _record_step_columns(
    conn: sqlite3.Connection, step: str, df: pd.DataFrame, replace: bool = False
) -> None:
    """Record the columns of *df* not yet known for *step*, or, with
    *replace*, the columns of *df* only."""
    known = _load_step_columns(conn, step)
    if replace:
        columns = list(df.columns)
    else:
        columns = known + [col for col in df.columns if col not in known]
    if not columns or columns == known:
        return
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_COLUMNS_TABLE} "
        "(step TEXT PRIMARY KEY, columns TEXT NOT NULL)"
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {_COLUMNS_TABLE} (step, columns) VALUES (?, ?)",
        (step, json.dumps(columns)),
    )


# ---------------------------------------------------------------------------
# Sequence store
#
# The rows of a step repeat the sequence of their item (each BLAST hit its
# query, each HYPO row its proxy): most of an analysis file, for a few distinct
# sequences. A *_sequence column of a step table holds the id of its sequence
# in the sequence store, and a result read back gets the sequence again. The
# sequences table (the analysis input, read directly) and rows saved before
# keep the sequence itself, which reading leaves as it is.
# ---------------------------------------------------------------------------

_SEQUENCE_STORE = "sequence_store"
_PLAIN_TABLES = {"sequences"}


def _sequence_columns(table_name: str, columns) -> list[str]:
    """The *columns* of *table_name* whose sequences go to the sequence store."""
    if table_name in _PLAIN_TABLES:
        return []
    return [col for col in columns if str(col).endswith("_sequence")]


def _store_sequences(
    conn: sqlite3.Connection, table_name: str, df: pd.DataFrame
) -> pd.DataFrame:
    """*df* with the sequence of each sequence column replaced by its id in
    the sequence store (as text), added there if new."""
    columns = _sequence_columns(table_name, df.columns)
    if not columns:
        return df
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_SEQUENCE_STORE} "
        "(id INTEGER PRIMARY KEY, sequence TEXT NOT NULL UNIQUE)"
    )
    sequences = list({v for col in columns for v in df[col] if isinstance(v, str)})
    conn.executemany(
        f"INSERT OR IGNORE INTO {_SEQUENCE_STORE} (sequence) VALUES (?)",
        [(sequence,) for sequence in sequences],
    )
    # Ids as text: a number column with a missing value would be written as
    # floats ("1.0").
    ids = dict(
        conn.execute(
            f"SELECT sequence, CAST(id AS TEXT) FROM {_SEQUENCE_STORE} "
            "WHERE sequence IN (SELECT value FROM json_each(?))",
            (json.dumps(sequences),),
        )
    )
    return df.assign(**{col: df[col].map(lambda v: ids.get(v, v)) for col in columns})


def _restore_sequences(
    conn: sqlite3.Connection, table_name: str, df: pd.DataFrame
) -> None:
    """Put back, in place, the sequences of *df* stored as ids. Only the ids
    are replaced, so a column keeps the type and the missing values reading
    it gives; a sequence saved before the store is no id and stays. An id
    comes back as text, or as a number in a column typed as one (12.0)."""
    columns = _sequence_columns(table_name, df.columns)
    if not columns or not table_exists(conn, _SEQUENCE_STORE):
        return
    keys = {_key_str(v) for col in columns for v in df[col].dropna().unique()}
    ids = [int(key) for key in keys if key.isdigit()]
    by_id = dict(
        conn.execute(
            f"SELECT CAST(id AS TEXT), sequence FROM {_SEQUENCE_STORE} "
            "WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(ids),),
        )
    )
    for col in columns:
        restored = df[col].map(lambda v: by_id.get(_key_str(v)))
        found = restored.notna()
        if found.any():
            if df[col].dtype.kind in "fiu":  # a number column cannot take text
                df[col] = df[col].astype(object)
            df.loc[found, col] = restored[found]


def _after_deleting(conn: sqlite3.Connection) -> None:
    """After step rows were deleted: delete the sequences no row holds any
    more, then give the freed space back to the file (a file made before
    auto_vacuum keeps it until a VACUUM)."""
    if table_exists(conn, _SEQUENCE_STORE):
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        used = [
            f'SELECT CAST("{col}" AS INTEGER) FROM {table} WHERE "{col}" IS NOT NULL'
            for (table,) in tables
            for col in _sequence_columns(
                table, [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            )
        ]
        where = f" WHERE id NOT IN ({' UNION '.join(used)})" if used else ""
        conn.execute(f"DELETE FROM {_SEQUENCE_STORE}{where}")
    conn.commit()
    conn.execute("PRAGMA incremental_vacuum").fetchall()


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
    column names and dtypes (90 and 90.0 differ); insensitive to row and
    column order.
    """
    if df is None or len(df) == 0:
        return "empty"
    df = df[sorted(df.columns, key=str)]
    # Row-order-independent: sort the per-row hashes (vectorised) and hash the
    # raw bytes, avoiding a per-row str()/join over large frames.
    row_hashes = np.sort(pd.util.hash_pandas_object(df, index=False).to_numpy())
    return hashlib.sha1(row_hashes.tobytes()).hexdigest()[:16]


def _fingerprint_params(kwargs: dict, fingerprint_on: list[str] | None) -> dict:
    """Build ``{_fp_<kwarg>: fingerprint(df)}`` for each named upstream frame."""
    if not fingerprint_on:
        return {}
    return {f"_fp_{name}": fingerprint(kwargs.get(name)) for name in fingerprint_on}


def _item_filter(
    values: list, match_map: dict[str, str] | None
) -> Callable[[dict], bool]:
    """Build a predicate that returns True for the cache items *values* names.

    *values* holds identifying values (e.g. species names) the pipeline
    function reported to ``failure_sink`` (``result_sink`` reads its values
    the same way). A plain value is
    matched against the first ``match_map`` key of each cache item, a tuple
    against all its keys in the order of ``match_map``. With no values or no
    match_map the predicate is always False.

    A plain value names every item sharing that first key. This is intentional
    for gbif: name matches fail per species but are cached per (species,
    extent), so a failed species drops all its extents. A step whose first key
    recurs across items that fail independently (the HYPO proxy records,
    shared by several sequences) reports tuples instead, so a failure drops only
    its own item.
    """
    if not values or not match_map:
        return lambda item: False
    keys = list(match_map.keys())
    # Tuples compare as strings, like the run_id assignment of result rows.
    tuple_keys = {
        tuple(map(_key_str, value)) for value in values if isinstance(value, tuple)
    }
    plain_values = {value for value in values if not isinstance(value, tuple)}
    return lambda item: (
        item.get(keys[0]) in plain_values or _item_key(item, match_map) in tuple_keys
    )


def _key_str(value) -> str:
    """*value* as a key string. A whole float is written as an int, as in the
    inputs table: pandas turns an int column with a missing value into floats,
    and the row of extent 500.0 still belongs to the item of extent 500."""
    return str(_normalize_number(value))


def _item_key(item: dict, match_map: dict[str, str] | None) -> tuple:
    """The values of *item* its result rows carry (see ``_row_keys``), as
    strings; () for a step without match_map (one item)."""
    return tuple(_key_str(item.get(k)) for k in match_map or {})


def _row_keys(df: pd.DataFrame, match_map: dict[str, str]) -> pd.Series:
    """The item key (``_item_key``) of each row of *df*."""
    columns = [df[col].map(_key_str) for col in match_map.values()]
    return pd.Series(list(zip(*columns)), index=df.index, dtype=object)


def _find_cached_run_ids(
    conn: sqlite3.Connection, table_name: str, items: list[dict]
) -> list[int | None]:
    """The run_id of each of *items*: the first inputs row holding its values
    (as ``_params_to_sql_values`` writes them), or None. None for all when
    the inputs table, or one of an item's keys in it, does not exist yet.

    The inputs table is read once: a step can have thousands of items.
    """
    try:
        cursor = conn.execute(f"SELECT * FROM {table_name}_inputs")
    except sqlite3.OperationalError:
        return [None] * len(items)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    run_id_at = columns.index("run_id")
    found = []
    by_keys: dict[tuple, dict] = {}  # item keys -> their values -> run_id
    for item in items:
        keys = tuple(item)
        if keys not in by_keys:
            by_keys[keys] = {}
            if set(keys) <= set(columns):
                at = [columns.index(k) for k in keys]
                for row in rows:
                    stored = tuple(_stored_key(row[i]) for i in at)
                    by_keys[keys].setdefault(stored, row[run_id_at])
        values = tuple(_params_to_sql_values(item).values())
        found.append(by_keys[keys].get(values))
    return found


def _stored_key(value):
    """An inputs value as the text ``_params_to_sql_values`` writes: a number
    stored as a number (by a column's type) matches its text, as SQLite's
    ``IS`` comparison makes it."""
    if value is None or isinstance(value, str):
        return value
    return str(_normalize_number(value))


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Whether *table_name* exists.

    Results tables are created lazily by ``to_sql`` on the first run that
    produces rows, so a 0-row result leaves only its ``_inputs`` entry and no
    results table. Every reader treats a missing results table as "0 rows".
    """
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    )
    return cursor.fetchone() is not None


def _index_run_id(conn: sqlite3.Connection, table_name: str) -> None:
    """Index the run_id of *table_name*, if it exists: each cached item is
    read by its run_id, which would otherwise scan the whole table."""
    if table_exists(conn, table_name):
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name}_run_id ON {table_name} (run_id)"
        )


def _load_cached_items(
    conn: sqlite3.Connection, table_name: str, run_ids: list[int]
) -> pd.DataFrame:
    """The rows of the items *run_ids*, in that order (each item's rows in
    table order), without the run_id column; empty without rows.

    The column types are those reading each item alone and joining the
    reads would give, in a few reads instead of one per item (seconds on
    thousands of items): pandas types a column from the kinds of values it
    reads (text, whole numbers, decimals, empty...), so the items whose
    columns hold the same SQLite kinds (``typeof``) are read together, then
    joined.
    """
    if not run_ids or not table_exists(conn, table_name):
        return pd.DataFrame()
    in_ids = "run_id IN (SELECT value FROM json_each(?))"
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")]
    kinds_sql = ", ".join(f'group_concat(DISTINCT typeof("{col}"))' for col in columns)
    kinds = conn.execute(
        f"SELECT run_id, {kinds_sql} FROM {table_name} WHERE {in_ids} GROUP BY run_id",
        (json.dumps(run_ids),),
    ).fetchall()
    if not kinds:  # the items have no rows (a search with no result)
        return pd.DataFrame()
    groups: dict[tuple, list[int]] = {}
    for run_id, *columns in kinds:
        key = tuple(frozenset(column.split(",")) for column in columns)
        groups.setdefault(key, []).append(run_id)
    reads = [
        pd.read_sql_query(
            f"SELECT * FROM {table_name} WHERE {in_ids} ORDER BY rowid",
            conn,
            params=(json.dumps(ids),),
        )
        for ids in groups.values()
    ]
    df = pd.concat(reads, ignore_index=True)
    _restore_sequences(conn, table_name, df)
    position = {run_id: i for i, run_id in enumerate(run_ids)}
    order = df["run_id"].map(position).to_numpy().argsort(kind="stable")
    return df.iloc[order].drop(columns="run_id").reset_index(drop=True)


def _delete_run(conn: sqlite3.Connection, table_name: str, run_id: int) -> None:
    """Delete all rows for *run_id* from *table_name*."""
    if not table_exists(conn, table_name):
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


def _delete_orphan_rows(conn: sqlite3.Connection, table_name: str) -> None:
    """Delete the result rows without an inputs row, left by a process killed
    while saving (see ``_save_items``): their run_id may be given again. With
    no inputs table yet (killed during the first save), every row is one."""
    inputs_table = f"{table_name}_inputs"
    if not table_exists(conn, table_name):
        return
    where = (
        f" WHERE run_id NOT IN (SELECT run_id FROM {inputs_table})"
        if table_exists(conn, inputs_table)
        else ""
    )
    cursor = conn.execute(f"DELETE FROM {table_name}{where}")
    conn.commit()
    if cursor.rowcount:
        logger.debug(f"{table_name}: deleted {cursor.rowcount} rows of no item")
        _after_deleting(conn)


def _append_rows(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    """Append *df* to *table_name*, creating the table with the first rows.

    A column with no value in the rows that create the table is added untyped,
    like a column added later (``_add_missing_columns``): pandas would type it
    from those rows alone (TEXT for None, REAL for NaN), and SQLite would then
    store its later values as text or floats (an id 126160 as '126160' or
    126160.0). Sequences go to the sequence store (``_store_sequences``) in
    the transaction that writes the rows, after every commit before them: a
    cleanup on another connection (``_after_deleting``) cannot delete them
    before the rows that hold them are written.
    """
    if not table_exists(conn, table_name):
        empty = df.columns[df.isna().all()]
        df.drop(columns=empty).iloc[:0].to_sql(name=table_name, con=conn, index=False)
    _add_missing_columns(conn, table_name, df)
    df = _store_sequences(conn, table_name, df)
    df.to_sql(name=table_name, con=conn, if_exists="append", index=False)


def _save_items(
    conn: sqlite3.Connection,
    table_name: str,
    items: list[dict],
    df: pd.DataFrame | None,
    match_map: dict[str, str] | None,
    new_ids: Iterator[int],
) -> None:
    """Save the cache *items*, which finished without failing: their rows of
    *df* (rows of other items are left out), one inputs row each (also for an
    item without rows, so a genuine empty is not queried again) and their
    provenance, the UTC date they were queried from the live source. Each item
    takes the next run_id of *new_ids*, shared by the saves of one call.

    pandas commits each table it writes, so the inputs rows, which make the
    items cached, are written last: a process killed in between leaves rows
    without their item, never an item without its rows. The next call deletes
    those rows (``_delete_orphan_rows``) before giving their run_id again.
    """
    if not match_map and len(items) != 1:
        raise ValueError(
            f"{table_name}: match_map is empty but {len(items)} items to save. "
            "Cannot assign run_ids without mapping."
        )
    _set_schema_version(conn)
    run_ids = {_item_key(item, match_map): next(new_ids) for item in items}

    n_rows = 0
    if df is not None and len(df) > 0:
        if match_map:
            ids = _row_keys(df, match_map).map(run_ids)
        else:
            ids = pd.Series(next(iter(run_ids.values())), index=df.index)
        rows = df[ids.notna()].assign(run_id=ids.dropna().astype(int))
        if len(rows) > 0:
            _append_rows(conn, table_name, rows)
        n_rows = len(rows)

    inputs_df = pd.DataFrame(
        [
            {
                **_params_to_sql_values(item),
                "run_id": run_ids[_item_key(item, match_map)],
            }
            for item in items
        ]
    )
    _add_missing_columns(conn, f"{table_name}_inputs", inputs_df)
    # Committed with the inputs rows.
    _write_provenance(
        conn, table_name, list(run_ids.values()), _today_utc(), _trident_version()
    )
    inputs_df.to_sql(
        name=f"{table_name}_inputs", con=conn, if_exists="append", index=False
    )
    logger.debug(f"{table_name}: saved {n_rows} rows of {len(items)} items")


def _result_sink(
    db_path: str | Path,
    table_name: str,
    items: list[dict],
    match_map: dict[str, str] | None,
    saved: set[tuple],
    new_ids: Iterator[int],
) -> Callable[[pd.DataFrame, list], None]:
    """The ``result_sink`` given to a step function (see ``save_to_db``).

    Called with the rows of items that just finished and their identifying
    values (as for ``failure_sink``), it saves those of *items* not saved yet
    and adds their keys to *saved*. It may be called from the step's worker
    threads: each call saves on its own connection, one call at a time. It
    never raises: items it cannot save are saved when the step returns, and
    the step goes on.
    """
    lock = threading.Lock()
    first_key = next(iter(match_map))
    by_key = {_item_key(item, match_map): item for item in items}
    by_first: dict = {}  # items sharing their first key value (see _item_filter)
    for item in items:
        by_first.setdefault(item.get(first_key), []).append(item)

    def result_sink(df: pd.DataFrame, done: list) -> None:
        finished = [
            item
            for value in done
            for item in (
                [by_key.get(tuple(map(_key_str, value)))]
                if isinstance(value, tuple)
                else by_first.get(value, [])
            )
            if item is not None
        ]
        with lock:
            batch = [
                item for item in finished if _item_key(item, match_map) not in saved
            ]
            if not batch:
                return
            try:
                with get_connection(db_path) as conn:
                    _record_step_columns(conn, table_name, df)
                    _save_items(conn, table_name, batch, df, match_map, new_ids)
            except Exception as exc:
                logger.warning(
                    f"Could not save {len(batch)} results to the analysis file "
                    f"yet ({exc}): they are saved when the step ends."
                )
                return
            saved.update(_item_key(item, match_map) for item in batch)

    return result_sink


def _as_booleans(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Give *columns* of *df* as booleans, in place: SQLite gives them back as
    1/0, and an empty result as untyped columns."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].eq(True)


def _id_ranges(ids: list[int]) -> str:
    """*ids* sorted, runs written as ranges: "1-6, 9, 12-14"."""
    ranges: list[list[int]] = []
    for i in sorted(set(ids)):
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def warn_empty_params(params: dict, name: str) -> None:
    """Log when an upstream params dict is empty (weakens cache key). At DEBUG:
    it is for developers, and an empty upstream result also gives one."""
    if not params:
        logger.debug(
            f"'{name}' is empty (an earlier step found nothing, or its settings "
            "are unknown)"
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

    # Column a result row needs for its item to count as a result (see
    # _empty_run_ids); None: any row does.
    result_col: str | None = None

    # Whether one call of the step makes its whole result (a step that only
    # processes previous results): its columns are then those of its last
    # call, not every column it ever returned (see _record_step_columns).
    whole_result: bool = False

    @property
    def match_map(self) -> dict[str, str] | None:
        """Map from cache-dict keys to output DataFrame columns.

        Used to assign run_ids to result rows.
        Return ``None`` for single-entry caching (FullCache).
        """
        return None


# Version of the results of the steps computed locally (FullCache, and
# CustomCache with versioned=True). Their key is made of settings and data, not
# code: bump it when a step's output columns or rows change, so each step
# computes its result again (locally) instead of returning one saved by the
# old code.
# It is part of the step's params, so the later steps' keys follow. Searches
# are not versioned: a bump never sends their requests again.
RESULTS_VERSION = 2


class FullCache(CacheStrategy):
    """Whole-step caching: one entry keyed by merged upstream + local params.

    Use for filter / merge / finalize steps where the result is either
    fully cached or fully recomputed.
    """

    whole_result = True

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
        params["_results_version"] = RESULTS_VERSION
        return [params], params


class PartialCache(CacheStrategy):
    """Per-item caching: each item cached independently, only missing
    items are re-executed.

    Use for search steps where each item (sequence, species, genus)
    can be fetched independently.

    No ``fingerprint_on``: per-item caches already re-run only changed/new
    items, so whole-batch content fingerprinting does not apply here.

    *result_col*: for a step that saves a row explaining why an item has no
    result (e.g. a species without a BOLD sequence), the column only real
    results fill; an item with no row filling it counts as empty, so "Ask again
    for empty results" queries it again.

    *constants*: cache-key values that are not arguments (e.g. the BLAST
    database), so results saved under another value are not reused.
    """

    def __init__(
        self,
        items_kwarg: str,
        item_key: str,
        extract: Callable | None = None,
        params: dict[str, str] | None = None,
        output_col: str | None = None,
        result_col: str | None = None,
        constants: dict | None = None,
    ):
        self.result_col = result_col
        self.items_kwarg = items_kwarg
        self.item_key = item_key
        self.extract = extract
        self.params = params or {}
        self.constants = constants or {}
        self.output_col = output_col

    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        items = kwargs[self.items_kwarg]
        global_params = {
            **{
                cache_key: kwargs[kwarg_name]
                for cache_key, kwarg_name in self.params.items()
            },
            **self.constants,
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
    FullCache/PartialCache. *versioned* adds RESULTS_VERSION to the key, for a
    step computed locally."""

    def __init__(
        self,
        prepare_fn: Callable,
        rebuild_fn: Callable | None = None,
        match_map_dict: dict[str, str] | None = None,
        fingerprint_on: list[str] | None = None,
        whole_result: bool = False,
        versioned: bool = False,
    ):
        self.whole_result = whole_result
        self._prepare_fn = prepare_fn
        self._rebuild_fn = rebuild_fn
        self._match_map = match_map_dict
        self.fingerprint_on = fingerprint_on or []
        self.versioned = versioned

    def prepare(self, **kwargs) -> tuple[list[dict], dict]:
        inputs, params = self._prepare_fn(**kwargs)
        if self.versioned:
            version = {"_results_version": RESULTS_VERSION}
            params = {**params, **version}
            inputs = [{**item, **version} for item in inputs]
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


def _log_cache_use(
    name: str, log_as: tuple[str, str] | None, n_cached: int, n_new: int
) -> None:
    """First line of a step: its items found in the database, and the others,
    which it queries (see ``save_to_db``)."""
    if log_as is None:
        logger.debug(f"{name}: {n_cached} cached, {n_new} to execute")
        return
    step, one = log_as
    n = n_cached + n_new
    noun = pluralize(n, one)
    if not n_cached:
        logger.info(f"{step}: {n} {noun} to query")
    elif not n_new:
        logger.info(f"{step}: all {n} {noun} already queried")
    else:
        logger.info(f"{step}: {n} {noun}, {n_cached} already queried, {n_new} to query")


def save_to_db(
    table_name: str,
    cache: CacheStrategy,
    log_as: tuple[str, str] | None = None,
    log_reuse_as: str | None = None,
    bool_columns: tuple[str, ...] = (),
) -> Callable:
    """Decorator that adds SQLite caching to a pipeline function.

    Always returns ``(DataFrame, params_dict)``.  When ``db_path`` is
    ``None`` caching is bypassed but the return shape is preserved.

    The wrapped function gains three keyword arguments:
    ``db_path`` (str | Path | None), ``force_rerun`` (bool),
    and ``retry_empty`` (bool).

    If the wrapped function declares a ``failure_sink`` parameter, it receives a
    list to append the identifying values of items whose query failed: a value
    of the first ``match_map`` key, or a tuple of all its keys to fail one item
    only. Those items are excluded from caching and retried on the next run
    (see ``_item_filter``). The values reported during this call are also
    exposed as ``df.attrs["failed_items"]`` on the returned DataFrame, so the
    UI can tell failed items from genuinely empty ones. The attribute lives in
    memory only (not stored in SQLite) and describes this call: items that
    failed in an earlier call are re-executed, so they appear again only if
    they fail again. pandas may copy ``attrs`` onto frames derived from the
    result; only read it on the frame returned here.

    If it declares a ``result_sink`` parameter (given only with a db_path and
    a per-item cache, one with a ``match_map``), it
    receives a function to call with the rows of items that just finished and
    their identifying values, as for ``failure_sink``: those items are saved at
    once, so a crash or a killed process loses only the items still running.
    It may be called from worker threads. The function still returns all its
    rows; the items it did not report are saved when it returns.

    A ``progress_handler`` (a dict with a ``"total"`` key, or tqdm-like) gets
    as total the number of items this call executes, so a run resumed from the
    cache ends its progress bar full. The function must count one step per
    item.

    A result keeps its columns, even empty: each step records the columns its
    function returns, and they are restored on a cache hit. With no items to
    look up (empty input), the function is called directly and nothing is
    cached.

    *log_as* is (step name, what an item is), e.g. ``("BOLD search",
    "species")``: the step's first log line then says how
    many of its items are already in the database and how many it queries.
    Without it (the steps that only process a previous step's results), that
    line is logged at DEBUG.

    *bool_columns* are the step's true/false columns: SQLite stores them as
    1/0, and a result read back gives them as booleans again, as a live one
    does.

    *log_reuse_as* is the name a step that only processes previous results
    gives its summary line, e.g. ``"GEO filter"``: when its whole result is
    reused, the step does not run, so it logs that instead of its summary.

    A step named by *log_as* or *log_reuse_as* lists its own settings at
    INFO after its first line, whether it runs or its result is reused, so a
    kept log says which settings produced the result: the keys of its params
    that have a label (SETTING_LABELS), except those it inherits (found in a
    dict argument, the params of an earlier step).
    """
    match_map = cache.match_map
    step = log_as[0] if log_as else log_reuse_as

    def log_step_settings(params: dict, bound_kwargs: dict) -> None:
        if not step:
            return
        inherited = set().union(
            *(v.keys() for v in bound_kwargs.values() if isinstance(v, dict))
        )
        own = {k: v for k, v in params.items() if k not in inherited}
        settings = describe_settings(own)
        if settings:
            logger.info(f"{step} settings: {settings}")

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

            # One entry per distinct item: a duplicate would get its own inputs
            # row, while its result rows all go to one run_id, so a cache hit
            # could load the empty one.
            cache_items = list(
                {
                    tuple(_params_to_sql_values(item).items()): item
                    for item in cache_items
                }.values()
            )

            # Nothing to cache without a db_path or without items (empty
            # input). The function is still called, so the caller gets its
            # empty result with its columns, and its failed items as with a db.
            if db_path is None or not cache_items:
                if cache_items:
                    _log_cache_use(func.__name__, log_as, 0, len(cache_items))
                    log_step_settings(params, bound_kwargs)
                sink = bound_kwargs.get("failure_sink")
                if sink is None and "failure_sink" in sig.parameters:
                    sink = kwargs["failure_sink"] = []
                result = func(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    _as_booleans(result, bool_columns)
                    if sink is not None:
                        result.attrs["failed_items"] = list(sink)
                return result, params

            with get_connection(db_path) as conn:
                # 0. Clear empty cache entries so failed items get retried
                if retry_empty:
                    clear_empty_cache(
                        db_path,
                        table_name,
                        cache.result_col,
                        step=log_as[0] if log_as else table_name,
                    )

                cached_dfs = []
                cached_ids = []
                uncached_items = []

                # 1. Classify each item as cached or uncached
                _index_run_id(conn, table_name)
                run_ids = _find_cached_run_ids(conn, table_name, cache_items)
                replaced = []
                for item, run_id in zip(cache_items, run_ids):
                    if run_id is not None and not force_rerun:
                        cached_ids.append(run_id)
                    else:
                        if run_id is not None:
                            _delete_run(conn, table_name, run_id)
                            _delete_run(conn, f"{table_name}_inputs", run_id)
                            _delete_provenance(conn, table_name, [run_id])
                            replaced.append(run_id)
                        uncached_items.append(item)
                if replaced:
                    _after_deleting(conn)
                    logger.debug(
                        f"{table_name}: replaced run_ids {_id_ranges(replaced)}"
                    )
                if cached_ids:
                    logger.debug(
                        f"{table_name}: reused run_ids {_id_ranges(cached_ids)}"
                    )

                cached_dfs.append(_load_cached_items(conn, table_name, cached_ids))
                _log_cache_use(
                    func.__name__, log_as, len(cached_ids), len(uncached_items)
                )
                log_step_settings(params, bound_kwargs)
                if log_reuse_as and not uncached_items:
                    logger.info(f"{log_reuse_as}: saved result reused (same settings)")
                # A progress tracker counts items as they are executed, so its
                # total is what this call executes: a resumed run starts at 0
                # and ends at its total, cached items aside.
                set_progress_total(
                    bound_kwargs.get("progress_handler"), len(uncached_items)
                )

                # 2. Execute uncached items. A function that accepts a
                # ``failure_sink`` can report items whose query failed (e.g.
                # network/403); those are excluded from caching below so they
                # are retried on the next run instead of cached as empty. One
                # that accepts a ``result_sink`` saves each item as it finishes.
                failure_sink: list = []
                saved: set[tuple] = set()  # keys of the items result_sink saved
                if uncached_items:
                    run_kwargs = cache.rebuild(uncached_items, bound_kwargs)
                    if "failure_sink" in sig.parameters:
                        run_kwargs["failure_sink"] = failure_sink
                    _delete_orphan_rows(conn, table_name)
                    new_ids = itertools.count(_next_run_id(conn, table_name))
                    if "result_sink" in sig.parameters and match_map:
                        run_kwargs["result_sink"] = _result_sink(
                            db_path,
                            table_name,
                            uncached_items,
                            match_map,
                            saved,
                            new_ids,
                        )
                    # result_sink writes on its own connections, which this
                    # one would block with uncommitted deletions.
                    conn.commit()
                    new_df = func(**run_kwargs)
                else:
                    new_df = None

                # 3. Save the items not saved yet. Items the function reported
                # as failed get neither result rows nor an inputs row, so they
                # are retried next run; their rows are still returned.
                if uncached_items:
                    if new_df is not None and not isinstance(new_df, pd.DataFrame):
                        raise TypeError(
                            f"{func.__name__} must return a pandas DataFrame, "
                            f"got {type(new_df)}"
                        )
                    if new_df is not None:
                        _record_step_columns(
                            conn, table_name, new_df, replace=cache.whole_result
                        )
                    failed = _item_filter(failure_sink, match_map)
                    rest = [
                        item
                        for item in uncached_items
                        if not failed(item) and _item_key(item, match_map) not in saved
                    ]
                    if rest:
                        _save_items(conn, table_name, rest, new_df, match_map, new_ids)
                    n_failed = sum(map(failed, uncached_items))
                    if n_failed:
                        logger.debug(
                            f"{func.__name__}: {n_failed} item(s) failed and were "
                            "not cached (will retry on next run)"
                        )

                    if new_df is not None and len(new_df) > 0:
                        if match_map:
                            keys = {
                                _item_key(item, match_map) for item in uncached_items
                            }
                            matched = (
                                _row_keys(new_df, match_map)
                                .map(keys.__contains__)
                                .astype(bool)
                            )
                            if not matched.all():
                                logger.warning(
                                    f"{(~matched).sum()} rows could not be matched "
                                    "to a cache entry"
                                )
                            new_df = new_df[matched]
                        cached_dfs.append(new_df)

                # 4. Combine and return
                non_empty = [df for df in cached_dfs if not df.empty]
                result_df = (
                    pd.concat(non_empty, ignore_index=True)
                    if non_empty
                    else pd.DataFrame()
                )
                # The step's columns, in the order it recorded them: a result
                # read back has the table's order, where a column added after
                # the table was created comes last, so a live and a cached
                # result would differ.
                columns = _load_step_columns(conn, table_name)
                ensure_columns(result_df, columns)
                # Columns the table keeps from older results stay out of a
                # whole result.
                extra = [] if cache.whole_result and columns else result_df.columns
                result_df = result_df[columns + [c for c in extra if c not in columns]]
                _as_booleans(result_df, bool_columns)
                result_df.attrs["failed_items"] = list(failure_sink)
                return result_df, params

        return wrapper

    return decorator


def _empty_run_ids(
    conn: sqlite3.Connection, table_name: str, result_col: str | None = None
) -> list[int]:
    """run_ids cached as input rows but with no matching results rows (no row
    filling *result_col*, when given)."""
    inputs_table = f"{table_name}_inputs"
    if not table_exists(conn, inputs_table):
        return []
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    if columns:
        filled = f" AND r.{result_col} IS NOT NULL" if result_col else ""
        cursor = conn.execute(
            f"SELECT i.run_id FROM {inputs_table} i "
            f"LEFT JOIN {table_name} r ON i.run_id = r.run_id{filled} "
            f"WHERE r.run_id IS NULL"
        )
    else:
        # No results table at all: every cached input run is empty.
        cursor = conn.execute(f"SELECT run_id FROM {inputs_table}")
    return [row[0] for row in cursor.fetchall()]


def clear_empty_cache(
    db_path: str | Path,
    table_name: str,
    result_col: str | None = None,
    step: str | None = None,
) -> int:
    """Remove cached entries that produced zero result rows (no row filling
    *result_col*, when given; see PartialCache).

    Their items are then re-queried on the next run, without re-running items
    that returned rows. Failed queries do not need this: they are never cached
    (see ``failure_sink`` in ``save_to_db``) and are retried on every run. It
    serves genuinely empty results worth querying again (the source database
    may have changed), and databases written before failures were tracked,
    where a failed query was cached as empty.

    *step* names the step in the log line (the table name by default).

    Returns the number of entries cleared.
    """
    inputs_table = f"{table_name}_inputs"
    with get_connection(db_path) as conn:
        empty_ids = _empty_run_ids(conn, table_name, result_col)
        if empty_ids and result_col and table_exists(conn, table_name):
            # Their explanation rows go too, or the re-query would add to them.
            placeholders = ",".join("?" * len(empty_ids))
            conn.execute(
                f"DELETE FROM {table_name} WHERE run_id IN ({placeholders})",
                empty_ids,
            )

        if empty_ids:
            placeholders = ",".join("?" * len(empty_ids))
            conn.execute(
                f"DELETE FROM {inputs_table} WHERE run_id IN ({placeholders})",
                empty_ids,
            )
            _delete_provenance(conn, table_name, empty_ids)
            _after_deleting(conn)
            n = len(empty_ids)
            logger.info(
                f"{step or table_name}: {n} empty saved {pluralize(n, 'result')} "
                "cleared"
            )
        return len(empty_ids)
