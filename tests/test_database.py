"""Tests for trident.core.database — save_to_db decorator and cache strategies."""

import random
import sqlite3

import pandas as pd
import pytest
from loguru import logger

from datetime import datetime, timezone

from trident.core import database
from trident.core.stop import run_threads
from trident.core.utils import notify_progress
from trident.core.database import (
    CustomCache,
    FullCache,
    PartialCache,
    save_to_db,
    clear_empty_cache,
    load_provenance,
    fingerprint,
    SCHEMA_VERSION,
    check_schema_version,
    get_connection,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


# --- FullCache: basic round-trip ---


def test_full_cache_reuses_a_result_for_the_same_params_only(db_path):
    """Same params: the saved result, not the new input; other params: computed
    again."""
    calls = []

    @save_to_db("test_full", cache=FullCache(local={"threshold": "threshold"}))
    def my_step(data, threshold=90):
        calls.append(threshold)
        return pd.DataFrame({"x": data})

    df1, params1 = my_step([1, 2, 3], threshold=90, db_path=db_path)
    assert params1["threshold"] == 90
    df2, params2 = my_step([99, 99], threshold=90, db_path=db_path)
    assert df2["x"].tolist() == [1, 2, 3]
    assert params2 == params1
    my_step([1], threshold=95, db_path=db_path)
    assert calls == [90, 95]


def test_result_read_back_keeps_its_true_false_columns(db_path):
    """A step's true/false column comes back from the analysis as booleans,
    as the live result gives it (SQLite stores 1/0)."""

    @save_to_db(
        "test_bools", cache=FullCache(local={"t": "t"}), bool_columns=("in_mol",)
    )
    def my_step(t=1):
        return pd.DataFrame({"name": ["A", "B"], "in_mol": [True, False]})

    live, _ = my_step(db_path=db_path)
    cached, _ = my_step(db_path=db_path)
    pd.testing.assert_frame_equal(live, cached, check_dtype=True)


def test_full_cache_computes_again_for_a_new_results_version(db_path, monkeypatch):
    """A result saved under an older RESULTS_VERSION is computed again, with
    the columns the step returns now: one it no longer returns stays out,
    read live or from the cache."""

    @save_to_db("test_version", cache=FullCache(local={"t": "t"}))
    def my_step(data, t=1, name="x"):
        return pd.DataFrame({name: data})

    my_step([1, 2, 3], db_path=db_path)
    monkeypatch.setattr(database, "RESULTS_VERSION", database.RESULTS_VERSION + 1)
    live, _ = my_step([7], name="y", db_path=db_path)
    cached, _ = my_step([8], name="y", db_path=db_path)
    assert live.to_dict("list") == cached.to_dict("list") == {"y": [7]}


def test_full_cache_whole_float_and_int_share_the_cache(db_path):
    """The notebook passes 90.0 where the app passes 90: same cache entry,
    also inside lists. A fractional value stays distinct."""
    call_count = 0

    @save_to_db("test_full_num", cache=FullCache(local={"t": "t", "e": "extents"}))
    def my_step(t=90, extents=None):
        nonlocal call_count
        call_count += 1
        return pd.DataFrame({"val": [t]})

    my_step(t=90.0, extents=[500.0, "global"], db_path=db_path)
    my_step(t=90, extents=[500, "global"], db_path=db_path)
    assert call_count == 1

    my_step(t=90.5, extents=[500, "global"], db_path=db_path)
    assert call_count == 2


def test_step_logs_its_settings_and_whether_its_result_was_reused(db_path):
    """A step lists its own settings in the log (not those of the earlier step
    it inherits), whether it runs or its result is reused; a reused step does
    not run, so it logs that it was reused instead of its summary."""

    @save_to_db(
        "test_reuse",
        cache=FullCache(
            local={"gbif_min_occurrences": "t"}, inherit_from=["merge_params"]
        ),
        log_reuse_as="GEO filter",
    )
    def my_step(t=3, merge_params=None):
        logger.info("GEO filter: 3 species kept")
        return pd.DataFrame({"val": [t]})

    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    logger.enable("trident")  # as the app does (the package disables it)
    try:
        merge_params = {"gbif_area": "circle"}
        my_step(t=3, merge_params=merge_params, db_path=db_path)
        my_step(t=3, merge_params=merge_params, db_path=db_path)
        my_step(t=5, merge_params=merge_params, db_path=db_path)
    finally:
        logger.disable("trident")
        logger.remove(sink)
    assert lines == [
        "GEO filter settings: Minimum occurrences: 3",
        "GEO filter: 3 species kept",
        "GEO filter settings: Minimum occurrences: 3",
        "GEO filter: saved result reused (same settings)",
        "GEO filter settings: Minimum occurrences: 5",
        "GEO filter: 3 species kept",
    ]


def test_full_cache_force_rerun(db_path):
    """force_rerun bypasses cache."""
    call_count = 0

    @save_to_db("test_force", cache=FullCache(local={"t": "threshold"}))
    def my_step(threshold=90):
        nonlocal call_count
        call_count += 1
        return pd.DataFrame({"val": [threshold]})

    my_step(threshold=90, db_path=db_path)
    my_step(threshold=90, db_path=db_path, force_rerun=True)
    assert call_count == 2


def test_full_cache_no_db():
    """No db_path → bypass caching, still returns (df, params)."""

    @save_to_db("test_nodb", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    df, params = my_step(t=5)
    assert len(df) == 1
    assert params["t"] == 5


def test_full_cache_key_holds_every_upstream_and_own_params(db_path):
    """A step inheriting the params of two upstream steps is computed again
    when any of them or its own params change, and leaves theirs untouched."""
    calls = []

    @save_to_db(
        "test_inherit",
        cache=FullCache(inherit_from=["mol_params", "tax_params"], local={"t": "t"}),
    )
    def my_step(t=90, mol_params=None, tax_params=None):
        calls.append((t, mol_params["a"], tax_params["b"]))
        return pd.DataFrame({"val": [t]})

    mol, tax = {"a": 1}, {"b": 1}
    _, params = my_step(t=90, mol_params=mol, tax_params=tax, db_path=db_path)
    assert (params["a"], params["b"], params["t"]) == (1, 1, 90)
    assert (mol, tax) == ({"a": 1}, {"b": 1})
    for a, b, t in [(2, 1, 90), (2, 2, 90), (2, 2, 95), (2, 2, 95)]:
        my_step(t=t, mol_params={"a": a}, tax_params={"b": b}, db_path=db_path)
    assert calls == [(90, 1, 1), (90, 2, 1), (90, 2, 2), (95, 2, 2)]


# --- PartialCache: per-item caching ---


def test_partial_cache_caches_per_item(db_path):
    """PartialCache re-executes only uncached items."""
    executed_items = []

    @save_to_db(
        "test_partial",
        cache=PartialCache(items_kwarg="items", item_key="name"),
    )
    def my_step(items):
        executed_items.extend(items)
        return pd.DataFrame({"name": items, "val": range(len(items))})

    # First call: all items executed
    df1, _ = my_step(["a", "b", "c"], db_path=db_path)
    assert len(df1) == 3
    assert executed_items == ["a", "b", "c"]

    # Second call: add "d", only "d" should execute
    executed_items.clear()
    df2, _ = my_step(["a", "b", "c", "d"], db_path=db_path)
    assert len(df2) == 4
    assert executed_items == ["d"]


def test_partial_cache_reuses_items_only_under_the_same_constants(db_path):
    """Items saved without a constant (an older analysis) or under another
    value of it (another BLAST database) are searched again."""
    executed = []

    def step(constants):
        @save_to_db(
            "test_constants",
            cache=PartialCache(
                items_kwarg="items", item_key="name", constants=constants
            ),
        )
        def my_step(items):
            executed.extend(items)
            return pd.DataFrame({"name": items})

        return my_step

    step(None)(["a"], db_path=db_path)
    step({"blast_database": "core_nt"})(["a"], db_path=db_path)
    step({"blast_database": "nt"})(["a"], db_path=db_path)
    step({"blast_database": "core_nt"})(["a"], db_path=db_path)
    assert executed == ["a", "a", "a"]


def test_progress_total_counts_only_the_items_executed(db_path):
    """A resumed run's progress bar ends full: its total is the items this
    call executes, not the cached ones."""

    @save_to_db(
        "test_progress",
        cache=PartialCache(items_kwarg="items", item_key="name"),
    )
    def my_step(items, progress_handler=None):
        for _ in items:
            notify_progress(progress_handler)
        return pd.DataFrame({"name": items})

    my_step(["a", "b"], db_path=db_path)
    progress = {"current": 0, "total": 3}
    my_step(["a", "b", "c"], progress_handler=progress, db_path=db_path)

    assert progress == {"current": 1, "total": 1}


def test_partial_cache_with_extract(db_path):
    """PartialCache with extract function (e.g. SeqRecord → seq_id)."""

    @save_to_db(
        "test_extract",
        cache=PartialCache(
            items_kwarg="records",
            item_key="id",
            extract=lambda r: r["id"],
            output_col="id",
        ),
    )
    def my_step(records):
        return pd.DataFrame(records)

    records = [{"id": "s1", "val": 10}, {"id": "s2", "val": 20}]
    df, _ = my_step(records, db_path=db_path)
    assert len(df) == 2


# --- Empty results: results table is never created, only the inputs table ---


def test_empty_result_is_cached_with_its_columns(db_path):
    """A 0-row result creates no results table; read back, it is not computed
    again and keeps its columns."""
    calls = []

    @save_to_db("test_empty", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        calls.append(t)
        return pd.DataFrame({"x": [], "y": []})

    df1, _ = my_step(t=1, db_path=db_path)
    df2, _ = my_step(t=1, db_path=db_path)
    assert calls == [1]
    assert df1.empty and df2.empty
    assert list(df1.columns) == list(df2.columns) == ["x", "y"]


def test_empty_result_gets_every_column_seen_for_the_step(db_path):
    """An empty result gets every column the step has returned before,
    optional ones included."""

    @save_to_db("test_union", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names):
        rows = [{"name": n, "extra": 1} for n in names if n == "full"]
        return pd.DataFrame(rows, columns=["name", *(["extra"] if rows else [])])

    step(["full"], db_path=db_path)
    step(["empty"], db_path=db_path)
    df, _ = step(["empty"], db_path=db_path)  # cache hit, no rows
    assert df.empty
    assert set(df.columns) == {"name", "extra"}


def test_cached_result_has_the_columns_of_a_live_run(db_path):
    """A result read back has the step's columns in the same order as when it
    ran, though a column empty when its table was created is stored last."""

    @save_to_db(
        "test_col_order", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names):
        return pd.DataFrame({"name": names, "note": [None] * len(names), "n": 1})

    live, _ = step(["a"], db_path=db_path)
    cached, _ = step(["a"], db_path=db_path)
    assert list(live.columns) == list(cached.columns) == ["name", "note", "n"]


def test_cached_item_found_when_stored_as_a_number(db_path):
    """An inputs value stored as a number (by its column's type) still matches
    the item, as the text the cache writes."""
    calls = []

    @save_to_db(
        "test_num",
        cache=PartialCache(items_kwarg="names", item_key="name", params={"k": "k"}),
    )
    def step(names, k=90):
        calls.append(names)
        return pd.DataFrame({"name": names})

    step(["a"], db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE copy AS SELECT name, CAST(k AS INTEGER) AS k, run_id FROM test_num_inputs"
        )
        conn.execute("DROP TABLE test_num_inputs")
        conn.execute("ALTER TABLE copy RENAME TO test_num_inputs")
    step(["a"], db_path=db_path)
    assert calls == [["a"]]


def test_cached_items_read_together_keep_their_types(db_path):
    """Cached items read together come back typed and ordered as reading each
    on its own and joining them: pandas types a column from the values it
    reads, and here items mix every kind SQLite returns."""
    kinds = [None, 7, 2.5, "text", b"bytes"]
    rng = random.Random(0)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE step (a, b, c, run_id INTEGER)")
        for run_id in range(1, 61):
            for _ in range(rng.randint(0, 3)):
                values = [rng.choice(kinds[: rng.randint(1, 5)]) for _ in range(3)]
                conn.execute("INSERT INTO step VALUES (?, ?, ?, ?)", (*values, run_id))
    order = rng.sample(range(1, 61), 60)

    with sqlite3.connect(db_path) as conn:
        alone = [
            pd.read_sql_query("SELECT * FROM step WHERE run_id = ?", conn, params=(i,))
            for i in order
        ]
        together = database._load_cached_items(conn, "step", order)
    expected = pd.concat([df for df in alone if not df.empty], ignore_index=True)
    pd.testing.assert_frame_equal(
        together, expected.drop(columns="run_id"), check_dtype=True
    )


def test_no_items_runs_the_function_for_its_columns(db_path):
    """With an empty input, the function is called directly and its empty
    result keeps its columns."""

    @save_to_db(
        "test_no_items", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names):
        return pd.DataFrame({"name": names, "v": [1] * len(names)})

    df, _ = step([], db_path=db_path)
    assert df.empty
    assert list(df.columns) == ["name", "v"]


def test_empty_run_is_replaced_or_cleared_without_a_results_table(db_path):
    """force_rerun and clear_empty_cache work on an empty run whose results
    table was never created."""

    @save_to_db("test_empty_force", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": []})

    my_step(t=1, db_path=db_path)
    df, _ = my_step(t=1, db_path=db_path, force_rerun=True)
    assert df.empty
    assert clear_empty_cache(db_path, "test_empty_force") == 1


def test_retry_empty_requeries_items_saved_with_only_a_reason(db_path):
    """With result_col, an item whose rows only explain why it has no result
    counts as empty: retry_empty queries it again and replaces its rows."""
    calls = []

    @save_to_db(
        "test_reason_rows",
        cache=PartialCache(items_kwarg="names", item_key="name", result_col="seq"),
    )
    def step(names):
        calls.extend(names)
        return pd.DataFrame(
            {
                "name": names,
                "seq": ["ACGT" if n == "found" else None for n in names],
                "why": [None if n == "found" else "not in BOLD" for n in names],
            }
        )

    step(["found", "absent"], db_path=db_path)
    df, _ = step(["found", "absent"], db_path=db_path, retry_empty=True)

    assert calls == ["found", "absent", "absent"]
    assert sorted(df["name"]) == ["absent", "found"]


# --- Provenance ---


def test_provenance_recorded_once_per_executed_run(db_path):
    """A run records the UTC query date and trident version; a cache hit adds
    no row."""

    @save_to_db("test_prov", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    my_step(t=1, db_path=db_path)
    my_step(t=1, db_path=db_path)
    prov = load_provenance(db_path, "test_prov")
    assert len(prov) == 1
    assert prov["queried_on"].iloc[0] == datetime.now(timezone.utc).date().isoformat()
    assert prov["trident_version"].iloc[0]


def test_provenance_per_item_for_partial_cache(db_path):
    """PartialCache records one provenance row per executed item."""

    @save_to_db(
        "test_prov_partial",
        cache=PartialCache(items_kwarg="items", item_key="name"),
    )
    def my_step(items):
        return pd.DataFrame({"name": items, "v": range(len(items))})

    my_step(["a", "b"], db_path=db_path)
    # Re-run with two cached + one new item: only the new item is executed,
    # but every cached item already has its own provenance row.
    my_step(["a", "b", "c"], db_path=db_path)
    prov = load_provenance(db_path, "test_prov_partial")
    assert len(prov) == 3  # a, b, c each have one row


# --- Failure handling (failure_sink) ---


def test_failed_items_are_retried_and_genuine_empties_cached(db_path):
    """An item reported to failure_sink is not cached (and gets no provenance
    row), so it runs again; an item with no rows that did not fail is cached."""
    calls = []

    @save_to_db("test_fail", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names, failure_sink=None):
        calls.append(list(names))
        failure_sink.extend(n for n in names if n == "bad")
        rows = [{"name": n, "v": 1} for n in names if n == "a"]
        return pd.DataFrame(rows, columns=["name", "v"])

    step(["a", "empty", "bad"], db_path=db_path)
    assert len(load_provenance(db_path, "test_fail")) == 2
    calls.clear()
    step(["a", "empty", "bad"], db_path=db_path)
    assert calls == [["bad"]]


def test_failed_items_exposed_on_result_for_this_call_only(db_path):
    """The result carries this call's failures, with or without a database, so
    the UI can tell them apart from genuine empties; a later call that
    succeeds reports none."""
    failing = {"bad"}

    @save_to_db(
        "test_fail_attrs", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names, failure_sink=None):
        rows = []
        for n in names:
            if n in failing:
                failure_sink.append(n)
                continue
            if n != "empty":
                rows.append({"name": n, "v": 1})
        return pd.DataFrame(rows, columns=["name", "v"])

    df, _ = step(["a", "bad", "empty"], db_path=db_path)
    assert df.attrs["failed_items"] == ["bad"]
    df, _ = step(["a", "bad", "empty"])
    assert df.attrs["failed_items"] == ["bad"]

    failing.clear()  # the source recovers: 'bad' is retried and succeeds
    df, _ = step(["a", "bad", "empty"], db_path=db_path)
    assert df.attrs["failed_items"] == []
    assert set(df["name"]) == {"a", "bad"}


def test_failed_items_leave_no_orphan_rows(db_path):
    """A plain failure value is matched on the first key only, so a sibling
    item that succeeded is not cached either. Its rows must not be saved without an
    inputs row: their run_id would later be reused by another item, which
    would then load them as its own results."""

    def prepare(pairs, **kwargs):
        items = [{"seq_id": s, "query": q} for s, q in pairs]
        return items, {}

    def rebuild(uncached, bound_kwargs):
        keep = {(i["seq_id"], i["query"]) for i in uncached}
        return {
            **bound_kwargs,
            "pairs": [p for p in bound_kwargs["pairs"] if p in keep],
        }

    @save_to_db(
        "test_orphans",
        cache=CustomCache(
            prepare_fn=prepare,
            rebuild_fn=rebuild,
            match_map_dict={"seq_id": "seq_id", "query": "query"},
        ),
    )
    def step(pairs, failure_sink=None):
        rows = []
        for seq_id, query in pairs:
            if (seq_id, query) == ("P1", "A"):
                failure_sink.append("P1")  # P1 errors under query A only
            elif seq_id in ("P1", "P2"):
                rows.append(
                    {"seq_id": seq_id, "query": query, "hit": f"{seq_id}{query}"}
                )
        return pd.DataFrame(rows, columns=["seq_id", "query", "hit"])

    # P1 fails under A, succeeds under B: neither is cached, no row is saved.
    df, _ = step([("P1", "A"), ("P1", "B")], db_path=db_path)
    assert list(df["hit"]) == ["P1B"]  # still returned for this run
    with get_connection(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'test_orphans'"
            ).fetchone()[0]
            == 0
        )

    # New items take run_ids 1 and 2; P3 is a genuine empty.
    step([("P2", "C"), ("P3", "D")], db_path=db_path)
    df, _ = step([("P3", "D")], db_path=db_path)  # served from the cache
    assert df.empty


# --- Saving each item as it finishes (result_sink) ---


def test_item_reported_to_result_sink_survives_a_crash(db_path):
    """An item the step reports as finished, from a worker thread, is saved at
    once: when the step then crashes, also during a forced rerun, the next run
    queries only the other items."""
    queried = []

    @save_to_db("test_sink", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names, crash=False, result_sink=None):
        queried.extend(names)

        def finish(name):
            result_sink(pd.DataFrame({"name": [name], "v": [1]}), [name])

        run_threads(finish, [(names[0],)], n_jobs=2)
        if crash:
            raise RuntimeError("process killed")
        return pd.DataFrame({"name": names, "v": [1] * len(names)})

    step(["a", "b"], db_path=db_path)
    with pytest.raises(RuntimeError):
        step(["a", "b"], crash=True, force_rerun=True, db_path=db_path)

    queried.clear()
    df, _ = step(["a", "b"], db_path=db_path)
    assert queried == ["b"]
    assert sorted(df["name"]) == ["a", "b"]


def test_items_saved_alone_come_back_with_every_column_of_the_step(db_path):
    """An item saved on its own has only its own columns (a check without a
    hit has no hit columns): read back, it gets every column the step
    returned."""

    @save_to_db(
        "test_sink_cols", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names, result_sink=None):
        for name in names:
            result_sink(pd.DataFrame({"name": [name], "hit_found": [False]}), [name])
        return pd.DataFrame(
            {"name": names, "hit_found": False, "query_cover": float("nan")}
        )

    step(["a"], db_path=db_path)
    df, _ = step(["a"], db_path=db_path)  # served from the cache
    assert "query_cover" in df.columns


def test_failed_save_leaves_the_step_running_and_no_stray_rows(db_path, monkeypatch):
    """A save that fails after writing the rows does not stop the step: its
    item is saved when the step returns, and the rows the failed save left are
    deleted by the next call."""
    real_write_provenance = database._write_provenance
    fail_once = [True]

    def flaky_write_provenance(*args):
        if fail_once.pop() if fail_once else False:
            raise sqlite3.OperationalError("database is locked")
        real_write_provenance(*args)

    monkeypatch.setattr(database, "_write_provenance", flaky_write_provenance)

    @save_to_db(
        "test_sink_fail", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names, result_sink=None):
        for name in names:
            result_sink(pd.DataFrame({"name": [name], "v": [1]}), [name])
        return pd.DataFrame({"name": names, "v": [1] * len(names)})

    df, _ = step(["a"], db_path=db_path)
    assert df["name"].tolist() == ["a"]
    step(["b"], db_path=db_path)

    df, _ = step(["a", "b"], db_path=db_path)  # served from the cache
    assert sorted(df["name"]) == ["a", "b"]
    with get_connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM test_sink_fail").fetchone()[0] == 2


def test_rows_match_their_item_when_a_number_column_became_float(db_path):
    """pandas turns an int column with a missing value into floats: the row of
    extent 500.0 is still saved as the result of the item of extent 500."""

    @save_to_db(
        "test_float_key", cache=PartialCache(items_kwarg="extents", item_key="extent")
    )
    def step(extents):
        return pd.DataFrame({"extent": [float(e) for e in extents], "n": [3]})

    step([500], db_path=db_path)
    df, _ = step([500], db_path=db_path)  # served from the cache
    assert df["n"].tolist() == [3]


def test_column_empty_in_the_first_saved_rows_keeps_later_values(db_path):
    """A column with no value in the rows that create the table does not take
    a type from them: an id saved later comes back as the same int."""
    ids = {"not found": None, "found": 126160}

    @save_to_db("test_types", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names):
        return pd.DataFrame({"name": names, "aphia_id": [ids[n] for n in names]})

    step(["not found"], db_path=db_path)
    step(["found"], db_path=db_path)
    df, _ = step(["found"], db_path=db_path)  # served from the cache
    assert df["aphia_id"].tolist() == [126160]


# --- Fingerprint / content-based invalidation ---


def test_fingerprint_order_independent_and_value_sensitive():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = df1.iloc[::-1].reset_index(drop=True)  # same rows, reversed
    df3 = pd.DataFrame({"a": [1, 3], "b": ["x", "y"]})  # a value changed
    assert fingerprint(df1) == fingerprint(df2)  # row order does not matter
    # nor column order: a result read back from the database may have its
    # columns in another order than the step returned them
    assert fingerprint(df1) == fingerprint(df1[["b", "a"]])
    assert fingerprint(df1) != fingerprint(df3)  # value change flips it
    assert fingerprint(pd.DataFrame()) == "empty"
    assert fingerprint(None) == "empty"


def test_fullcache_reruns_on_upstream_data_change(db_path):
    """A FullCache step with fingerprint_on re-runs when its input data changes."""
    calls = []

    @save_to_db(
        "test_fp",
        cache=FullCache(local={"t": "threshold"}, fingerprint_on=["src_df"]),
    )
    def step(src_df, threshold=1):
        calls.append(len(src_df))
        return pd.DataFrame({"n": [len(src_df)]})

    src = pd.DataFrame({"x": [1, 2]})
    step(src, threshold=1, db_path=db_path)
    step(src, threshold=1, db_path=db_path)  # same data + params -> cached
    assert calls == [2]

    step(pd.DataFrame({"x": [1, 2, 3]}), threshold=1, db_path=db_path)  # data changed
    assert calls == [2, 3]  # re-ran


def test_fingerprint_propagates_via_inherited_params(db_path):
    """An upstream data change cascades to a downstream step via inherited params."""

    @save_to_db("up", cache=FullCache(local={"t": "t"}, fingerprint_on=["src_df"]))
    def up(src_df, t=1):
        return pd.DataFrame({"v": [len(src_df)]})

    down_calls = []

    @save_to_db("down", cache=FullCache(inherit_from=["up_params"]))
    def down(data, up_params=None):
        down_calls.append(1)
        return pd.DataFrame({"d": [1]})

    df, up_params = up(pd.DataFrame({"x": [1]}), db_path=db_path)
    down(df, up_params=up_params, db_path=db_path)

    # Upstream data grows -> up's params carry a new fingerprint -> down re-runs
    df2, up_params2 = up(pd.DataFrame({"x": [1, 2]}), db_path=db_path)
    down(df2, up_params=up_params2, db_path=db_path)
    assert len(down_calls) == 2


# --- Schema version ---


def test_database_is_stamped_with_its_schema_version_and_accepted(db_path):
    @save_to_db("test_sv", cache=FullCache(local={"t": "t"}))
    def step(t=1):
        return pd.DataFrame({"x": [t]})

    step(t=1, db_path=db_path)
    with get_connection(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert check_schema_version(db_path) == (True, "")


def test_newer_database_is_refused_and_unversioned_one_accepted(db_path):
    """A database from a newer TRIDENT is refused; one made before versioning
    (user_version 0) is accepted."""
    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE sequences (seq_id TEXT)")
    assert check_schema_version(db_path)[0]

    with get_connection(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    compatible, message = check_schema_version(db_path)
    assert not compatible
    assert "newer" in message.lower()


def test_load_provenance_missing_table(db_path):
    """Databases without a provenance table return an empty frame, not an error."""
    # No step has run, so no provenance table exists yet.
    result = load_provenance(db_path)
    assert result.empty
    assert list(result.columns) == [
        "step",
        "run_id",
        "queried_on",
        "trident_version",
    ]


def test_rows_of_a_first_save_killed_early_go_to_no_item(db_path):
    """A first save killed before its inputs table exists leaves rows of no
    item: the next item saved does not take them over with their run_id."""

    @save_to_db("test_first", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names):
        return pd.DataFrame({"name": names, "v": [1] * len(names)})

    with sqlite3.connect(db_path) as conn:
        pd.DataFrame({"name": ["a"], "v": [9], "run_id": [1]}).to_sql(
            "test_first", conn, index=False
        )
    conn.close()
    step(["b"], db_path=db_path)
    df, _ = step(["b"], db_path=db_path)
    assert df[["name", "v"]].values.tolist() == [["b", 1]]
