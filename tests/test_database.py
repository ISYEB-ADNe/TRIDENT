"""Tests for trident.core.database — save_to_db decorator and cache strategies."""

import pandas as pd
import pytest

from datetime import datetime, timezone

from trident.core.database import (
    FullCache,
    PartialCache,
    save_to_db,
    view_cached_runs,
    clear_cache,
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


def test_full_cache_miss_then_hit(db_path):
    """First call executes, second call returns cached data."""

    @save_to_db("test_full", cache=FullCache(local={"threshold": "threshold"}))
    def my_step(data, threshold=90):
        return pd.DataFrame({"x": data})

    df1, params1 = my_step([1, 2, 3], threshold=90, db_path=db_path)
    assert len(df1) == 3
    assert params1 == {"threshold": 90}

    # Second call: same params → cached
    df2, params2 = my_step([99, 99], threshold=90, db_path=db_path)
    assert len(df2) == 3  # cached result, not the new input
    assert params2 == {"threshold": 90}


def test_full_cache_different_params(db_path):
    """Different params → cache miss → re-executes."""
    call_count = 0

    @save_to_db("test_full2", cache=FullCache(local={"t": "threshold"}))
    def my_step(threshold=90):
        nonlocal call_count
        call_count += 1
        return pd.DataFrame({"val": [threshold]})

    my_step(threshold=90, db_path=db_path)
    my_step(threshold=95, db_path=db_path)
    assert call_count == 2


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
    assert params == {"t": 5}


def test_full_cache_inherit_from(db_path):
    """FullCache inherits upstream params into the cache key."""

    @save_to_db(
        "test_inherit",
        cache=FullCache(inherit_from=["upstream_params"], local={"t": "threshold"}),
    )
    def my_step(threshold=90, upstream_params=None):
        return pd.DataFrame({"val": [threshold]})

    df1, p1 = my_step(threshold=90, upstream_params={"a": 1}, db_path=db_path)
    # Same threshold but different upstream → cache miss
    df2, p2 = my_step(threshold=90, upstream_params={"a": 2}, db_path=db_path)
    # Params should include both upstream and local
    assert "a" in p2
    assert "t" in p2


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


# --- view_cached_runs / clear_cache ---


def test_view_and_clear_cache(db_path):
    @save_to_db("test_view", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    my_step(t=1, db_path=db_path)
    runs = view_cached_runs(db_path, "test_view")
    assert len(runs) == 1
    assert runs["row_count"].iloc[0] == 1

    clear_cache(db_path, "test_view")
    runs_after = view_cached_runs(db_path, "test_view")
    assert runs_after.empty


# --- Empty results: results table is never created, only the inputs table ---


def test_empty_result_cached_then_hit(db_path):
    """A 0-row result caches its input run_id but creates no results table.

    The second (cached) call must not crash on the missing results table.
    """
    calls = []

    @save_to_db("test_empty", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        calls.append(t)
        return pd.DataFrame({"x": []})

    df1, _ = my_step(t=1, db_path=db_path)
    assert df1.empty

    # Second call hits the cache: must return empty, not re-execute, not crash.
    df2, _ = my_step(t=1, db_path=db_path)
    assert df2.empty
    assert calls == [1]  # executed once, second call served from cache


def test_empty_result_force_rerun(db_path):
    """force_rerun replaces an empty cached run without crashing on the missing table."""

    @save_to_db("test_empty_force", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": []})

    my_step(t=1, db_path=db_path)
    df2, _ = my_step(t=1, db_path=db_path, force_rerun=True)
    assert df2.empty


def test_clear_empty_cache_without_results_table(db_path):
    """retry_empty must clear empty entries even when the results table was never created."""

    @save_to_db("test_empty_clear", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": []})

    my_step(t=1, db_path=db_path)
    cleared = clear_empty_cache(db_path, "test_empty_clear")
    assert cleared == 1


def test_view_cached_runs_with_empty_result(db_path):
    """view_cached_runs lists empty runs (row_count 0) even with no results table."""

    @save_to_db("test_empty_view", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": []})

    my_step(t=1, db_path=db_path)
    runs = view_cached_runs(db_path, "test_empty_view")
    assert len(runs) == 1
    assert runs["row_count"].iloc[0] == 0


# --- Provenance ---


def test_provenance_recorded_on_run(db_path):
    """Each executed run records the UTC query date and trident version."""

    @save_to_db("test_prov", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    my_step(t=1, db_path=db_path)
    prov = load_provenance(db_path, "test_prov")
    assert len(prov) == 1
    assert prov["queried_on"].iloc[0] == datetime.now(timezone.utc).date().isoformat()
    assert prov["trident_version"].iloc[0]  # non-empty version string


def test_provenance_not_duplicated_on_cache_hit(db_path):
    """A cache hit must not add or change provenance rows."""

    @save_to_db("test_prov_hit", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    my_step(t=1, db_path=db_path)
    my_step(t=1, db_path=db_path)  # cached
    prov = load_provenance(db_path, "test_prov_hit")
    assert len(prov) == 1


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


def test_provenance_cleared_with_cache(db_path):
    """Clearing a cache also drops its provenance rows."""

    @save_to_db("test_prov_clear", cache=FullCache(local={"t": "t"}))
    def my_step(t=1):
        return pd.DataFrame({"x": [t]})

    my_step(t=1, db_path=db_path)
    clear_cache(db_path, "test_prov_clear")
    assert load_provenance(db_path, "test_prov_clear").empty


# --- Failure handling (failure_sink) ---


def test_failed_items_not_cached_and_retried(db_path):
    """Items reported via failure_sink are not cached, so they re-run next time."""
    calls = []

    @save_to_db("test_fail", cache=PartialCache(items_kwarg="names", item_key="name"))
    def step(names, failure_sink=None):
        calls.append(list(names))
        rows = []
        for n in names:
            if n == "bad":  # simulate a failed query
                if failure_sink is not None:
                    failure_sink.append(n)
                continue
            rows.append({"name": n, "v": 1})
        return pd.DataFrame(rows, columns=["name", "v"])

    step(["a", "bad"], db_path=db_path)
    cached_names = set(view_cached_runs(db_path, "test_fail")["name"])
    assert cached_names == {"a"}  # 'bad' was not cached

    # Re-run: 'a' is cached, only 'bad' re-executes
    calls.clear()
    step(["a", "bad"], db_path=db_path)
    assert calls == [["bad"]]


def test_genuine_empty_is_cached_not_retried(db_path):
    """A 0-row result that is NOT a failure stays cached and is not re-run."""
    calls = []

    @save_to_db(
        "test_empty_ok", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names, failure_sink=None):
        calls.append(list(names))
        # 'empty' returns no rows but is a genuine empty (not added to sink)
        rows = [{"name": n, "v": 1} for n in names if n != "empty"]
        return pd.DataFrame(rows, columns=["name", "v"])

    step(["x", "empty"], db_path=db_path)
    # Both cached: 'x' with a row, 'empty' as a 0-row entry
    calls.clear()
    step(["x", "empty"], db_path=db_path)
    assert calls == []  # nothing re-executed


def test_failed_items_not_provenanced(db_path):
    """Failed items get no provenance row (they were not cached)."""

    @save_to_db(
        "test_fail_prov", cache=PartialCache(items_kwarg="names", item_key="name")
    )
    def step(names, failure_sink=None):
        rows = []
        for n in names:
            if n == "bad":
                if failure_sink is not None:
                    failure_sink.append(n)
                continue
            rows.append({"name": n, "v": 1})
        return pd.DataFrame(rows, columns=["name", "v"])

    step(["a", "bad"], db_path=db_path)
    prov_names = set(view_cached_runs(db_path, "test_fail_prov")["name"])
    assert "bad" not in prov_names
    prov = load_provenance(db_path, "test_fail_prov")
    assert len(prov) == 1  # only 'a'


# --- Fingerprint / content-based invalidation ---


def test_fingerprint_order_independent_and_value_sensitive():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = df1.iloc[::-1].reset_index(drop=True)  # same rows, reversed
    df3 = pd.DataFrame({"a": [1, 3], "b": ["x", "y"]})  # a value changed
    assert fingerprint(df1) == fingerprint(df2)  # row order does not matter
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


def test_schema_version_stamped_on_write(db_path):
    @save_to_db("test_sv", cache=FullCache(local={"t": "t"}))
    def step(t=1):
        return pd.DataFrame({"x": [t]})

    step(t=1, db_path=db_path)
    with get_connection(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_check_schema_version_current_ok(db_path):
    @save_to_db("test_sv2", cache=FullCache(local={"t": "t"}))
    def step(t=1):
        return pd.DataFrame({"x": [t]})

    step(t=1, db_path=db_path)
    compatible, message = check_schema_version(db_path)
    assert compatible
    assert message == ""


def test_check_schema_version_newer_refused(db_path):
    @save_to_db("test_sv3", cache=FullCache(local={"t": "t"}))
    def step(t=1):
        return pd.DataFrame({"x": [t]})

    step(t=1, db_path=db_path)
    with get_connection(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    compatible, message = check_schema_version(db_path)
    assert not compatible
    assert "newer" in message.lower()


def test_check_schema_version_legacy_accepted(db_path):
    """A pre-versioning database (user_version 0) is still accepted."""
    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE sequences (seq_id TEXT)")  # leaves user_version 0
    compatible, _ = check_schema_version(db_path)
    assert compatible


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
