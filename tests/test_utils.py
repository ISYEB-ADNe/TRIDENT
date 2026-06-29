"""Tests for trident.core.utils — decorators and helper functions."""

import pandas as pd

from trident.core.utils import (
    extract_specific_epithet,
    find_exclusion_pipeline_step,
    group_species_by_flag,
    notify_progress,
    preserve_sequence_order,
    reorder_taxonomy_columns,
    ensure_columns,
    top_hit_per_group,
    normalize_name,
)


# --- normalize_name ---


def test_normalize_name_casefold_and_whitespace():
    assert normalize_name("Gadus morhua") == "gadus morhua"
    assert normalize_name("  Gadus   morhua  ") == "gadus morhua"
    assert normalize_name("GADUS MORHUA") == "gadus morhua"


def test_normalize_name_passes_through_non_strings():
    import math

    assert normalize_name(None) is None
    nan = float("nan")
    assert math.isnan(normalize_name(nan))  # NaN stays NaN -> won't spuriously match


# --- ensure_columns ---


def test_ensure_columns_adds_missing_and_keeps_existing():
    df = pd.DataFrame({"a": [1, 2]})
    out = ensure_columns(df, ["a", "b", "c"])
    assert list(out.columns) == ["a", "b", "c"]
    assert out["a"].tolist() == [1, 2]  # existing untouched
    assert out["b"].isna().all()


def test_ensure_columns_custom_fill():
    df = ensure_columns(pd.DataFrame({"a": [1]}), ["b"], fill=0)
    assert df["b"].tolist() == [0]


# --- top_hit_per_group ---


def _hits():
    return pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s1", "s2"],
            "scientificName": ["A", "A", "B", "C"],
            "identity_percentage": [90.0, 99.0, 95.0, 80.0],
            "query_cover": [100, 50, 100, 100],
        }
    )


def test_top_hit_per_group_keeps_best_per_group():
    out = top_hit_per_group(
        _hits(),
        keys=["seq_id", "scientificName"],
        sort=["identity_percentage", "query_cover"],
    )
    # (s1, A) -> the 99.0 row; one row per (seq_id, scientificName)
    assert len(out) == 3
    a = out[(out.seq_id == "s1") & (out.scientificName == "A")]
    assert a["identity_percentage"].iloc[0] == 99.0


def test_top_hit_per_group_matches_sort_drop_duplicates():
    df = _hits()
    expected = df.sort_values(
        ["identity_percentage", "query_cover"], ascending=False
    ).drop_duplicates(subset=["seq_id", "scientificName"])
    out = top_hit_per_group(
        df,
        keys=["seq_id", "scientificName"],
        sort=["identity_percentage", "query_cover"],
    )
    pd.testing.assert_frame_equal(out, expected)


def test_top_hit_per_group_top_n_and_columns():
    out = top_hit_per_group(
        _hits(),
        keys=["seq_id"],
        sort=["identity_percentage", "query_cover"],
        n=2,
        columns=["seq_id", "identity_percentage"],
    )
    assert list(out.columns) == ["seq_id", "identity_percentage"]
    # s1 has 3 rows -> top 2 by identity (99, 95); s2 has 1
    s1 = out[out.seq_id == "s1"]["identity_percentage"].tolist()
    assert s1 == [99.0, 95.0]


# --- notify_progress ---


def test_notify_progress_dict():
    handler = {"current": 0, "total": 10}
    notify_progress(handler, n=3)
    assert handler["current"] == 3


def test_notify_progress_none():
    notify_progress(None)  # should not raise


def test_notify_progress_tqdm_like():
    class FakeTqdm:
        def __init__(self):
            self.n = 0

        def update(self, n):
            self.n += n

    t = FakeTqdm()
    notify_progress(t, n=5)
    assert t.n == 5


# --- extract_specific_epithet ---


def test_extract_epithet_normal():
    assert extract_specific_epithet("Gadus morhua", "Gadus") == "morhua"


def test_extract_epithet_genus_mismatch():
    assert extract_specific_epithet("Gadus morhua", "Zeus") is None


def test_extract_epithet_empty():
    assert extract_specific_epithet(None, "Gadus") is None
    assert extract_specific_epithet("Gadus morhua", None) is None
    assert extract_specific_epithet("", "") is None


# --- reorder_taxonomy_columns ---


def test_reorder_taxonomy_columns():
    df = pd.DataFrame({"seq_id": ["a"], "genus": ["G"], "family": ["F"], "other": [1]})
    result = reorder_taxonomy_columns(df)
    # family should come before genus in taxonomy order
    cols = list(result.columns)
    assert cols.index("family") < cols.index("genus")
    # seq_id stays first, other stays after taxonomy
    assert cols[0] == "seq_id"
    assert "other" in cols


def test_reorder_no_taxonomy_cols():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = reorder_taxonomy_columns(df)
    assert list(result.columns) == ["a", "b"]


# --- group_species_by_flag ---


def test_group_species_by_flag():
    df = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s2"],
            "scientificName": ["A", "B", "C"],
            "in_mol": [True, False, True],
        }
    )
    result = group_species_by_flag(df, "in_mol")
    assert result == {"s1": ["A"], "s2": ["C"]}


# --- find_exclusion_pipeline_step ---


def test_find_exclusion_step():
    all_ids = ["s1", "s2", "s3"]
    steps = [
        ("MOL Search", {"s1", "s2", "s3"}),
        ("MOL Filter", {"s1", "s2"}),
        ("GEO", {"s1"}),
    ]
    result = find_exclusion_pipeline_step(all_ids, steps)
    assert len(result) == 2
    assert result[result["seq_id"] == "s3"]["pipeline_step"].iloc[0] == "MOL Filter"
    assert result[result["seq_id"] == "s2"]["pipeline_step"].iloc[0] == "GEO"


def test_find_exclusion_step_none_excluded():
    result = find_exclusion_pipeline_step(
        ["s1"], [("step1", {"s1"}), ("step2", {"s1"})]
    )
    assert result.empty


# --- preserve_sequence_order ---


def test_preserve_order_basic():
    @preserve_sequence_order("seq_id", "input_df")
    def scramble(input_df):
        return input_df.sort_values("val").reset_index(drop=True)

    df = pd.DataFrame({"seq_id": ["b", "a", "c"], "val": [2, 1, 3]})
    result = scramble(df)
    assert list(result["seq_id"]) == ["b", "a", "c"]


def test_preserve_order_with_duplicates():
    @preserve_sequence_order("seq_id", "input_df")
    def identity(input_df):
        return input_df.copy()

    df = pd.DataFrame({"seq_id": ["b", "a", "b", "a"], "val": [1, 2, 3, 4]})
    result = identity(df)
    assert list(result["seq_id"]) == ["b", "b", "a", "a"]


def test_preserve_order_empty_result():
    @preserve_sequence_order("seq_id", "input_df")
    def empty(input_df):
        return pd.DataFrame(columns=input_df.columns)

    df = pd.DataFrame({"seq_id": ["a"], "val": [1]})
    result = empty(df)
    assert result.empty


def test_preserve_order_missing_column():
    """Gracefully passes through when column is absent."""

    @preserve_sequence_order("seq_id", "input_df")
    def identity(input_df):
        return input_df.copy()

    df = pd.DataFrame({"other": [1, 2]})
    result = identity(df)
    assert list(result["other"]) == [1, 2]
