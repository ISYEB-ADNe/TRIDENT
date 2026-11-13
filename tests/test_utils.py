"""Tests for trident.core.utils, one per helper."""

import math

import pandas as pd
from loguru import logger

from trident import logging as trident_logging
from trident.logging import setup_logging
from trident.core.utils import (
    StepLog,
    extract_specific_epithet,
    notify_progress,
    preserve_sequence_order,
    ensure_columns,
    top_hit_per_group,
    normalize_name,
    to_binomial,
)


def test_names_match_ignoring_case_and_spacing_and_missing_names_match_nothing():
    assert normalize_name("  GADUS   morhua  ") == "gadus morhua"
    assert normalize_name(None) is None
    assert math.isnan(normalize_name(float("nan")))


def test_ensure_columns_adds_missing_columns_filled_and_keeps_existing():
    out = ensure_columns(pd.DataFrame({"a": [1, 2]}), ["a", "b", "c"])
    assert list(out.columns) == ["a", "b", "c"]
    assert out["a"].tolist() == [1, 2]
    assert out[["b", "c"]].isna().all().all()
    assert ensure_columns(pd.DataFrame({"a": [1]}), ["b"], fill=0)["b"].tolist() == [0]


def test_top_hit_per_group_keeps_the_best_rows_of_each_group():
    """Same result as sorting then dropping duplicates; with n, the n best
    rows of each group, restricted to the columns asked."""
    df = pd.DataFrame(
        {
            "seq_id": ["s1", "s1", "s1", "s2"],
            "scientificName": ["A", "A", "B", "C"],
            "identity_percentage": [90.0, 99.0, 95.0, 80.0],
            "query_cover": [100, 50, 100, 100],
        }
    )
    sort = ["identity_percentage", "query_cover"]
    expected = df.sort_values(sort, ascending=False).drop_duplicates(
        subset=["seq_id", "scientificName"]
    )
    out = top_hit_per_group(df, keys=["seq_id", "scientificName"], sort=sort)
    pd.testing.assert_frame_equal(out, expected)

    out = top_hit_per_group(
        df, keys=["seq_id"], sort=sort, n=2, columns=["seq_id", "identity_percentage"]
    )
    assert list(out.columns) == ["seq_id", "identity_percentage"]
    assert out["identity_percentage"].tolist() == [99.0, 95.0, 80.0]


def test_progress_reaches_a_dict_or_tqdm_handler_and_none_is_ignored():
    handler = {"current": 0, "total": 10}
    notify_progress(handler, n=3)
    assert handler["current"] == 3

    class FakeTqdm:
        n = 0

        def update(self, n):
            self.n += n

    bar = FakeTqdm()
    notify_progress(bar, n=5)
    assert bar.n == 5
    notify_progress(None)


def test_epithet_only_when_the_name_starts_with_the_genus():
    assert extract_specific_epithet("Gadus morhua", "Gadus") == "morhua"
    assert extract_specific_epithet("Gadus morhua", "Zeus") is None
    assert extract_specific_epithet(None, "Gadus") is None
    assert extract_specific_epithet("Gadus morhua", None) is None
    assert extract_specific_epithet("", "") is None


def test_step_output_follows_the_input_sequence_order():
    """Rows are grouped by sequence in the order of the input, duplicates
    included; an empty result or one without the column passes through. A step
    returning its input unchanged leaves the caller's frame as it was."""

    @preserve_sequence_order("seq_id", "input_df")
    def scramble(input_df):
        return input_df.sort_values("val").reset_index(drop=True)

    df = pd.DataFrame({"seq_id": ["b", "a", "b", "a"], "val": [4, 1, 3, 2]})
    assert scramble(df)["seq_id"].tolist() == ["b", "b", "a", "a"]

    @preserve_sequence_order("seq_id", "input_df")
    def identity(input_df):
        return input_df

    identity(df)
    assert df.columns.tolist() == ["seq_id", "val"]

    @preserve_sequence_order("seq_id", "input_df")
    def empty(input_df):
        return pd.DataFrame(columns=input_df.columns)

    assert empty(df).empty
    assert scramble(pd.DataFrame({"val": [2, 1]}))["val"].tolist() == [1, 2]


def test_to_binomial_keeps_the_species_a_qualifier_refers_to():
    """Two different 'cf.' comparisons stay two names; 'sp.' codes and
    subspecies are dropped."""
    assert to_binomial("Lutjanus cf. russellii") == "Lutjanus cf. russellii"
    assert (
        to_binomial("Andinoacara cf. pulcher 'Venezuela'") == "Andinoacara cf. pulcher"
    )
    assert to_binomial("Rattus sp. NH 2147") == "Rattus sp."
    assert to_binomial("Odobenus rosmarus rosmarus") == "Odobenus rosmarus"


def test_step_log_shows_progress_between_items_and_counts_them_all_at_the_end(
    monkeypatch,
):
    """A slow item (BOLD, about 10 s each) is shown as itself; quick items go to
    DEBUG, and the next line shown after them is a progress line saying how
    many are done. A notable item is always shown as itself, and the end line
    counts every item."""
    # Seconds: StepLog's start, then each item, then the end line.
    clock = iter([0, 0, 3, 6, 9, 12, 16, 25, 26, 32, 33])
    monkeypatch.setattr(StepLog, "clock", staticmethod(lambda: next(clock)))
    lines = []
    sink = logger.add(
        lambda m: lines.append((m.record["level"].name, m.record["message"]))
    )
    logger.enable("trident")  # as the app does (the package disables it)
    log = StepLog("BOLD search", 9, "species")
    try:
        log.item("Gadus morhua", "45 BOLD sequences")
        log.item("Gadus ogac", "12 BOLD sequences")
        log.item("Gadus nobody", "not in BOLD", notable=True)
        log.item("Gadus macrocephalus", "8 BOLD sequences")
        log.item("Gadus minor", "2 BOLD sequences")
        log.item("Gadus quick", "1 BOLD sequence")  # 10 s after the last shown
        log.item("Gadus chalcogrammus", "3 BOLD sequences")  # 9 s on its own
        log.item("Gadus fast", "4 BOLD sequences")
        log.item("Gadus slow", "5 BOLD sequences")  # slow, after a hidden one
        log.done("80 BOLD sequences for 8 species")
    finally:
        logger.remove(sink)
        logger.disable("trident")
    assert lines[:9] == [
        ("INFO", "BOLD search 1/9: Gadus morhua, 45 BOLD sequences"),
        ("DEBUG", "BOLD search 2/9: Gadus ogac, 12 BOLD sequences"),
        ("INFO", "BOLD search 3/9: Gadus nobody, not in BOLD"),
        ("DEBUG", "BOLD search 4/9: Gadus macrocephalus, 8 BOLD sequences"),
        ("DEBUG", "BOLD search 5/9: Gadus minor, 2 BOLD sequences"),
        (
            "INFO",
            "BOLD search: 6 of 9 species done (last: Gadus quick, 1 BOLD sequence)",
        ),
        ("INFO", "BOLD search 7/9: Gadus chalcogrammus, 3 BOLD sequences"),
        ("DEBUG", "BOLD search 8/9: Gadus fast, 4 BOLD sequences"),
        (
            "INFO",
            "BOLD search: 9 of 9 species done (last: Gadus slow, 5 BOLD sequences)",
        ),
    ]
    level, end = lines[9]
    assert level == "INFO" and end.endswith(
        ": 9 of 9 species queried, 80 BOLD sequences for 8 species"
    )


def test_changing_the_log_level_keeps_the_other_log_handlers(monkeypatch):
    """Setting the level again replaces only the console handler: a step's
    log in the app, added elsewhere, keeps receiving, also when the console
    handler was already removed (a notebook's logger.remove())."""
    # Leave the session's handlers alone, and its logging state as it was.
    monkeypatch.setattr(trident_logging, "_configured", True)
    monkeypatch.setattr(trident_logging, "_handler_ids", [])
    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]))
    try:
        setup_logging("INFO")
        logger.remove(trident_logging._handler_ids[0])
        setup_logging("DEBUG")
        logger.info("still received")
    finally:
        for handler_id in trident_logging._handler_ids:
            logger.remove(handler_id)
        logger.remove(sink)
        logger.disable("trident")
    assert lines == ["still received"]
