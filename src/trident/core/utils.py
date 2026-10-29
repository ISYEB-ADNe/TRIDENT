import contextlib
from collections.abc import Sequence
from functools import wraps
import inspect
import re
import threading
import time
from typing import Callable

import pandas as pd
from loguru import logger

from trident.core.http import request_label


##### MISCELLANEOUS UTILITIES #####


def normalize_name(name):
    """Normalize a taxonomic name for joining: collapse whitespace + casefold.

    Used only as a join key, never to replace the displayed name. Non-strings
    (e.g. NaN) are returned unchanged so they do not spuriously match each other.
    """
    if not isinstance(name, str):
        return name
    return " ".join(name.split()).casefold()


# Open-nomenclature qualifiers that refer to a species: 'Genus cf. epithet'.
_SPECIES_QUALIFIERS = {"cf.", "aff."}


def drop_subgenus(name: str) -> str:
    """The name without its parts in parentheses (a subgenus, or an NCBI
    qualifier such as '(in: foraminifera)'), as GBIF and BOLD write it:
    'Phascolosoma (Phascolosoma) arcuatum' -> 'Phascolosoma arcuatum'."""
    return " ".join(re.sub(r"\([^)]*\)", " ", name).split())


def to_binomial(name) -> str:
    """Genus and species epithet of a taxon name (its first two words).

    Subspecies or strain suffixes and a subgenus in parentheses are dropped,
    e.g. 'Octopus (Amphioctopus) minor ssp' -> 'Octopus minor'. A 'cf.' or
    'aff.' qualifier keeps the species it refers to ('Lutjanus cf. russellii'),
    so two different comparisons stay two names; 'Genus sp. CODE' gives
    'Genus sp.'. A one-word name is returned as is; an empty or missing name
    (None, NaN) gives ''.
    """
    words = drop_subgenus(name).split() if isinstance(name, str) else []
    n_words = 3 if len(words) > 2 and words[1] in _SPECIES_QUALIFIERS else 2
    return " ".join(words[:n_words])


def as_ids(values: pd.Series) -> pd.Series:
    """Numeric ids (e.g. AphiaIDs) as nullable integers, whatever type they were
    stored as (SQLite gives int, float or text)."""
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def ensure_columns(
    df: pd.DataFrame, cols: "list[str] | tuple[str, ...]", fill=pd.NA
) -> pd.DataFrame:
    """Add any of *cols* missing from *df*, filled with *fill*. Mutates and returns df."""
    for col in cols:
        if col not in df.columns:
            df[col] = fill
    return df


def top_hit_per_group(
    df: pd.DataFrame,
    keys: "list[str]",
    sort: "list[str]",
    ascending: bool = False,
    n: int = 1,
    columns: "list[str] | None" = None,
    cover_first: float | None = None,
) -> pd.DataFrame:
    """Keep the top *n* rows per *keys* group, ordered by *sort* (descending by default).

    Row order follows the sort (not the group keys). With n == 1 this is
    sort + drop_duplicates(keep first); with n > 1 it is sort + groupby.head(n).
    With *cover_first*, rows whose query_cover reaches it rank before the
    others, so a short fragment is the top row only when the group has no
    longer one. Optionally narrow to *columns*.
    """
    if cover_first is not None:
        cover = pd.to_numeric(df["query_cover"], errors="coerce")
        df = df.assign(_full_cover=cover >= cover_first)
        sort = ["_full_cover", *sort]
    ranked = df.sort_values(sort, ascending=ascending)
    if n == 1:
        top = ranked.drop_duplicates(subset=keys)
    else:
        top = ranked.groupby(keys, sort=False, observed=False).head(n)
    if columns is not None:
        top = top[columns]
    return top


def unbeaten_hits(df: pd.DataFrame, keys: "list[str]", n: int = 1) -> pd.DataFrame:
    """The hits of each *keys* group that no other hit of the group beats on
    both identity_percentage and query_cover (at least as high on both, higher
    on one), with at most *n* hits of the same identity and cover.

    Whatever minimum identity and cover a filter applies later, the best hit
    passing it is among these: a short high-identity fragment cannot push out
    a full-length hit, as a top N by identity would. Rows without a group key
    are left out, as in a groupby.
    """
    scores = ["identity_percentage", "query_cover"]
    points = (
        df[[*keys, *scores]]
        .dropna(subset=keys)
        .drop_duplicates()
        .sort_values(scores, ascending=False)
    )
    # Sorted by identity, a point is beaten when an earlier one of its group
    # (higher identity, or the same with a higher cover) covers at least as much.
    best_cover = points.groupby(keys, sort=False)["query_cover"].cummax()
    before = best_cover.groupby([points[k] for k in keys], sort=False).shift()
    unbeaten = points[before.isna() | (points["query_cover"] > before)]
    kept = df.merge(unbeaten, on=[*keys, *scores])
    return kept.groupby([*keys, *scores], sort=False).head(n).reset_index(drop=True)


# Plurals that are not the word + "s", so each is written once.
_PLURALS = {
    "batch": "batches",
    "genus": "genera",
    "is": "are",
    "query": "queries",
    "search": "searches",
    "species": "species",
    "taxon": "taxa",
}


def pluralize(n: int, one: str, many: str | None = None) -> str:
    """The word for *n* items: *one* for exactly 1, else its plural
    (``_PLURALS``, or *one* + "s"), e.g. ``f"{n} {pluralize(n, 'genus')}"``.
    A word of several takes the plural on its last one ("BLAST search" ->
    "BLAST searches"); *many* gives it when that rule cannot ("names or
    genera")."""
    if n == 1:
        return one
    if many:
        return many
    head, _, last = one.rpartition(" ")
    plural = _PLURALS.get(last, last + "s")
    return f"{head} {plural}" if head else plural


_progress_lock = threading.Lock()


def set_progress_total(handler: dict | object | None, total: int) -> None:
    """Set the total of a progress tracker: the ``"total"`` key of a dict
    tracker, the ``total`` attribute of a tqdm-like one. No-op otherwise."""
    with _progress_lock:
        if isinstance(handler, dict) and "total" in handler:
            handler["total"] = total
        elif hasattr(handler, "total") and hasattr(handler, "refresh"):
            handler.total = total
            handler.refresh()


def notify_progress(handler: dict | object | None, n: int = 1) -> None:
    """Increment a progress tracker by n steps.

    Supports dict-based trackers (Streamlit: handler["current"] += n)
    and object-based trackers (tqdm: handler.update(n)).
    No-op if handler is None. Safe to call from worker threads: an increment
    is a read then a write, so two threads could otherwise lose one.
    """
    with _progress_lock:
        if isinstance(handler, dict) and "current" in handler:
            handler["current"] += n
        elif hasattr(handler, "update"):
            handler.update(n)


def duration(seconds: float) -> str:
    """A duration as the log shows it: seconds under 2 minutes, then minutes."""
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"


def species_for_sequences(df: pd.DataFrame) -> str:
    """Size of a species list as the log shows it: "12 species for 5
    sequences" (the eDNA sequences of the ``seq_id`` column)."""
    n_species = df["scientificName"].nunique() if "scientificName" in df else 0
    n_seqs = df["seq_id"].nunique() if "seq_id" in df else 0
    return f"{n_species:,} species for {n_seqs:,} {pluralize(n_seqs, 'sequence')}"


def name_list(
    names: Sequence, shown: int = 5, sep: str = ", ", more: str = " and"
) -> str:
    """Names as a log line lists them: the first *shown*, joined by *sep*,
    then "and N more" (*more* gives what precedes the number)."""
    text = sep.join(str(name) for name in names[:shown])
    if len(names) > shown:
        text += f"{more} {len(names) - shown} more"
    return text


def count_phrase(
    values: pd.Series, noun: str = "", order: Sequence | None = None
) -> str:
    """How many times each value occurs, as in "15 not in BOLD, 14 only
    GenBank copies": the values of *order* that occur, in that order, or
    else all, the most frequent first (*noun* after each count, e.g.
    "species")."""
    counts = values.value_counts()
    noun = f" {noun}" if noun else ""
    order = counts.index if order is None else [o for o in order if o in counts]
    return ", ".join(f"{counts[o]}{noun} {o}" for o in order)


# The label of each setting: its widget shows it, and so do the Parameters
# tab (ordered there by section) and the settings line of each step's log.
SETTING_LABELS: dict[str, str] = {
    # MOL (NCBI BLAST)
    "blast_database": "BLAST database",
    "ncbi_max_hits": "Max hits per query",
    "ncbi_ev_exponent": "E-value exponent",
    "ncbi_query_cover": "Query cover (%)",
    "ncbi_method": "Filtering method",
    "ncbi_gap_size": "Gap size (%)",
    "ncbi_gap_min_top": "Gap min top (%)",
    "ncbi_low_identity_threshold": "Low identity warning (%)",
    "ncbi_enforce_low_identity": "Enforce as hard filter",
    # TAX (WoRMS)
    "worms_marine_only": "Marine species only",
    "tax_drop_mol_rejected": "Exclude species already rejected by MOL",
    # GEO (GBIF)
    "gbif_latitude": "Latitude",
    "gbif_longitude": "Longitude",
    "gbif_extents": "Search extents (km)",
    "gbif_area": "Search area",
    "gbif_filter_extent": "Filtering area",
    "gbif_min_occurrences": "Minimum occurrences",
    # EXTRA (BOLD)
    "bold_keep_only_COI5P": "COI-5P records only",
    "bold_keep_ncbi": "Include records mined from GenBank",
    "bold_longest_n": "Max sequences per species",
    "bold_similarity": "Redundancy identity (%)",
    # HYPO (BOLD + NCBI)
    "hypo_max_hits": "Max hits per query",
    "hypo_ev_exponent": "E-value exponent",
    "hypo_identity_cutoff": "Min hit identity (%)",
    "hypo_ntop": "Hits kept per score",
    "hypo_query_cover": "Query cover (%)",
    "hypo_identity": "Proxy identity (%)",
    "hypo_check_ev_exponent": "HYPO check E-value exponent",
}

FILTER_METHOD_LABELS = {
    "barcoding_gap": "Barcoding gap",
    "similarity": "Similarity range",
}

AREA_LABELS = {"circle": "Circle", "box": "Bounding box"}

# Settings whose values the tabs show under another name.
VALUE_LABELS = {
    "ncbi_method": FILTER_METHOD_LABELS,
    "gbif_area": AREA_LABELS,
    "bold_longest_n": {None: "No limit"},
}


def format_setting(value, key: str = "") -> str:
    """A setting's value as the tabs show it (*key*: the setting)."""
    labels = VALUE_LABELS.get(key, {})
    if not isinstance(value, list) and value in labels:
        return labels[value]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def describe_settings(params: dict) -> str:
    """The settings among *params* (those with a label) as the log lists them,
    in the order of the Parameters tab: "BLAST database: core_nt; Max hits per
    query: 500". The gap top is left out with the similarity range, which does
    not use it."""
    unused = (
        {"ncbi_gap_min_top"} if params.get("ncbi_method") == "similarity" else set()
    )
    return "; ".join(
        f"{label}: {format_setting(params[key], key)}"
        for key, label in SETTING_LABELS.items()
        if key in params and key not in unused
    )


def failed_items(df) -> list[str]:
    """Items whose query failed in the call that returned *df*.

    Read from ``df.attrs["failed_items"]``, set by ``save_to_db``. Empty when
    *df* is missing or did not come from a cached step. An item reported as a
    full cache key (a tuple) is shown by its first value, e.g. the sequence id.
    """
    if df is None:
        return []
    names = (
        str(x[0] if isinstance(x, tuple) else x)
        for x in df.attrs.get("failed_items", [])
    )
    return list(dict.fromkeys(names))


class StepLog:
    """Log lines of one search step, the same for every source: a line per
    item it queries ("BOLD search 12/33: Gadus morhua, 45 BOLD sequences") and
    an end line ("BOLD search done after 6 min: 33 of 33 species queried, 1240
    BOLD sequences for 30 species").

    An item is logged at INFO when it is the first, when it took at least
    half of ITEM_INTERVAL seconds since the one before (a slow source: BOLD
    takes about 10 s per species), or when the last line shown is at least
    ITEM_INTERVAL seconds old; the others go to DEBUG. It is shown as its own
    line when no item was hidden since the last line shown, else as progress
    ("BOLD search: 12 of 33 species done (last: Gadus morhua, 45 BOLD
    sequences)"), so the hidden items read as done, not missing. A slow
    source shows every item as its own line, a fast one (WoRMS, GBIF) one
    progress line every few seconds, so a long step is never silent and a
    quick one does not flood the log. A failed item, or one worth
    a look (*notable*: not found, doubtful match), is always shown. *noun* is
    what an item is. Safe to call from worker threads.
    """

    ITEM_INTERVAL = 10.0
    # The clock of the intervals (a test can replace it).
    clock = staticmethod(time.monotonic)

    def __init__(self, step: str, total: int = 0, noun: str = "item"):
        self.step = step
        self.total = total
        self.noun = noun
        # _last: the last line shown; _previous: the last item
        self._start = self._last = self._previous = self.clock()
        self._count = 0
        self._hidden = 0  # items logged at DEBUG since the last line shown
        self._lock = threading.Lock()

    def _of_total(self, count: int) -> str:
        """ "12 of 33 species" ("12 species" without a total)."""
        if not self.total:
            return f"{count} {pluralize(count, self.noun)}"
        return f"{count} of {self.total} {pluralize(self.total, self.noun)}"

    def _advance(self, shown: bool, now: float) -> str:
        """Count the next item, done at *now* (under the lock); its prefix
        ("BOLD search 12/33"). A *shown* line restarts the interval."""
        self._count += 1
        self._previous = now
        if shown:
            self._last = now
            self._hidden = 0
        else:
            self._hidden += 1
        total = f"/{self.total}" if self.total else ""
        return f"{self.step} {self._count}{total}"

    @contextlib.contextmanager
    def query(self, name: str):
        """Label the requests made for item *name*, for their retry lines."""
        with request_label(f"{self.step}: {name}"):
            yield

    def item(self, name: str, outcome: str, notable: bool = False) -> None:
        # Numbered and logged under the lock, so worker threads log in order.
        with self._lock:
            now = self.clock()
            shown = (
                notable
                or self._count == 0
                or now - self._previous >= self.ITEM_INTERVAL / 2
                or now - self._last >= self.ITEM_INTERVAL
            )
            hidden = self._hidden
            prefix = self._advance(shown, now)
            if notable or (shown and not hidden):
                logger.info(f"{prefix}: {name}, {outcome}")
            elif shown:
                logger.info(
                    f"{self.step}: {self._of_total(self._count)} done "
                    f"(last: {name}, {outcome})"
                )
            else:
                logger.debug(f"{prefix}: {name}, {outcome}")

    def failed(self, name: str, error: object) -> None:
        with self._lock:
            prefix = self._advance(True, self.clock())
            logger.warning(f"{prefix}: {name} failed: {error}")

    def done(
        self,
        summary: str,
        failed: Sequence[str] = (),
        retry: str = "Run the step again",
    ) -> None:
        """Log the end of the step: how many of its items it queried (with a
        total; *summary* need not repeat it), *summary* of its results, then
        the *failed* items (by name), which the next run queries again
        (*retry* says how)."""
        elapsed = duration(self.clock() - self._start)
        if self.total:
            summary = f"{self._of_total(self._count)} queried, {summary}"
        logger.info(f"{self.step} done after {elapsed}: {summary}")
        if failed:
            them = "it" if len(failed) == 1 else "them"
            logger.warning(
                f"{self.step}: {len(failed)} {pluralize(len(failed), self.noun)} "
                f"failed ({name_list(failed)}). {retry} to retry {them}."
            )


def extract_specific_epithet(
    scientific_name: str | None, genus: str | None
) -> str | None:
    """Extract the specific epithet from a binomial scientific name.

    Args:
        scientific_name: Full scientific name (e.g. 'Gadus morhua').
        genus: Genus name used to validate the first word of the name.

    Returns:
        The specific epithet if the name starts with genus, otherwise None.
    """
    if not scientific_name or not genus:
        return None

    name_parts = scientific_name.split()

    if len(name_parts) >= 2 and name_parts[0] == genus:
        return name_parts[1]
    return None


##### DECORATORS #####


def preserve_sequence_order(column_name: str, source_df_name: str) -> Callable:
    """Decorator that restores the original sequence order of the input DataFrame.

    Preserves any internal sorting (e.g. by scientificName) performed inside
    the wrapped function, while re-imposing the original seq_id order from the
    input DataFrame.

    Args:
        column_name: Column used to define and restore order (e.g. 'seq_id').
        source_df_name: Name of the function parameter holding the input DataFrame.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            input_df = bound_args.arguments.get(source_df_name)

            if (
                not isinstance(input_df, pd.DataFrame)
                or column_name not in input_df.columns
            ):
                return func(*args, **kwargs)

            original_order = {
                val: i for i, val in enumerate(input_df[column_name].unique())
            }

            result = func(*args, **kwargs)

            if not isinstance(result, pd.DataFrame) or result.empty:
                return result

            # A sort key, not a column: the result may be the caller's own frame.
            result = result.sort_values(
                column_name, key=lambda ids: ids.map(original_order), kind="stable"
            ).reset_index(drop=True)

            return result

        return wrapper

    return decorator
