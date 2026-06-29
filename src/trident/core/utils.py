from functools import wraps
import inspect
from typing import Callable

import pandas as pd


##### MISCELLANEOUS UTILITIES #####


def normalize_name(name):
    """Normalize a taxonomic name for joining: collapse whitespace + casefold.

    Used only as a join key, never to replace the displayed name. Non-strings
    (e.g. NaN) are returned unchanged so they do not spuriously match each other.
    """
    if not isinstance(name, str):
        return name
    return " ".join(name.split()).casefold()


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
) -> pd.DataFrame:
    """Keep the top *n* rows per *keys* group, ordered by *sort* (descending by default).

    Row order follows the sort (not the group keys). With n == 1 this is
    sort + drop_duplicates(keep first); with n > 1 it is sort + groupby.head(n).
    Optionally narrow to *columns*.
    """
    ranked = df.sort_values(sort, ascending=ascending)
    if n == 1:
        top = ranked.drop_duplicates(subset=keys)
    else:
        top = ranked.groupby(keys, sort=False, observed=False).head(n)
    if columns is not None:
        top = top[columns]
    return top


def notify_progress(handler: dict | object | None, n: int = 1) -> None:
    """Increment a progress tracker by n steps.

    Supports dict-based trackers (Streamlit: handler["current"] += n)
    and object-based trackers (tqdm: handler.update(n)).
    No-op if handler is None.
    """
    if isinstance(handler, dict) and "current" in handler:
        handler["current"] += n
    elif hasattr(handler, "update"):
        handler.update(n)


_TAXON_ORDER = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "scientificName",
    "specificEpithet",
    "scientificNameAuthorship",
    "taxonRank",
    "taxonID",
    "taxonID_db",
    "taxonURL",
]


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


def reorder_taxonomy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Group taxonomy columns together at their first natural position.

    Args:
        df: DataFrame potentially containing taxonomy columns.

    Returns:
        DataFrame with taxonomy columns grouped and ordered canonically.
        Returns the original DataFrame unchanged if no taxonomy columns are found.
    """
    first_taxon_idx = next(
        (i for i, col in enumerate(df.columns) if col in _TAXON_ORDER), None
    )

    if first_taxon_idx is None:
        return df

    taxon_cols = [col for col in _TAXON_ORDER if col in df.columns]
    before_taxon = list(df.columns[:first_taxon_idx])
    after_taxon = [
        col for col in df.columns[first_taxon_idx + 1 :] if col not in taxon_cols
    ]

    return df[before_taxon + taxon_cols + after_taxon]


def group_species_by_flag(
    df: pd.DataFrame,
    flag_col: str,
    seq_id_col: str = "seq_id",
    name_col: str = "scientificName",
) -> dict[str, list[str]]:
    """Group species per seq_id where a boolean flag is True.

    Args:
        df: DataFrame with seq_id, scientificName, and a boolean flag column.
        flag_col: Name of the boolean column to filter on.
        seq_id_col: Name of the sequence ID column.
        name_col: Name of the species name column.

    Returns:
        Dict mapping each seq_id to its list of species where flag is True.
    """
    flagged = df[df[flag_col].astype(bool)]
    return flagged.groupby(seq_id_col)[name_col].apply(list).to_dict()


def find_exclusion_pipeline_step(
    all_ids: list[str],
    steps: list[tuple[str, set[str]]],
    id_col: str = "seq_id",
) -> pd.DataFrame:
    """Find the first pipeline step where each ID is absent.

    Walks through steps in order for each ID. The first step whose set
    does not contain the ID is recorded as the exclusion point.

    Args:
        all_ids: All IDs to check.
        steps: Ordered (step_name, set_of_ids_present) pairs.
        id_col: Column name for the ID in the output DataFrame.

    Returns:
        DataFrame with columns: {id_col}, pipeline_step. Only contains
        IDs that were excluded (absent from at least one step).
    """
    excluded = []
    for item_id in all_ids:
        for step_name, id_set in steps:
            if item_id not in id_set:
                excluded.append({id_col: item_id, "pipeline_step": step_name})
                break
    return pd.DataFrame(excluded)


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

            temp_key = "_original_order_idx"
            result[temp_key] = result[column_name].map(original_order)
            result = (
                result.sort_values(by=temp_key, ascending=True, kind="stable")
                .drop(columns=[temp_key])
                .reset_index(drop=True)
            )

            return result

        return wrapper

    return decorator
