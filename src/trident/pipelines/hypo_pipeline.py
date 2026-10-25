"""
HYPO pipeline — hypothetical species validation via proxy sequences.

Produces species that have no target marker in GenBank but are validated
via CO1 proxy (BOLD sequences BLASTed against NCBI) and confirmed as
geographically plausible.

Steps:
    1. prepare_hypo_input       build per-seq_id BLAST inputs from EXTRA + MOL/TAX
    2. run_hypo_search          NCBI BLAST on proxy sequences            [CustomCache]
    3. run_hypo_merge           map proxy IDs back to targets (EXTRA)     [FullCache]
    4. run_hypo_filter          threshold on identity / query cover       [FullCache]
    5. hypo_check_queries       Entrez query of each hypothetical species
       run_hypo_check           per-species NCBI BLAST verification       [CustomCache]
    6. finalize_hypo_results    label, merge check results, final df      [FullCache]
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from loguru import logger

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from trident.core.constants import (
    HYPO_EV_EXPONENT,
    HYPO_MAX_HITS,
    HYPO_IDENTITY_CUTOFF,
    HYPO_NTOP,
    HYPO_QUERY_COVER,
    HYPO_IDENTITY,
    HYPO_CHECK_EV_EXPONENT,
    NCBI_QUERY_COVER,
)
from trident.core.database import save_to_db, FullCache, CustomCache
from trident.core.utils import (
    StepLog,
    normalize_name,
    pluralize,
    species_for_sequences,
    preserve_sequence_order,
    notify_progress,
    ensure_columns,
    top_hit_per_group,
)
from trident.clients.ncbi import (
    BLAST_GIVE_UP,
    BLAST_HIT_COLUMNS,
    BlastCircuit,
    BlastCpuLimit,
    BlastGaveUp,
    blast_batch,
    create_sequence_batches,
    run_blast_pool,
    process_blast_results,
    submit_blast_batch,
)
from trident.pipelines.names import ncbi_names, other_names


### DATA PREPARATION ###


def orgn_query(names: list[str]) -> str:
    """Entrez query of the records of any of *names*."""
    return "(" + " OR ".join(f'"{n}"[orgn]' for n in names) + ")"


def hypo_queries(tax_df, mol_df, name_table) -> pd.DataFrame:
    """Entrez organism query of each seq_id: the species of its MOL list.

    A hypothetical species is compared with the species that passed the MOL
    filter for the sequence, whether or not they passed GEO, among those WoRMS
    kept in the TAX list (in_mol). That leaves out what WoRMS does not list
    there (terrestrial contaminants with "Marine only", "cf." and "sp." hits,
    "unidentified"). Species go by name, the WoRMS accepted one and the NCBI
    names the sequence's MOL hits gave them (see species_of): [orgn] matches a
    name with its subspecies but not hybrid records (they hold the other
    parent's mitogenome).

    Args:
        tax_df: TAX list with seq_id, scientificName and in_mol.
        mol_df: MOL list (NCBI names).
        name_table: See build_name_table.

    Returns:
        DataFrame with columns seq_id, entrez_query and mol_species (the
        names, joined by "; "), one row per seq_id that has such species.
        prepare_hypo_input and run_hypo_merge both take it, so search and
        merge see the same species.
    """
    columns = ["seq_id", "entrez_query", "mol_species"]
    if tax_df.empty:
        return pd.DataFrame(columns=columns)
    mol = tax_df.loc[tax_df["in_mol"].astype(bool), ["seq_id", "scientificName"]]
    # The species' own name, also spelled with another case or spacing, is
    # already in the query.
    synonyms = mol.assign(
        scientificName=ncbi_names(mol, mol_df, name_table, synonyms_only=True)
    ).explode("scientificName")
    names = pd.concat([mol, synonyms.dropna()])
    species = names.groupby("seq_id")["scientificName"].agg(lambda s: sorted(set(s)))
    return pd.DataFrame(
        {
            "seq_id": species.index,
            "entrez_query": [orgn_query(s) for s in species],
            "mol_species": ["; ".join(s) for s in species],
        }
    )[columns]


def count_hypo_searches(extra_df, queries_df) -> tuple[int, int]:
    """Searches a HYPO run makes, and the seq_ids they cover.

    A proxy shared by seq_ids with the same query is searched once (the
    search's cache item is the proxy record and its query).
    """
    searched = extra_df.merge(queries_df, on="seq_id")
    n_searches = len(searched[["seq_id_extra", "entrez_query"]].drop_duplicates())
    return n_searches, searched["seq_id"].nunique()


def _query_map(queries_df):
    """{seq_id: entrez_query} of a hypo_queries table."""
    return dict(zip(queries_df["seq_id"], queries_df["entrez_query"]))


def prepare_hypo_input(extra_df, queries_df):
    """Build per-seq_id BLAST inputs from EXTRA proxy sequences.

    For each seq_id, creates SeqRecords from EXTRA sequences, with its Entrez
    organism query to constrain the BLAST search. A seq_id without a query
    gets no input: an unconstrained search would let a proxy match its own
    species.

    Args:
        extra_df: EXTRA DataFrame with proxy sequences (seq_id_extra, dna_sequence_extra).
        queries_df: Entrez query of each seq_id (see hypo_queries).

    Returns:
        dict: {seq_id: {'sequences': [SeqRecord, ...], 'entrez_query': str}, ...}
    """
    queries = _query_map(queries_df)

    input_data = {}
    for seq_id, group in extra_df.groupby("seq_id", observed=False):
        if seq_id not in queries:
            continue
        seq_records = [
            SeqRecord(Seq(seq.replace("-", "N")), id=seq_id_extra)
            for seq, seq_id_extra in zip(
                group["dna_sequence_extra"], group["seq_id_extra"]
            )
        ]
        input_data[seq_id] = {
            "sequences": seq_records,
            "entrez_query": queries[seq_id],
        }

    n_seq = sum(len(v["sequences"]) for v in input_data.values())
    logger.debug(f"Prepared HYPO input: {n_seq} BOLD records")
    return input_data


### SEARCH ###


def _hypo_search_params(
    sequences_dict,
    ev_exponent: int,
    max_hits: int,
    identity_cutoff: float,
    ntop: int,
    **kwargs,
):
    """Prepare parameters dict for HYPO search step."""
    params = {
        "hypo_ev_exponent": ev_exponent,
        "hypo_max_hits": max_hits,
        "hypo_identity_cutoff": identity_cutoff,
        "hypo_ntop": ntop,
    }
    inputs = [
        {"seq_id": s.id, "entrez_query": v["entrez_query"], **params}
        for v in sequences_dict.values()
        for s in v["sequences"]
    ]
    return inputs, params


def _hypo_search_rebuild(input_selected, original_inputs):
    """Rebuild inputs for HYPO search step."""
    selected_lookup = {
        (item["seq_id"], item["entrez_query"]) for item in input_selected
    }

    new_sequences_dict = {}
    for sample_id, data in original_inputs["sequences_dict"].items():
        matched_records = [
            rec
            for rec in data["sequences"]
            if (rec.id, data["entrez_query"]) in selected_lookup
        ]
        if matched_records:
            new_sequences_dict[sample_id] = {
                "sequences": matched_records,
                "entrez_query": data["entrez_query"],
            }

    new_inputs = original_inputs.copy()
    new_inputs["sequences_dict"] = new_sequences_dict
    return new_inputs


@save_to_db(
    "hypo_search",
    cache=CustomCache(
        prepare_fn=_hypo_search_params,
        rebuild_fn=_hypo_search_rebuild,
        match_map_dict={"seq_id": "seq_id", "entrez_query": "entrez_query"},
    ),
    log_as=("HYPO search", "BLAST search"),
)
def run_hypo_search(
    sequences_dict,
    batch_size: int = 50,
    num_threads: int = 4,
    ev_exponent: int = HYPO_EV_EXPONENT,
    max_hits: int = HYPO_MAX_HITS,
    identity_cutoff: float = HYPO_IDENTITY_CUTOFF,
    ntop: int = HYPO_NTOP,
    give_up_minutes: float = BLAST_GIVE_UP / 60,
    progress_handler: dict | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """NCBI BLAST search on the proxy sequences of each seq_id, restricted to
    its Entrez query.

    A proxy record shared by several seq_ids with the same Entrez query is
    searched once: (proxy record id, entrez_query) is one cache item, and its
    hits are returned once. Batches are made per query across seq_ids, and all
    of them run as one parallel BLAST run (``run_blast_pool``).

    Args:
        sequences_dict: {seq_id: {'sequences': [SeqRecord], 'entrez_query': str}}.
        batch_size: Sequences per NCBI batch.
        num_threads: Batches run at once at normal pace, plus
            BLAST_SLOW_THREADS while jobs are slow; shared by every seq_id.
        ev_exponent: Negative exponent for E-value threshold (e.g. 10 → 1e-10).
        max_hits: Maximum hits per query.
        identity_cutoff: Minimum identity percentage to retain.
        ntop: If > 0, keep only top N hits per (proxy record, hit species,
            Entrez query) by identity.
        give_up_minutes: Minutes a BLAST job may wait at NCBI before it is
            resubmitted once, then counted as failed (see ``BlastGaveUp``).
        progress_handler: Optional progress tracker.
        failure_sink: If provided, the (proxy record id, entrez_query) of each
            proxy record (BOLD sequence) whose BLAST errored is appended here,
            so the cache layer retries it instead of caching it as empty.
        result_sink: Given by save_to_db, which saves the proxy records of each
            batch as soon as its BLAST is done.

    Returns:
        DataFrame with NCBI BLAST results for proxy sequences.
    """
    log = StepLog("HYPO search")

    # Batches are made per Entrez query, across MOTUs: MOTUs with the same MOL
    # species fill one batch instead of sending one small job each. All batches
    # run in one pool, so a slow NCBI job holds back only its own batch.
    by_query: dict[str, list[SeqRecord]] = {}
    owner = {}  # (proxy record id, entrez_query) -> MOTU that queued it
    for seq_id, seq_dict in sequences_dict.items():
        query = seq_dict["entrez_query"]
        for rec in seq_dict["sequences"]:
            if (rec.id, query) not in owner:
                owner[(rec.id, query)] = seq_id
                by_query.setdefault(query, []).append(rec)

    batches = [
        (batch, query)
        for query, records in by_query.items()
        for batch in create_sequence_batches(records, batch_size)
    ]
    n_seqs = len(sequences_dict)
    logger.debug(
        f"HYPO search: {len(owner)} {pluralize(len(owner), 'search')} "
        f"for {n_seqs} {pluralize(n_seqs, 'sequence')}, in {len(batches)} "
        f"{pluralize(len(batches), 'batch')}"
    )

    circuit = BlastCircuit()  # shared by every job of this run
    blast_kwargs = {"evalue": 10 ** (-ev_exponent), "hitlist_size": max_hits}
    no_hits = pd.DataFrame(columns=[*BLAST_HIT_COLUMNS, "entrez_query"])

    def keep(hits: pd.DataFrame) -> pd.DataFrame:
        """The hits above the identity cutoff, then the top N of each cache item
        (proxy record, Entrez query) and hit species: hits under two queries do
        not compete, so a batch's hits are kept on their own."""
        if identity_cutoff > 0:
            hits = hits[hits["identity_percentage"] >= identity_cutoff]
        if ntop > 0:
            hits = top_hit_per_group(
                hits,
                keys=["seq_id", "scientificName", "entrez_query"],
                sort=["identity_percentage", "query_cover"],
                n=ntop,
            )
        return hits

    def job(k, batch, query):
        # The label names the sequences whose BOLD records the batch searches,
        # so every log line of the batch says which.
        owners = list(dict.fromkeys(owner[(r.id, query)] for r in batch))
        others = len(owners) - 1
        more = f" and {others} {pluralize(others, 'other')}" if others else ""
        records = pluralize(len(batch), "BOLD sequence")
        label = f"{k}/{len(batches)} ({len(batch)} {records}, for {owners[0]}{more})"
        dfs, failed_ids = blast_batch(
            batch, label, circuit, progress_handler, entrez_query=query, **blast_kwargs
        )
        found = [
            process_blast_results(df, fasta_sequences=batch).assign(entrez_query=query)
            for df in dfs
            if not df.empty
        ]
        hits = keep(pd.concat(found, ignore_index=True)) if found else no_hits
        if result_sink is not None:
            done = [(r.id, query) for r in batch if r.id not in failed_ids]
            result_sink(hits, done)
        return hits, failed_ids, sum(len(df) for df in dfs)

    jobs = [(k, batch, query) for k, (batch, query) in enumerate(batches, start=1)]

    kept_hits = []
    outcomes = run_blast_pool(job, jobs, num_threads, give_up_minutes * 60)
    failed_records = []
    n_hits = 0  # as the batch lines count them, before the cutoff and top N
    for (_, batch, query), (hits, failed_ids, n_found) in zip(jobs, outcomes):
        # Report the proxy records whose BLAST errored, with the batch's query:
        # (proxy seq_id, entrez_query) is the cache key, so they are retried on
        # the next run rather than cached as empty. The same proxy under another
        # query stays cached where it succeeded.
        failed_records.extend(f"{rid} for {owner[(rid, query)]}" for rid in failed_ids)
        if failed_ids and failure_sink is not None:
            failure_sink.extend((rid, query) for rid in failed_ids)
        n_hits += n_found
        if not hits.empty:
            kept_hits.append(hits)
    search_df = pd.concat(kept_hits, ignore_index=True) if kept_hits else no_hits

    kept = []
    if identity_cutoff > 0:
        kept.append(f"identity ≥ {identity_cutoff}%")
    if ntop > 0:
        kept.append(f"best {ntop} per BOLD sequence and species")
    rule = f" ({', '.join(kept)})" if kept else ""
    log.done(
        f"{len(search_df)} of {n_hits} {pluralize(n_hits, 'hit')} kept{rule}",
        failed_records,
        "BLAST search",
    )
    return search_df


### MERGE ###


@save_to_db(
    table_name="hypo_merge",
    cache=FullCache(
        inherit_from=["hypo_search_params", "extra_params"],
        fingerprint_on=["hypo_search_df", "queries_df"],
    ),
)
@preserve_sequence_order("seq_id", "extra_df")
def run_hypo_merge(
    hypo_search_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    hypo_search_params: dict | None = None,
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """Map proxy sequence IDs back to target sequences via EXTRA.

    Joins BLAST search results (keyed by seq_id_extra and the Entrez query)
    with the EXTRA DataFrame to recover the original seq_id, scientificName,
    and target dna_sequence. Each seq_id gets only the hits found under its
    own query: a proxy shared by two seq_ids is searched once per query.

    Args:
        hypo_search_df: Search results from run_hypo_search.
        extra_df: EXTRA DataFrame with proxy-to-target mapping.
        queries_df: The Entrez query of each seq_id the search input was
            prepared with (see hypo_queries).
        hypo_search_params: Params from run_hypo_search (for caching).
        extra_params: Params from finalize_extra_results (for caching).

    Returns:
        Merged DataFrame with BLAST stats and EXTRA metadata.
    """
    if hypo_search_params is None:
        hypo_search_params = {}
    if extra_params is None:
        extra_params = {}

    search_cols = [
        "seq_id",
        "scientificName",
        "query_cover",
        "identity_percentage",
        "hit_def",
        "hit_url",
        "entrez_query",
    ]
    queries = _query_map(queries_df)
    merge_df = extra_df.assign(
        entrez_query=pd.Series(
            [queries.get(s, "") for s in extra_df["seq_id"]],
            index=extra_df.index,
            dtype=object,
        )
    ).merge(
        hypo_search_df[search_cols],
        left_on=["seq_id_extra", "entrez_query"],
        right_on=["seq_id", "entrez_query"],
        suffixes=("", "_blast"),
    )
    merge_df = merge_df.drop(columns="entrez_query")

    # seq_id_blast is the duplicate from search (= seq_id_extra), drop it
    # scientificName_blast is the BLAST hit species (not the target species)
    merge_df = merge_df.rename(columns={"scientificName_blast": "scientificName_hit"})
    cols_to_drop = [col for col in merge_df.columns if col.endswith("_blast")]
    merge_df = merge_df.drop(columns=cols_to_drop)

    # NCBI's [orgn] also matches other nodes (e.g. "Gerres sp." records for
    # "Gerres oyena", records named after the hypothetical species itself):
    # only hits on a species of the sequence's MOL list count.
    mol_species = {
        seq_id: {normalize_name(n) for n in names.split("; ")}
        for seq_id, names in zip(queries_df["seq_id"], queries_df["mol_species"])
    }
    in_list = pd.Series(
        [
            normalize_name(hit) in mol_species.get(seq_id, set())
            for seq_id, hit in zip(merge_df["seq_id"], merge_df["scientificName_hit"])
        ],
        index=merge_df.index,
        dtype=bool,
    )
    return merge_df[in_list]


### FILTER ###


@save_to_db(
    table_name="hypo_filter",
    cache=FullCache(
        local={
            "hypo_query_cover": "query_cover",
            "hypo_identity": "identity",
        },
        inherit_from=["hypo_merge_params"],
    ),
)
@preserve_sequence_order("seq_id", "hypo_merge_df")
def run_hypo_filter(
    hypo_merge_df: pd.DataFrame,
    query_cover: float = HYPO_QUERY_COVER,
    identity: float = HYPO_IDENTITY,
    hypo_merge_params: dict | None = None,
) -> pd.DataFrame:
    """Filter HYPO merge results by query coverage and identity thresholds.

    Args:
        hypo_merge_df: Merged results from run_hypo_merge.
        query_cover: Minimum query coverage percentage.
        identity: Minimum identity percentage.
        hypo_merge_params: Params from run_hypo_merge (for caching).

    Returns:
        Filtered DataFrame keeping rows meeting both the query_cover and
        identity thresholds.
    """
    if hypo_merge_params is None:
        hypo_merge_params = {}

    return hypo_merge_df[
        (hypo_merge_df["query_cover"] >= query_cover)
        & (hypo_merge_df["identity_percentage"] >= identity)
    ].copy()


### FILTER SUMMARY ###


@preserve_sequence_order("seq_id", "hypo_filter_df")
def build_hypo_filter_summary(hypo_filter_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize filtered HYPO BLAST results: best hit and total count per species.

    For each (seq_id, scientificName) pair, returns the top hit by identity
    and query cover, plus the number of BLAST hits that passed filtering.

    Args:
        hypo_filter_df: Filtered results from run_hypo_filter.

    Returns:
        Summary DataFrame with columns: seq_id, scientificName, hit_count,
        top_identity_percentage, top_query_cover.
    """
    out_cols = [
        "seq_id",
        "scientificName",
        "hit_count",
        "identity_percentage",
        "query_cover",
    ]

    best_hits = (
        hypo_filter_df.sort_values(
            ["identity_percentage", "query_cover"], ascending=[False, False]
        )
        .groupby(["seq_id", "scientificName"], observed=False, as_index=False)
        .first()
    )

    hit_counts = (
        hypo_filter_df.groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="hit_count")
    )

    summary = best_hits.merge(hit_counts, on=["seq_id", "scientificName"])

    return summary[out_cols].reset_index(drop=True)


### CHECK ###

# Output columns of run_hypo_check, also present when there are no results.
# Species without a hit leave the hit columns empty.
HYPO_CHECK_COLUMNS = [
    "seq_id",
    "scientificName",
    "dna_sequence",
    "scientificName_NCBI",
    "query_cover",
    "identity_percentage",
    "hit_found",
    "hit_def",
    "hit_url",
]


# Sources of the other names a hypothetical species is checked under. NCBI
# knows many old names as synonyms, but not all: it finds nothing for
# Phalacrocorax pygmaeus, which it files as Microcarbo pygmaeus.
HYPO_CHECK_NAME_SOURCES = ["ncbi", "bold", "gbif"]


def hypo_check_queries(species: list[str], name_table) -> dict[str, str]:
    """Entrez organism query of each hypothetical species: its own name and
    the other names the name table gives it (see other_names, which leaves
    out a name that stands for another species).

    Args:
        species: Names of the hypothetical species (scientificName of the
            HYPO filter).
        name_table: See build_name_table.

    Returns:
        {species: entrez_query}
    """
    others = other_names(list(species), name_table, HYPO_CHECK_NAME_SOURCES)
    return {
        name: orgn_query([name, *(n for n, _ in others.get(name, []))])
        for name in species
    }


def _check_query(name: str, check_queries: dict[str, str] | None) -> str:
    """Entrez query a species is checked under (see hypo_check_queries)."""
    return (check_queries or {}).get(name) or orgn_query([name])


def _hypo_check_params(
    hypo_filter_df: pd.DataFrame,
    ev_exponent: int,
    check_queries: dict[str, str] | None = None,
    **kwargs,
):
    """Prepare parameters dict for HYPO check step."""
    sequences_df = hypo_filter_df[["seq_id", "scientificName"]].drop_duplicates()
    params = {
        "hypo_check_ev_exponent": ev_exponent,
    }
    # The query is part of each item, so a species checks again when it gets
    # another name.
    inputs = [
        {
            "seq_id": row["seq_id"],
            "scientificName": row["scientificName"],
            "entrez_query": _check_query(row["scientificName"], check_queries),
            **params,
        }
        for _, row in sequences_df.iterrows()
    ]
    return inputs, params


def _hypo_check_rebuild(input_selected, original_inputs):
    """Rebuild inputs for HYPO check step."""
    selected_lookup = {
        (entry["seq_id"], entry["scientificName"]) for entry in input_selected
    }

    filter_df = original_inputs["hypo_filter_df"]
    mask = pd.MultiIndex.from_frame(filter_df[["seq_id", "scientificName"]]).isin(
        selected_lookup
    )
    original_inputs = original_inputs.copy()
    original_inputs["hypo_filter_df"] = filter_df[mask]
    return original_inputs


@save_to_db(
    table_name="hypo_check",
    cache=CustomCache(
        prepare_fn=_hypo_check_params,
        rebuild_fn=_hypo_check_rebuild,
        match_map_dict={"seq_id": "seq_id", "scientificName": "scientificName"},
    ),
    log_as=("HYPO check", "species check"),
)
@preserve_sequence_order("seq_id", "hypo_filter_df")
def run_hypo_check(
    hypo_filter_df: pd.DataFrame,
    check_queries: dict[str, str] | None = None,
    ev_exponent: int = HYPO_CHECK_EV_EXPONENT,
    num_threads: int = 4,
    give_up_minutes: float = BLAST_GIVE_UP / 60,
    progress_handler: dict | None = None,
    failure_sink: list | None = None,
    result_sink: Callable | None = None,
) -> pd.DataFrame:
    """Per-species NCBI BLAST verification of HYPO candidates.

    BLASTs each target sequence against NCBI filtered to that species, under
    all its names, to confirm the species has the target marker in GenBank.
    The checks are independent jobs, run as one parallel BLAST run
    (``run_blast_pool``).

    Args:
        hypo_filter_df: Filtered DataFrame from run_hypo_filter.
        check_queries: Entrez query of each species (see hypo_check_queries);
            a species without one is checked under its own name only.
        ev_exponent: Negative exponent for E-value threshold.
        num_threads: Checks run at once at normal pace, plus
            BLAST_SLOW_THREADS while jobs are slow (NCBI's request spacing
            makes more than about 6 useless).
        give_up_minutes: Minutes a BLAST job may wait at NCBI before it is
            resubmitted once, then counted as failed (see ``BlastGaveUp``).
        progress_handler: Optional progress tracker.
        failure_sink: If provided, the (seq_id, scientificName) of each check
            whose BLAST errored is appended here, so the cache layer retries it
            instead of caching it as a (false) no-hit.
        result_sink: Given by save_to_db, which saves each check as soon as
            its BLAST is done.

    Returns:
        DataFrame with NCBI check results per species.
    """
    sequences_df = (
        hypo_filter_df[["seq_id", "scientificName", "dna_sequence"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    log = StepLog("HYPO check")

    circuit = BlastCircuit()  # one check per sequence: stop retrying if NCBI is down

    def check(index, seq_id, sci_name, seq_dna) -> pd.DataFrame | None:
        sequence = SeqRecord(Seq(seq_dna), id=seq_id)
        try:
            blast_res = submit_blast_batch(
                [sequence],
                entrez_query=_check_query(sci_name, check_queries),
                evalue=10 ** (-ev_exponent),
                batch_id=f"{index + 1}/{len(sequences_df)} ({seq_id}: {sci_name})",
                job_name="Check",
                hitlist_size=10,
                circuit=circuit,
            )
        except (BlastCpuLimit, BlastGaveUp):  # failed: retried on the next run
            blast_res = None
        notify_progress(progress_handler)
        if blast_res is None:
            return None
        if blast_res.empty:
            result = pd.DataFrame(
                {
                    "seq_id": [seq_id],
                    "scientificName": [sci_name],
                    "dna_sequence": [seq_dna],
                    "hit_found": [False],
                }
            )
        else:
            ncbi_data = process_blast_results(blast_res, fasta_sequences=[sequence])
            result = pd.DataFrame(
                {
                    "seq_id": seq_id,
                    "scientificName": sci_name,
                    "dna_sequence": seq_dna,
                    "scientificName_NCBI": ncbi_data["scientificName"],
                    "query_cover": ncbi_data["query_cover"],
                    "identity_percentage": ncbi_data["identity_percentage"],
                    "hit_found": True,
                    "hit_def": ncbi_data["hit_def"],
                    "hit_url": ncbi_data["hit_url"],
                }
            )
        if result_sink is not None:
            result_sink(result, [(seq_id, sci_name)])
        return result

    calls = list(
        zip(
            sequences_df.index,
            sequences_df["seq_id"],
            sequences_df["scientificName"],
            sequences_df["dna_sequence"],
        )
    )
    outcomes = run_blast_pool(check, calls, num_threads, give_up_minutes * 60)
    all_results = []
    failed = []
    for (_, seq_id, sci_name, _), result in zip(calls, outcomes):
        if result is None:
            failed.append(f"{seq_id}: {sci_name}")
            # BLAST errored: report this (seq_id, species) so it is retried, not
            # recorded as a (false) no-hit. The other species of the seq_id
            # stay cached.
            if failure_sink is not None:
                failure_sink.append((seq_id, sci_name))
        else:
            all_results.append(result)

    # Every sequence may have errored (all reported to failure_sink).
    if not all_results:
        log.done(f"0/{len(sequences_df)} checks with hits", failed, "check")
        return pd.DataFrame(columns=HYPO_CHECK_COLUMNS)

    # Missing hit columns are filled with NaN (float), not pd.NA (object), so
    # they stay numeric.
    check_df = ensure_columns(
        pd.concat(all_results, ignore_index=True), HYPO_CHECK_COLUMNS, fill=np.nan
    )

    # Counted per (sequence, species), as the checks are.
    hits = check_df.loc[check_df["hit_found"], ["seq_id", "scientificName"]]
    log.done(
        f"{len(hits.drop_duplicates())}/{len(sequences_df)} checks with hits",
        failed,
        "check",
    )
    return check_df


### FINALIZE HYPO RESULTS (final) ###


def best_check_hits(hypo_check_df, query_cover, columns=None):
    """Best check hit of each (seq_id, scientificName).

    Hits covering at least the MOL query cover rank first: the best hit is the
    highest identity among hits as long as the MOL filter requires, so it is
    comparable with the MOL hits; a partial hit only when there is no other.
    """
    cover = pd.to_numeric(hypo_check_df["query_cover"], errors="coerce")
    return top_hit_per_group(
        hypo_check_df.assign(_full_cover=cover >= query_cover),
        keys=["seq_id", "scientificName"],
        sort=["_full_cover", "identity_percentage", "query_cover"],
        columns=columns,
    )


@save_to_db(
    table_name="results_hypo",
    cache=FullCache(
        inherit_from=["hypo_check_params", "hypo_filter_params"],
        local={"hypo_check_query_cover": "query_cover"},
        fingerprint_on=["hypo_check_df"],
    ),
)
@preserve_sequence_order("seq_id", "hypo_filter_df")
def finalize_hypo_results(
    hypo_filter_df: pd.DataFrame,
    hypo_check_df: pd.DataFrame,
    query_cover: float = NCBI_QUERY_COVER,
    hypo_filter_params: dict | None = None,
    hypo_check_params: dict | None = None,
) -> pd.DataFrame:
    """Build final HYPO species list by merging check results into filtered data.

    Produces one row per (seq_id, scientificName) with the best NCBI check
    hit stats. Only species not already in MOL — the final output can be
    concatenated with geo_df to form the complete species list.

    Args:
        hypo_filter_df: Filtered results from run_hypo_filter (has taxonomy from GEO).
        hypo_check_df: Check results from run_hypo_check.
        query_cover: The MOL filter's minimum query cover, for the best check
            hit (see best_check_hits).
        hypo_filter_params: Params from run_hypo_filter (for caching).
        hypo_check_params: Params from run_hypo_check (for caching).

    Returns:
        HYPO species DataFrame with check stats (one row per seq_id × species).
    """
    if hypo_filter_params is None:
        hypo_filter_params = {}
    if hypo_check_params is None:
        hypo_check_params = {}

    # Best check hit per (seq_id, scientificName)
    check_hits = best_check_hits(
        hypo_check_df[hypo_check_df["hit_found"].astype(bool)],
        query_cover,
        columns=[
            "seq_id",
            "scientificName",
            "identity_percentage",
            "query_cover",
            "hit_url",
        ],
    ).rename(
        columns={
            "identity_percentage": "ncbi_top_identity_percentage",
            "query_cover": "ncbi_top_query_cover",
            "hit_url": "ncbi_top_hit_url",
        }
    )

    # Keep best proxy hit per (seq_id, scientificName)
    hypo_df = top_hit_per_group(
        hypo_filter_df,
        keys=["seq_id", "scientificName"],
        sort=["identity_percentage", "query_cover"],
    ).reset_index(drop=True)

    # Merge check stats
    hypo_df = hypo_df.merge(check_hits, on=["seq_id", "scientificName"], how="left")

    logger.success(f"HYPO list: {species_for_sequences(hypo_df)}")
    return hypo_df


### CHECK SUMMARY ###


@preserve_sequence_order("seq_id", "hypo_check_df")
def build_hypo_check_summary(
    hypo_check_df: pd.DataFrame, query_cover: float = NCBI_QUERY_COVER
) -> pd.DataFrame:
    """Summarize check results: best hit and count per (seq_id, scientificName).

    Args:
        hypo_check_df: Check results from run_hypo_check.
        query_cover: The MOL filter's minimum query cover, for the best hit
            (see best_check_hits).

    Returns:
        Summary DataFrame with columns: seq_id, scientificName, hit_count,
        identity_percentage, query_cover, hit_url.
    """
    out_cols = [
        "seq_id",
        "scientificName",
        "hit_count",
        "identity_percentage",
        "query_cover",
        "hit_url",
    ]

    best_hits = best_check_hits(hypo_check_df, query_cover)

    hit_counts = (
        hypo_check_df[hypo_check_df["hit_found"].astype(bool)]
        .groupby(["seq_id", "scientificName"], observed=False)
        .size()
        .reset_index(name="hit_count")
    )

    summary = best_hits.merge(hit_counts, on=["seq_id", "scientificName"], how="left")
    summary["hit_count"] = summary["hit_count"].fillna(0).astype(int)

    return summary[out_cols]


### EMPTY INPUT ###


def empty_hypo_results(extra_df: pd.DataFrame) -> pd.DataFrame:
    """Return the HYPO list of an empty EXTRA list, with its usual columns.

    The HYPO steps run on no sequence and without the database, so nothing is
    queried or cached.
    """
    search_df, _ = run_hypo_search({}, db_path=None)
    merge_df, _ = run_hypo_merge(
        search_df,
        extra_df,
        hypo_queries(pd.DataFrame(), pd.DataFrame(), None),
        extra_params={},
        db_path=None,
    )
    filter_df, _ = run_hypo_filter(merge_df, db_path=None)
    check_df, _ = run_hypo_check(filter_df, db_path=None)
    hypo_df, _ = finalize_hypo_results(filter_df, check_df, db_path=None)
    return hypo_df
