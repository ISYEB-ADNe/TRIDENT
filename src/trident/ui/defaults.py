"""UI default values.

Scientific thresholds come from ``trident.core.constants`` (the single source
of truth, also used by the pipeline functions). Performance and UI-only defaults
(thread counts, batch sizes, the map centre) are defined here.
"""

from trident.clients.ncbi import BLAST_GIVE_UP
from trident.core.constants import (
    NCBI_MAX_HITS,
    NCBI_EV_EXPONENT,
    NCBI_METHOD,
    NCBI_QUERY_COVER,
    NCBI_GAP_SIZE,
    NCBI_GAP_MIN_TOP,
    NCBI_LOW_IDENTITY_THRESHOLD,
    NCBI_ENFORCE_LOW_IDENTITY,
    GBIF_EXTENT,
    GBIF_MIN_OCCURRENCES,
    BOLD_SIMILARITY,
    HYPO_MAX_HITS,
    HYPO_EV_EXPONENT,
    HYPO_IDENTITY_CUTOFF,
    HYPO_NTOP,
    HYPO_QUERY_COVER,
    HYPO_IDENTITY,
    HYPO_CHECK_EV_EXPONENT,
)

### NCBI/BLAST ###
NCBI_MAX_HITS_DEFAULT = NCBI_MAX_HITS
# Highest "Max hits per query" the MOL search accepts.
NCBI_MAX_HITS_LIMIT = 1000
NCBI_EV_EXPONENT_DEFAULT = NCBI_EV_EXPONENT
NCBI_NUM_THREADS_DEFAULT = 4
# NCBI takes one request every 10 s and each running job is polled once a
# minute, so beyond about 6 jobs at once the polls take every request slot.
BLAST_MAX_THREADS = 6
# Minutes a BLAST job may wait at NCBI before it is resubmitted once, then
# counted as failed (BlastGaveUp); most jobs take 1 to 3 min.
BLAST_GIVE_UP_MINUTES_DEFAULT = round(BLAST_GIVE_UP / 60)
# Below this, healthy jobs on a busy NCBI would be given up too.
BLAST_GIVE_UP_MINUTES_MIN = 5
# HYPO searches are limited to a few species and come back within about 2 min;
# a job waiting longer is stuck at NCBI and comes back soon once resubmitted.
HYPO_GIVE_UP_MINUTES_DEFAULT = BLAST_GIVE_UP_MINUTES_MIN
NCBI_BATCH_SIZE_DEFAULT = 50
NCBI_METHOD_DEFAULT = NCBI_METHOD
NCBI_QUERY_COVER_DEFAULT = NCBI_QUERY_COVER
NCBI_GAP_SIZE_DEFAULT = NCBI_GAP_SIZE
NCBI_GAP_MIN_TOP_DEFAULT = NCBI_GAP_MIN_TOP
NCBI_LOW_IDENTITY_THRESHOLD_DEFAULT = NCBI_LOW_IDENTITY_THRESHOLD
NCBI_ENFORCE_LOW_IDENTITY_DEFAULT = NCBI_ENFORCE_LOW_IDENTITY

### GBIF ###
GBIF_LATITUDE_DEFAULT = 0.0
GBIF_LONGITUDE_DEFAULT = 0.0
GBIF_EXTENT_DEFAULT = GBIF_EXTENT
GBIF_MIN_OCCURRENCES_DEFAULT = GBIF_MIN_OCCURRENCES

### BOLD ###
BOLD_SIMILARITY_DEFAULT = BOLD_SIMILARITY

### HYPO ###
HYPO_MAX_HITS_DEFAULT = HYPO_MAX_HITS
HYPO_EV_EXPONENT_DEFAULT = HYPO_EV_EXPONENT
HYPO_IDENTITY_CUTOFF_DEFAULT = HYPO_IDENTITY_CUTOFF
HYPO_NTOP_DEFAULT = HYPO_NTOP
HYPO_NUM_THREADS_DEFAULT = 4
HYPO_BATCH_SIZE_DEFAULT = 50
HYPO_QUERY_COVER_DEFAULT = HYPO_QUERY_COVER
HYPO_IDENTITY_DEFAULT = HYPO_IDENTITY
HYPO_CHECK_EVALUE_EXPONENT_DEFAULT = HYPO_CHECK_EV_EXPONENT
