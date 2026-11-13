"""Scientific default thresholds for the TRIDENT pipeline.

Single source of truth for the cutoffs applied at each step. The Streamlit UI
re-exports these (``trident/ui/defaults.py``) and the pipeline/client functions
use them as default arguments, so a notebook run matches the app.
Performance-only knobs (thread counts, batch sizes) live in the UI layer, not
here.
"""

# --- NCBI BLAST (MOL search, HYPO search, HYPO check) ---
# core_nt: nt without most eukaryotic chromosome sequences; it keeps the
# mitochondrial records and is the default database of NCBI's BLAST website.
BLAST_DATABASE = "core_nt"

# --- MOL (NCBI BLAST + barcoding-gap filter) ---
NCBI_MAX_HITS = 500
NCBI_EV_EXPONENT = 20
# A sequence under SHORT_SEQUENCE_LENGTH bp cannot reach a strict E-value with
# mismatches: above SHORT_SEQUENCE_EV_EXPONENT, it gets only near-identical
# hits, or none (an 80 bp read reaches about 82% identity at 1e-10).
SHORT_SEQUENCE_LENGTH = 80
SHORT_SEQUENCE_EV_EXPONENT = 10
NCBI_METHOD = "barcoding_gap"
NCBI_QUERY_COVER = 90
NCBI_GAP_SIZE = 2
NCBI_GAP_MIN_TOP = 97
NCBI_LOW_IDENTITY_THRESHOLD = 95
NCBI_ENFORCE_LOW_IDENTITY = False
# A search that reached its hit limit with every hit within this range (%) of a
# top hit at or above this identity may have left close species out.
NCBI_CUT_TOP_IDENTITY = 97
NCBI_CUT_RANGE = 3

# --- TAX (WoRMS) ---
WORMS_MARINE_ONLY = True
# Per sequence, leave out the TAX species that BLAST returned for it but that
# failed the MOL filter.
TAX_DROP_MOL_REJECTED = True

# --- GEO (GBIF) ---
GBIF_CONFIDENCE = 95
# "circle" (within each extent of the site) or "box" (the rectangle around it).
GBIF_AREA = "circle"
GBIF_EXTENT = 500
GBIF_MIN_OCCURRENCES = 3

# --- EXTRA (BOLD) ---
BOLD_SIMILARITY = 98
BOLD_KEEP_ONLY_COI5P = True
# Max sequences kept per species after the redundancy filter (None: no limit).
BOLD_LONGEST_N = None
# The BOLD records mined from GenBank: as proxies they would find themselves
# in HYPO.
BOLD_KEEP_NCBI = False

# --- HYPO (NCBI BLAST validation of BOLD proxies) ---
HYPO_MAX_HITS = 500
HYPO_EV_EXPONENT = 20
HYPO_IDENTITY_CUTOFF = 90
HYPO_NTOP = 3
HYPO_QUERY_COVER = 50
HYPO_IDENTITY = 95
HYPO_CHECK_EV_EXPONENT = 3
