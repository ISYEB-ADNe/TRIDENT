"""Scientific default thresholds for the TRIDENT pipeline.

Single source of truth for the cutoffs applied at each step. The Streamlit UI
re-exports these (``trident/ui/defaults.py``) and the pipeline/client functions
use them as default arguments, so a notebook run matches the app. See the
README "Methods and default parameters" section for each value. Performance-only knobs
(thread counts, batch sizes) live in the UI layer, not here.
"""

# --- MOL (NCBI BLAST + barcoding-gap filter) ---
NCBI_MAX_HITS = 500
NCBI_EV_EXPONENT = 20
NCBI_METHOD = "barcoding_gap"
NCBI_QUERY_COVER = 90
NCBI_GAP_SIZE = 2
NCBI_GAP_MIN_TOP = 97
NCBI_LOW_IDENTITY_THRESHOLD = 95
NCBI_ENFORCE_LOW_IDENTITY = False

# --- GEO (GBIF) ---
GBIF_CONFIDENCE = 95
GBIF_EXTENT = 500
GBIF_MIN_OCCURRENCES = 3

# --- EXTRA (BOLD) ---
BOLD_SIMILARITY = 98

# --- HYPO (NCBI BLAST validation of BOLD proxies) ---
HYPO_MAX_HITS = 500
HYPO_EV_EXPONENT = 20
HYPO_IDENTITY_CUTOFF = 90
HYPO_NTOP = 3
HYPO_QUERY_COVER = 50
HYPO_IDENTITY = 95
HYPO_CHECK_EV_EXPONENT = 3
