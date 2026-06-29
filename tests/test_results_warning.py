"""Tests for the low identity warning logic.

Exercises the real ``add_low_identity_warning`` from the results pipeline,
which is importable without Streamlit.
"""

import pandas as pd

from trident.pipelines.results_pipeline import add_low_identity_warning


def _make_results_df(**overrides):
    defaults = {
        "seq_id": ["s1", "s2", "s3"],
        "scientificName": ["SpecA", "SpecB", "SpecC"],
        "validation_step": ["MOL+GEO", "MOL+GEO", "HYPO"],
        "ncbi_top_identity_percentage": [99.0, 93.0, pd.NA],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


class TestAddLowIdentityWarning:
    def test_mol_geo_above_threshold(self):
        df = _make_results_df()
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[0, "low_identity_warning"] == False  # noqa: E712

    def test_mol_geo_below_threshold(self):
        df = _make_results_df()
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[1, "low_identity_warning"] == True  # noqa: E712

    def test_hypo_nan_not_flagged(self):
        """HYPO with no ncbi identity → not flagged (NaN is not low)."""
        df = _make_results_df()
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[2, "low_identity_warning"] == False  # noqa: E712

    def test_hypo_below_threshold(self):
        """HYPO with ncbi identity below threshold → flagged."""
        df = _make_results_df(
            ncbi_top_identity_percentage=[99.0, 99.0, 95.0],
        )
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[2, "low_identity_warning"] == True  # noqa: E712

    def test_hypo_above_threshold(self):
        """HYPO with ncbi identity above threshold → not flagged."""
        df = _make_results_df(
            ncbi_top_identity_percentage=[99.0, 99.0, 98.0],
        )
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[2, "low_identity_warning"] == False  # noqa: E712

    def test_all_nan_not_flagged(self):
        """All NaN identities → no warnings."""
        df = _make_results_df(
            ncbi_top_identity_percentage=[pd.NA, pd.NA, pd.NA],
        )
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert not result["low_identity_warning"].any()

    def test_threshold_at_boundary(self):
        """Exactly at threshold → not flagged (strict less-than)."""
        df = _make_results_df(
            ncbi_top_identity_percentage=[97.0, 97.0, 97.0],
        )
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert not result["low_identity_warning"].any()

    def test_string_identities_coerced(self):
        """String identities (e.g. from cache) are coerced before comparison."""
        df = _make_results_df(
            ncbi_top_identity_percentage=["99.0", "93.0", "not_a_number"],
        )
        result = add_low_identity_warning(df.copy(), threshold=97)
        assert result.loc[0, "low_identity_warning"] == False  # noqa: E712
        assert result.loc[1, "low_identity_warning"] == True  # noqa: E712
        assert result.loc[2, "low_identity_warning"] == False  # noqa: E712
