"""Coercion of stored widget overrides back to their default's type.

Cache inputs are persisted as strings, so a value read back from the DB has to
be restored to the type the widget expects. A notebook run stores ``98.0``
where the app stores ``98``; both must come back as the int ``98``.
"""

import pytest

from trident.ui.ui import coerce_stored_value


@pytest.mark.parametrize(
    ("stored", "default", "expected"),
    [
        ("98", 98, 98),
        ("98.0", 98, 98),  # notebook writes floats; int("98.0") raises
        ("2.0", 2, 2),
        (98.0, 98, 98),
        ("78.5", 0.0, 78.5),
        ("500", 500, 500),
    ],
)
def test_numeric_strings_are_coerced_to_the_default_type(stored, default, expected):
    result = coerce_stored_value(stored, default)
    assert result == expected
    assert type(result) is type(default)


@pytest.mark.parametrize(
    ("stored", "default"),
    [
        ((90, 100), 98),  # a corrupted range-slider value
        ("not a number", 98),
        (None, 98),
        ([98], 98),
    ],
)
def test_uncoercible_values_fall_back_to_the_default(stored, default):
    assert coerce_stored_value(stored, default) == default


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("False", False),
        ("True", True),
        (False, False),
        (True, True),
    ],
)
def test_boolean_strings_keep_their_meaning(stored, expected):
    # bool("False") is True, so a stored "False" must not flip to enabled.
    assert coerce_stored_value(stored, True) is expected


def test_value_is_returned_unchanged_when_no_default_type_is_known():
    assert coerce_stored_value("barcoding_gap", None) == "barcoding_gap"
