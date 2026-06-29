"""Tests for WoRMS failure tracking (network-free via monkeypatch).

run_worms_search relies on get_species_from_genus_list reporting only errored
genera as failures (so the cache layer retries them), not genera that simply
have no AphiaID or no children.
"""

from trident.clients import worms
from trident.clients.worms import get_species_from_genus_list


def test_errored_genus_is_a_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network error")

    monkeypatch.setattr(worms, "get_aphia_id", boom)
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == ["Gadus"]
    assert df.empty


def test_no_aphia_id_is_not_a_failure(monkeypatch):
    # Genus not in WoRMS -> valid empty, not a failure (won't be retried).
    monkeypatch.setattr(worms, "get_aphia_id", lambda *a, **k: None)
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == []


def test_no_children_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(worms, "get_aphia_id", lambda *a, **k: 12345)
    monkeypatch.setattr(worms, "get_children_by_aphia_id", lambda *a, **k: [])
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == []


def test_marine_only_threaded_to_children(monkeypatch):
    """The marine_only flag (default True) reaches the children query."""
    captured = {}
    monkeypatch.setattr(worms, "get_aphia_id", lambda *a, **k: 42)

    def fake_children(aphia_id, session=None, marine_only=True, **k):
        captured["marine_only"] = marine_only
        return []

    monkeypatch.setattr(worms, "get_children_by_aphia_id", fake_children)

    get_species_from_genus_list(["Gadus"], delay=0)  # default
    assert captured["marine_only"] is True

    get_species_from_genus_list(["Gadus"], delay=0, marine_only=False)
    assert captured["marine_only"] is False


# --- name resolution (R3) ---


def test_get_aphia_record_returns_none_on_404():
    """An unknown name (e.g. open nomenclature) 404s; that is 'not found', not an error."""

    class Resp:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("raise_for_status must not run on a 404")

        def json(self):
            raise AssertionError("json must not be parsed on a 404")

    class Sess:
        def get(self, *a, **k):
            return Resp()

    assert worms.get_aphia_record("Ammodytes sp.", session=Sess()) is None


def test_resolve_accepted_passthrough(monkeypatch):
    monkeypatch.setattr(
        worms,
        "get_aphia_record",
        lambda name, session=None: {
            "valid_name": "Gadus morhua",
            "valid_AphiaID": 126436,
            "status": "accepted",
        },
    )
    r = worms.resolve_accepted_name("Gadus morhua")
    assert r["accepted_name"] == "Gadus morhua"
    assert r["aphia_id"] == 126436
    assert r["is_synonym"] is False


def test_resolve_maps_synonym(monkeypatch):
    monkeypatch.setattr(
        worms,
        "get_aphia_record",
        lambda name, session=None: {
            "valid_name": "Gadus macrocephalus",
            "valid_AphiaID": 126437,
            "status": "unaccepted",
        },
    )
    r = worms.resolve_accepted_name("Gadus ogac")
    assert r["accepted_name"] == "Gadus macrocephalus"
    assert r["aphia_id"] == 126437
    assert r["is_synonym"] is True


def test_resolve_not_found_keeps_input(monkeypatch):
    monkeypatch.setattr(worms, "get_aphia_record", lambda name, session=None: None)
    r = worms.resolve_accepted_name("Ammodytes sp.")
    assert r["accepted_name"] == "Ammodytes sp."
    assert r["aphia_id"] is None
    assert r["is_synonym"] is False
