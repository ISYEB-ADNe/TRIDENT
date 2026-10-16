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

    monkeypatch.setattr(worms, "get_genus_aphia_ids", boom)
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == ["Gadus"]
    assert df.empty


def test_no_aphia_id_is_not_a_failure(monkeypatch):
    # Genus not in WoRMS -> valid empty, not a failure (won't be retried).
    monkeypatch.setattr(worms, "get_genus_aphia_ids", lambda *a, **k: [])
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == []


def test_no_children_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(worms, "get_genus_aphia_ids", lambda *a, **k: [12345])
    monkeypatch.setattr(worms, "get_children_by_aphia_id", lambda *a, **k: [])
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == []
    # The taxonomy columns are present even when no species is found.
    assert list(df.columns) == worms.WORMS_TAXON_COLUMNS


def test_marine_only_threaded_to_children(monkeypatch):
    """The marine_only flag (default True) reaches the children query."""
    captured = {}
    monkeypatch.setattr(worms, "get_genus_aphia_ids", lambda *a, **k: [42])

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


def test_resolve_follows_the_accepted_record_not_the_first(monkeypatch):
    """A name with an accepted record keeps it, even when WoRMS lists first an
    unaccepted homonym pointing to another species."""
    records = [
        {
            "AphiaID": 218229,
            "scientificname": "Epinephelus undulosus",
            "status": "unaccepted",
            "rank": "Species",
            "valid_AphiaID": 273876,
            "valid_name": "Mycteroperca acutirostris",
        },
        {
            "AphiaID": 367207,
            "scientificname": "Epinephelus undulosus",
            "status": "accepted",
            "rank": "Species",
            "valid_AphiaID": 367207,
            "valid_name": "Epinephelus undulosus",
        },
    ]
    monkeypatch.setattr(worms, "get_aphia_records", lambda name, session=None: records)
    r = worms.resolve_accepted_name("Epinephelus undulosus")
    assert r["accepted_name"] == "Epinephelus undulosus"
    assert r["aphia_id"] == 367207
    assert r["is_synonym"] is False


# --- genus to expand ---


def _genus(aphia_id, status, kingdom="Animalia"):
    return {
        "AphiaID": aphia_id,
        "status": status,
        "rank": "Genus",
        "kingdom": kingdom,
        "valid_AphiaID": aphia_id,
    }


def test_every_accepted_genus_homonym_is_expanded(monkeypatch):
    """Each accepted genus record of a name is expanded, and its species carry
    the genus record they come from (Liparis: a fish and a plant). An
    unaccepted homonym is not expanded (Delphinapterus), unless the name has
    only unaccepted genus records: the first one holds the accepted species
    (Aphaniops)."""
    records = {
        "Liparis": [
            _genus(126160, "accepted"),
            _genus(503293, "unaccepted"),
            _genus(425907, "accepted", "Plantae"),
        ],
        "Delphinapterus": [_genus(383420, "unaccepted"), _genus(137029, "accepted")],
        "Aphaniops": [{**_genus(296017, "unaccepted"), "valid_AphiaID": 125725}],
    }
    monkeypatch.setattr(
        worms, "get_aphia_records", lambda name, session=None: records[name]
    )
    assert worms.get_genus_aphia_ids("Delphinapterus") == [137029]
    assert worms.get_genus_aphia_ids("Aphaniops") == [296017]

    species = {126160: "Liparis liparis", 425907: "Liparis loeselii"}
    monkeypatch.setattr(
        worms,
        "get_children_by_aphia_id",
        lambda aphia_id, **k: [
            {
                "valid_name": species[aphia_id],
                "valid_authority": "(Linnaeus)",
                "status": "accepted",
                "rank": "Species",
                "genus": "Liparis",
            }
        ],
    )
    df, failed = get_species_from_genus_list(["Liparis"], delay=0)
    assert failed == []
    assert dict(zip(df["scientificName"], df["genusID"])) == {
        "Liparis liparis": 126160,
        "Liparis loeselii": 425907,
    }


# --- species list of a genus ---


class _Page:
    def __init__(self, status, records=None):
        self.status_code = status
        self._records = records
        self.text = "x"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise worms.requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        return self._records


class _Session:
    """Answers each page request in turn: a _Page, or an exception to raise."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def get(self, *a, **k):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self):
        pass


def test_failed_page_fails_the_genus_instead_of_a_partial_list(monkeypatch):
    """A genus whose second page fails is reported failed (retried next run),
    and none of its species are kept as if the list were complete."""
    first_page = [
        {"valid_name": f"Gadus sp{i}", "status": "accepted", "rank": "Species"}
        for i in range(50)
    ]
    session = _Session(
        _Page(200, first_page), worms.requests.exceptions.ConnectionError("reset")
    )
    monkeypatch.setattr(worms, "create_session", lambda **k: session)
    monkeypatch.setattr(worms, "get_genus_aphia_ids", lambda *a, **k: [42])
    df, failed = get_species_from_genus_list(["Gadus"], delay=0)
    assert failed == ["Gadus"]
    assert df.empty


# --- NCBI taxid as a complement to the name ---


def _lookups(monkeypatch, by_name, by_taxid):
    """Fake the two WoRMS lookups; returns the taxids looked up."""
    calls = []

    def taxid_lookup(taxid, session=None):
        calls.append(taxid)
        return by_taxid

    monkeypatch.setattr(worms, "get_aphia_record", lambda name, session=None: by_name)
    monkeypatch.setattr(worms, "get_aphia_record_by_ncbi_taxid", taxid_lookup)
    return calls


def test_taxid_is_looked_up_only_when_the_name_is_not_found(monkeypatch):
    found = {
        "valid_name": "Gadus morhua",
        "valid_AphiaID": 126436,
        "status": "accepted",
    }
    calls = _lookups(monkeypatch, by_name=found, by_taxid=None)
    r = worms.resolve_accepted_name("Gadus morhua", ncbi_taxid=8049)
    assert r["matched_by"] == "name" and calls == []


def test_name_unknown_to_worms_is_found_through_its_taxid(monkeypatch):
    """WoRMS knows Microcarbo niger only as Phalacrocorax niger."""
    record = {
        "valid_name": "Phalacrocorax niger",
        "valid_AphiaID": 1,
        "status": "accepted",
        "rank": "Species",
    }
    calls = _lookups(monkeypatch, by_name=None, by_taxid=record)
    r = worms.resolve_accepted_name("Microcarbo niger", ncbi_taxid=56300)
    assert calls == [56300]
    assert r["accepted_name"] == "Phalacrocorax niger"
    assert r["matched_by"] == "ncbi_taxid"


def test_taxid_leading_to_a_subspecies_resolves_to_its_species(monkeypatch):
    record = {
        "valid_name": "Aporrectodea caliginosa trapezoides",
        "valid_AphiaID": 1425154,
        "status": "accepted",
        "rank": "Subspecies",
        "parentNameUsageID": 1041019,
    }
    _lookups(monkeypatch, by_name=None, by_taxid=record)
    r = worms.resolve_accepted_name("Aporrectodea trapezoides", ncbi_taxid=408844)
    assert (r["accepted_name"], r["aphia_id"]) == ("Aporrectodea caliginosa", 1041019)
