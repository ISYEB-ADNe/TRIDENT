"""Tests for the WoRMS client: failures, name resolution and genus
expansion, without network (real answers as fixtures)."""

import pytest

from trident.clients import worms
from trident.clients.worms import get_species_from_genus_list


@pytest.mark.parametrize(
    "aphia_ids, children, failed",
    [
        (RuntimeError("network error"), None, ["Gadus"]),
        ([], None, []),
        ([12345], [], []),
    ],
    ids=["errored", "not in worms", "no children"],
)
def test_only_an_errored_genus_is_a_failure(monkeypatch, aphia_ids, children, failed):
    """A genus WoRMS does not know, or knows without species, is a valid empty
    (not retried); the taxonomy columns are there either way."""

    def genus_ids(*a, **k):
        if isinstance(aphia_ids, Exception):
            raise aphia_ids
        return aphia_ids

    monkeypatch.setattr(worms, "get_genus_aphia_ids", genus_ids)
    monkeypatch.setattr(worms, "get_children_by_aphia_id", lambda *a, **k: children)
    df, got = get_species_from_genus_list(["Gadus"], delay=0)
    assert got == failed
    assert df.empty
    assert list(df.columns) == worms.WORMS_TAXON_COLUMNS


def test_authorship_keeps_its_parentheses(monkeypatch):
    """WoRMS's authority reaches the TAX list as written: the parentheses of a
    species moved to another genus are part of it (Darwin Core)."""
    record = {
        "AphiaID": 137084,
        "valid_AphiaID": 137084,
        "scientificname": "Odobenus rosmarus",
        "valid_name": "Odobenus rosmarus",
        "valid_authority": "(Linnaeus,  1758)",
        "status": "accepted",
        "rank": "Species",
        "kingdom": "Animalia",
        "genus": "Odobenus",
        "url": "https://www.marinespecies.org/aphia.php?p=taxdetails&id=137084",
    }
    monkeypatch.setattr(worms, "get_genus_aphia_ids", lambda *a, **k: [137005])
    monkeypatch.setattr(worms, "get_children_by_aphia_id", lambda *a, **k: [record])
    df, _ = get_species_from_genus_list(["Odobenus"], delay=0)
    assert df["scientificNameAuthorship"].tolist() == ["(Linnaeus, 1758)"]


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


# --- name resolution ---


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

    assert worms.get_aphia_records("Ammodytes sp.", session=Sess()) == []


def test_answer_that_is_not_json_is_an_error_not_a_not_found():
    """A 200 answer that is not JSON (e.g. a maintenance page) raises, so the
    name is reported failed and asked again, not saved as unknown to WoRMS."""

    class MaintenancePage(_Page):
        def json(self):
            raise worms.requests.exceptions.JSONDecodeError(
                "Expecting value", "<html>", 0
            )

    with pytest.raises(worms.requests.exceptions.JSONDecodeError):
        worms.get_aphia_records("Gadus", session=_Session(MaintenancePage(200)))


@pytest.mark.parametrize(
    "name, records, expected",
    [
        (
            "Gadus morhua",
            [
                {
                    "valid_name": "Gadus morhua",
                    "valid_AphiaID": 126436,
                    "status": "accepted",
                }
            ],
            ("Gadus morhua", 126436, False),
        ),
        (
            "Gadus ogac",
            [
                {
                    "valid_name": "Gadus macrocephalus",
                    "valid_AphiaID": 126437,
                    "status": "unaccepted",
                }
            ],
            ("Gadus macrocephalus", 126437, True),
        ),
        ("Ammodytes sp.", [], ("Ammodytes sp.", None, False)),
    ],
    ids=["accepted", "synonym", "not found"],
)
def test_name_resolves_to_its_accepted_name_or_stays_as_written(
    monkeypatch, name, records, expected
):
    monkeypatch.setattr(worms, "get_aphia_records", lambda name, session=None: records)
    r = worms.resolve_accepted_name(name)
    assert (r["accepted_name"], r["aphia_id"], r["is_synonym"]) == expected


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


def test_name_whose_records_lead_to_several_species_stays_as_ncbi_names_it(
    monkeypatch,
):
    """WoRMS's records of Delphinus capensis (real answer): none is
    accepted, two synonyms of different species and a nomen dubium without an
    accepted name. The name is not resolved through the first record, and the
    reason is kept."""
    records = [
        {
            "AphiaID": 383816,
            "scientificname": "Delphinus capensis",
            "authority": "Rapp, 1837",
            "status": "nomen dubium",
            "rank": "Species",
            "valid_AphiaID": None,
            "valid_name": None,
        },
        {
            "AphiaID": 383815,
            "scientificname": "Delphinus capensis",
            "authority": "G. Cuvier, 1829",
            "status": "unaccepted",
            "rank": "Species",
            "valid_AphiaID": 254978,
            "valid_name": "Cephalorhynchus heavisidii",
        },
        {
            "AphiaID": 137093,
            "scientificname": "Delphinus capensis",
            "authority": "Gray, 1828",
            "status": "unaccepted",
            "rank": "Species",
            "valid_AphiaID": 137094,
            "valid_name": "Delphinus delphis",
        },
    ]
    monkeypatch.setattr(worms, "get_aphia_records", lambda name, session=None: records)
    monkeypatch.setattr(worms, "get_aphia_record_by_ncbi_taxid", lambda *a, **k: None)
    r = worms.resolve_accepted_name("Delphinus capensis", ncbi_taxid=103584)
    assert r["accepted_name"] == "Delphinus capensis"
    assert r["aphia_id"] is None and r["matched_by"] is None
    assert r["note"] == (
        "ambiguous in WoRMS: synonym of Cephalorhynchus heavisidii or synonym of "
        "Delphinus delphis"
    )

    # Without the Cuvier record, the synonyms lead to one species.
    del records[1]
    r = worms.resolve_accepted_name("Delphinus capensis")
    assert r["accepted_name"] == "Delphinus delphis" and r["aphia_id"] == 137094


# --- NCBI taxid as a complement to the name ---


def _lookups(monkeypatch, by_name, by_taxid):
    """Fake the two WoRMS lookups; returns the taxids looked up."""
    calls = []

    def taxid_lookup(taxid, session=None):
        calls.append(taxid)
        return by_taxid

    monkeypatch.setattr(
        worms,
        "get_aphia_records",
        lambda name, session=None: [by_name] if by_name else [],
    )
    monkeypatch.setattr(worms, "get_aphia_record_by_ncbi_taxid", taxid_lookup)
    return calls


def test_taxid_is_looked_up_only_for_a_name_unknown_to_worms(monkeypatch):
    """WoRMS knows Microcarbo niger only as Phalacrocorax niger; Gadus morhua
    is found by its name, so its taxid is not asked."""
    found = {
        "valid_name": "Gadus morhua",
        "valid_AphiaID": 126436,
        "status": "accepted",
    }
    calls = _lookups(monkeypatch, by_name=found, by_taxid=None)
    r = worms.resolve_accepted_name("Gadus morhua", ncbi_taxid=8049)
    assert r["matched_by"] == "name" and calls == []

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


@pytest.mark.parametrize(
    "record",
    [
        {
            "scientificname": "Gadus",
            "AphiaID": 125732,
            "status": "accepted",
            "rank": "Genus",
        },
        {
            "scientificname": "Gadus ogac",
            "AphiaID": 1,
            "status": "nomen dubium",
            "rank": "Species",
        },
    ],
    ids=["genus", "no accepted name"],
)
def test_taxid_record_is_used_only_when_it_leads_to_an_accepted_species(
    monkeypatch, record
):
    """A record linked to the taxid resolves as a name's record does: one
    without an accepted species leaves the name as NCBI writes it."""
    _lookups(monkeypatch, by_name=None, by_taxid=record)
    r = worms.resolve_accepted_name("Gadus ogac", ncbi_taxid=8053)
    assert (r["accepted_name"], r["aphia_id"], r["matched_by"]) == (
        "Gadus ogac",
        None,
        None,
    )
