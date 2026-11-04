"""Test della normalizzazione di utils/align_dates.py (nessun servizio richiesto).

Uso:  python api/tests/test_align_dates.py      (esce con 1 se qualcosa fallisce)
"""
import os, pathlib, sys, tempfile

os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="k", DATA_FOLDER=tempfile.mkdtemp(), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils.align_dates import align_metadata, parse_date, payload_changes

failures = 0


def check(label, got, expected):
    global failures
    if got != expected:
        failures += 1
        print(f"FALLITO {label}: {got!r} invece di {expected!r}")
    else:
        print(f"ok {label}")


for value, expected in [
    ("2025-11-26", ("iso", "2025-11-26")),
    ("2025-11-26T00:00:00Z", ("iso", "2025-11-26")),
    ("2025-11-26 10:30:00", ("iso", "2025-11-26")),
    ("26/11/2025", ("iso", "2025-11-26")),
    ("26-11-2025", ("iso", "2025-11-26")),
    ("26.11.2025", ("iso", "2025-11-26")),
    ("3/2/2025", ("iso", "2025-02-03")),
    ("2025/11/26", ("iso", "2025-11-26")),
    ("20251126", ("iso", "2025-11-26")),
    ("26 novembre 2025", ("iso", "2025-11-26")),
    ("1° gen 2020", ("iso", "2020-01-01")),
    ("29/08/62", ("iso", "1962-08-29")),
    ("2024", ("partial", "2024")),
    (2024, ("partial", "2024")),
    ("03/2024", ("partial", "2024")),
    ("2024-03", ("partial", "2024")),
    ("marzo 2024", ("partial", "2024")),
    ("0001-01-01", ("sentinel", None)),
    ("0001-01-01T00:00:00Z", ("sentinel", None)),
    ("31/02/2024", ("invalid", None)),
    ("non so", ("invalid", None)),
    ("", ("empty", None)),
    (None, ("empty", None)),
]:
    check(f"parse_date({value!r})", parse_date(value), expected)

new, events = align_metadata({"data": "26/11/2025", "data_esecutivita": "0001-01-01T00:00:00Z",
                              "data_pubblicazione": "2024", "database": "non è una data", "oggetto": "x"})
check("align: risultato", new, {"data": "2025-11-26", "anno": "2024", "database": "non è una data", "oggetto": "x"})
check("align: eventi", [e[2] for e in events], ["iso", "sentinel", "partial"])
check("align: anno esistente non sovrascritto", align_metadata({"data": "2024", "anno": "1999"})[0], {"anno": "1999"})
check("align: già allineato", align_metadata({"data": "2025-11-26"}), ({"data": "2025-11-26"}, []))
check("align: lista", align_metadata({"data_x": ["26/11/2025", "boh"]})[0], {"data_x": ["2025-11-26"]})
check("align: idempotente", align_metadata(new)[1], [])
check("payload_changes", payload_changes({"a": 1, "b": 2}, {"a": 1, "c": 3}), ({"c": 3}, ["b"]))

print(f"\n{'TUTTO OK' if not failures else f'{failures} FALLITI'}")
sys.exit(1 if failures else 0)
