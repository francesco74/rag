import codecs
import logging
import re
from typing import Optional


_SENTINEL_DATES_ISO = {"0001-01-01"}
log = logging.getLogger("UTILITY")
_WS_RE = re.compile(r'(?:&nbsp;|&#160;|\xa0|\s)+')

_SOAP_DECODE_HANDLER_NAME = "cp1252_fallback"


def _cp1252_fallback_handler(exc: UnicodeDecodeError):
    """
    Error handler per bytes.decode(): invece di scartare (errors='replace',
    che produce '\ufffd' e perde il dato originale) o fallire
    (errors='strict'), reinterpreta SOLO i byte non validi come UTF-8
    usando cp1252/Windows-1252 (l'encoding legacy con cui Sicr@Web a volte
    inietta caratteri come '°' = 0xB0 dentro payload altrimenti UTF-8).

    Logga ogni volta che scatta, con posizione/byte grezzi/risultato: serve
    a verificare empiricamente (guardando i log del pod) se il fallback
    viene davvero esercitato, e su quali byte esatti.
    """
    bad_bytes = exc.object[exc.start:exc.end]
    recovered = bad_bytes.decode("cp1252", errors="replace")
    log.warning(
        "decode_soap_response: byte non valido UTF-8 a offset %d "
        "(hex=%s) -> reinterpretato come cp1252: %r",
        exc.start, bad_bytes.hex(), recovered,
    )
    return recovered, exc.end


# Registrazione idempotente a import-time: se il modulo viene importato più
# volte non ci sono effetti collaterali, resta un solo handler per nome.
codecs.register_error(_SOAP_DECODE_HANDLER_NAME, _cp1252_fallback_handler)

# Sequenza UTF-8 del carattere di sostituzione Unicode U+FFFD ('�').
_UFFFD_UTF8_BYTES = "\ufffd".encode("utf-8")

# Entità numeriche XML che, una volta risolte da ET.fromstring(), producono
# U+FFFD pur essendo testo ASCII perfettamente valido nei byte grezzi (per
# questo decode_soap_response da sola non le rileverebbe come anomalia).
_UFFFD_XML_ENTITIES = (b"&#65533;", b"&#xFFFD;", b"&#xfffd;")


def decode_soap_response(raw_bytes: bytes) -> str:
    """
    Decodifica il body grezzo (bytes) di una risposta SOAP Sicr@Web.
    Da usare SEMPRE al posto di resp.text (requests) quando la risposta può
    contenere testo libero immesso dagli utenti del gestionale (oggetti
    atto, nomi allegato, ecc.): resp.text decodifica con errors='replace' e
    sostituisce silenziosamente i byte cp1252 legacy con '\ufffd' ('�'),
    perdendo il dato senza loggare nulla. Qui invece i byte problematici
    vengono recuperati come cp1252 (vedi _cp1252_fallback_handler).

    NB DIAGNOSTICO: se anche DOPO questo fallback il risultato contiene
    ancora '\ufffd', ci sono TRE possibilità distinte, distinguibili solo
    guardando i byte grezzi (motivo per cui logghiamo qui sotto):
      1) il fallback non è affatto scattato per quel punto (nessun log
         "byte non valido UTF-8" per quella posizione) -> va capito perché;
      2) il payload conteneva GIA' la sequenza UTF-8 di U+FFFD (byte
         EF BF BD) -> perso a monte, irrecuperabile;
      3) il payload conteneva un'ENTITA' NUMERICA XML (es. "&#65533;"),
         testo ASCII perfettamente valido che ET.fromstring() risolverà
         poi in U+FFFD durante il parsing: anche in questo caso il dato è
         perso a monte (lato Sicr@Web), solo con un meccanismo diverso.
    """
    if raw_bytes is None:
        return ""

    decoded = raw_bytes.decode("utf-8", errors=_SOAP_DECODE_HANDLER_NAME)

    entita_trovate = [ent for ent in _UFFFD_XML_ENTITIES if ent in raw_bytes]
    if entita_trovate:
        idx = raw_bytes.find(entita_trovate[0])
        contesto_raw = raw_bytes[max(0, idx - 40): idx + 40]
        log.error(
            "decode_soap_response: trovata entità numerica XML %s nei byte "
            "grezzi (equivalente a U+FFFD una volta risolta da ET.fromstring): "
            "il carattere originale è andato perso A MONTE (lato Sicr@Web) "
            "con lo stesso meccanismo (errors='replace') poi ri-serializzato "
            "come entità invece che come byte diretto. NON recuperabile lato "
            "client. Contesto byte grezzi intorno al match: %r",
            entita_trovate[0], contesto_raw,
        )

    if "\ufffd" in decoded:
        occorrenze_gia_presenti = raw_bytes.count(_UFFFD_UTF8_BYTES)
        if occorrenze_gia_presenti:
            idx = decoded.find("\ufffd")
            contesto = decoded[max(0, idx - 30): idx + 30]
            log.error(
                "decode_soap_response: il payload grezzo conteneva GIA' "
                "%d volte la sequenza UTF-8 di U+FFFD (EF BF BD), quindi il "
                "carattere originale è stato perso A MONTE (lato Sicr@Web) "
                "e NON è recuperabile lato client. Contesto: %r",
                occorrenze_gia_presenti, contesto,
            )
        elif not entita_trovate:
            log.warning(
                "decode_soap_response: '\ufffd' presente nell'output ma "
                "NON come sequenza EF BF BD né come entità XML nota nei "
                "byte grezzi: causa non ancora identificata, servono altri dati."
            )

    return decoded


def normalize_ws(text: str) -> str:
    """Collassa entità HTML di spaziatura e whitespace ripetuto in un solo
    spazio. Applicata al momento della query (rerank e rendering), NON in
    ingestione: i vettori in Qdrant restano calcolati sul testo originale."""
    if not text:
        return text
    return _WS_RE.sub(' ', text).strip()

def clean_iso_date(date_raw: Optional[str]) -> Optional[str]:
    """
    Normalizza una data proveniente da Sicr@Web al formato ISO 'YYYY-MM-DD'.

    Gestisce:
    - ISO con tempo/timezone, es. "2025-11-26T00:00:00Z" -> "2025-11-26"
    - ISO già "pulita" "2025-11-26" -> invariata
    - Italiano "DD/MM/YYYY", es. "26/11/2025" -> "2025-11-26"
    - Sentinella Sicr@Web "0001-01-01..." (usata per campi come
      DataEsecutivita quando l'atto non è ancora esecutivo: NON è una
      data reale) -> None
    - None/stringa vuota -> None

    Se il formato non è nessuno dei precedenti, logga un warning e
    restituisce None invece di propagare silenziosamente un valore
    inaffidabile (comportamento della vecchia implementazione).
    """
    if not date_raw:
        return None
    date_raw = date_raw.strip()
    if not date_raw:
        return None

    # ISO, con o senza componente oraria/timezone ("T..." o " ...")
    date_part = date_raw.split("T")[0].split(" ")[0]
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_part):
        if date_part in _SENTINEL_DATES_ISO:
            return None
        return date_part

    # Italiano "DD/MM/YYYY"
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_raw)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month}-{day}"

    log.warning("clean_iso_date: formato data non riconosciuto, scartato: %r", date_raw)
    return None