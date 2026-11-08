import codecs
import hashlib
import logging
import re
import secrets
from typing import Optional

from pa_common.testo import markdown_in_testo, normalizza_spazi, pulisci_ocr


_SENTINEL_DATES_ISO = {"0001-01-01"}
log = logging.getLogger("UTILITY")

# Lo scrive il converter, che ha chiavi e credenziali: l'API sa così se l'OCR esterno è utilizzabile.
EXTERNAL_OCR_MARKER = "converter/ocr_esterno_disponibile"

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


# Librerie che a INFO/DEBUG scrivono una riga per ogni richiesta HTTP (Qdrant,
# Gemini, OpenAI...): con LOG_LEVEL=DEBUG erano la gran parte del log.
NOISY_HTTP_LOGGERS = ("httpx", "httpcore")


def quiet_http_loggers():
    """Porta i logger HTTP a WARNING qualunque sia LOG_LEVEL: restano solo gli errori."""
    for name in NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# Pulizia del testo: in pa-common (pa_common.testo, condiviso con atti-pa).
#   normalize_ws: entità HTML di spaziatura e spazi ripetuti in un solo spazio. Applicata al
#     momento della query (rerank e rendering), NON in ingestione: i vettori in Qdrant
#     restano calcolati sul testo originale.
#   clean_ocr_text: toglie dal testo OCR gli spazi usati per imitare l'impaginazione, senza
#     perdere la struttura Markdown né le colonne di dati.
#   markdown_to_text: testo semplice dal Markdown, le righe di tabella diventano colonne.
normalize_ws = normalizza_spazi
clean_ocr_text = pulisci_ocr
markdown_to_text = markdown_in_testo


def is_permanent_error(error: Exception) -> bool:
    """Richiesta rifiutata dal servizio (modello inesistente, chiave o dimensione errata): ritentare non serve."""
    # Il provider Gemini incapsula ogni errore: lo stato è in quello originale.
    error = error.__cause__ or error
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        status = getattr(error, "code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 425, 429)



# Coda dei lavori di conversione: la usano API, converter e setup_rabbitmq, che
# devono dichiararla con argomenti identici (RabbitMQ rifiuta le differenze).
CONVERSION_QUEUE = "da-convertire-api"
CONVERSION_QUEUE_ARGS = {"x-queue-type": "quorum"}


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

def split_protected_metadata(metadata: dict, protected_keys) -> tuple[dict, list]:
    """
    Separa i metadati di un documento dalle chiavi di sistema
    (settings.protected_keys: content, source, topic_id, ...).

    Ritorna (metadati_puliti, chiavi_scartate). Le chiavi di sistema non sono
    metadati del documento: non vanno mostrate né modificate in revisione,
    né accettate da un manifest, né copiate nei metadati di MySQL. Usata da
    ingest, revisione e script di migrazione, così la regola è una sola.
    """
    if not isinstance(metadata, dict):
        return {}, []
    clean = {k: v for k, v in metadata.items() if k not in protected_keys}
    dropped = sorted(k for k in metadata if k in protected_keys)
    return clean, dropped


def new_external_key() -> tuple[str, str, str]:
    """Nuova chiave delle API esterne: (chiave, hash da conservare, prefisso riconoscibile)."""
    key = "rag_" + secrets.token_urlsafe(32)
    return key, hashlib.sha256(key.encode()).hexdigest(), key[:12]
