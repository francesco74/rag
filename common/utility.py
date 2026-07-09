
import logging
import re
from typing import Optional


_SENTINEL_DATES_ISO = {"0001-01-01"}
log = logging.getLogger("UTILITY")

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