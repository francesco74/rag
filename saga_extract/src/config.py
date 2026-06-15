from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

def _require(name: str) -> str:
    """Get env var or raise a clear error."""
    val = os.getenv(name)
    if val is None or not str(val).strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val.strip()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as e:
        raise RuntimeError(f"Invalid integer for {name}: {raw!r}") from e


def _get_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for {name}: {raw!r}")


@dataclass(frozen=True)
class Settings:
    # Common
    http_timeout_seconds: int
    
    # DocWSRicerche
    docws_ricerca_endpoint: str
    docws_atti_endpoint: str
    docws_cid: Optional[str]
    docws_codice_amministrazione: str
    docws_codice_aoo: str
    utente_docws: str
    ruolo_docws: str

    # RepWSSGateway
    repwss_endpoint: str
    repwss_j2ee_username: str
    repwss_j2ee_password: str
    repwss_username: str
    repwss_password: str

    # Optional toggles
    verify_tls: bool = True
    soap_version: str = "1.1"  # default coerente con client attuale


def load_settings() -> Settings:
    """
    Loads .env and returns strongly-typed settings.

    env_path can be:
      - ".env" (default)
      - an absolute/relative path to a .env file
    """

    http_timeout_seconds = _get_int("HTTP_TIMEOUT_SECONDS", 30)
    
    # DocWSRicerche
    docws_ricerca_endpoint = _require("DOCWS_RICERCA_ENDPOINT")
    docws_atti_endpoint = _require("DOCWS_ATTI_ENDPOINT")
    docws_cid = os.getenv("DOCWS_CID")
    if docws_cid is not None:
        docws_cid = docws_cid.strip() or None

    docws_codice_amministrazione = _require("DOCWS_CODICE_AMMINISTRAZIONE")
    docws_codice_aoo = _require("DOCWS_CODICE_AOO")

    # Nota: nel tuo .env ci sono con spazi attorno a '='. Li gestiamo con strip.
    utente_docws = _require("UTENTE_DOCWS")
    ruolo_docws = _require("RUOLO_DOCWS")

    # RepWSSGateway
    repwss_endpoint = _require("REPWSS_ENDPOINT")
    repwss_j2ee_username = _require("REPWSS_J2EE_USERNAME")
    repwss_j2ee_password = _require("REPWSS_J2EE_PASSWORD")
    repwss_username = _require("REPWSS_USERNAME")
    repwss_password = _require("REPWSS_PASSWORD")

    # Opzionali (se vuoi aggiungerli nel .env in futuro)
    verify_tls = _get_bool("VERIFY_TLS", True)
    soap_version = os.getenv("SOAP_VERSION", "1.1").strip() or "1.1"

    return Settings(
        http_timeout_seconds=http_timeout_seconds,
        
        docws_ricerca_endpoint=docws_ricerca_endpoint,
        docws_atti_endpoint=docws_atti_endpoint,
        docws_cid=docws_cid,
        docws_codice_amministrazione=docws_codice_amministrazione,
        docws_codice_aoo=docws_codice_aoo,
        utente_docws=utente_docws,
        ruolo_docws=ruolo_docws,

        repwss_endpoint=repwss_endpoint,
        repwss_j2ee_username=repwss_j2ee_username,
        repwss_j2ee_password=repwss_j2ee_password,
        repwss_username=repwss_username,
        repwss_password=repwss_password,

        verify_tls=verify_tls,
        soap_version=soap_version,
    )


# Comodità: import diretto "from config import settings"
settings = load_settings()