from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # App Config
    http_timeout_seconds: int
    data_folder: str
    log_level: str
    
    # RabbitMQ
    broker_host: str
    broker_port: int
    broker_username: str
    broker_password: str

    # DocWSRicerche
    docws_ricerca_endpoint: str
    docws_atti_endpoint: str
    docws_codice_amministrazione: str
    docws_codice_aoo: str
    ws_username: str
    ws_password: str
    ruolo_docws: str

    verify_tls: bool
    soap_version: str


def load_settings() -> Settings:
    # L'operatore "or" protegge dalle stringhe vuote. 
    # Es: se HTTP_TIMEOUT_SECONDS="", os.getenv() o 30 restituisce 30.
    
    return Settings(
        http_timeout_seconds=int(os.getenv("HTTP_TIMEOUT_SECONDS") or 30),
        data_folder=os.getenv("DATA_FOLDER") or str(Path(__file__).parent.resolve()),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        
        broker_host=os.getenv("BROKER_HOST", "rabbitmq-service.rag.svc.cluster.local"),
        broker_port=int(os.getenv("BROKER_PORT") or 5672),
        broker_username=os.getenv("BROKER_USERNAME", "guest"),
        broker_password=os.getenv("BROKER_PASSWORD", "guest"),
        
        docws_ricerca_endpoint=os.environ["DOCWS_RICERCA_ENDPOINT"],
        docws_atti_endpoint=os.environ["DOCWS_ATTI_ENDPOINT"],
        ws_username=os.getenv("WS_USERNAME", "").strip() or None,
        ws_password=os.getenv("WS_PASSWORD", "").strip() or None,
        docws_codice_amministrazione=os.environ["DOCWS_CODICE_AMMINISTRAZIONE"],
        docws_codice_aoo=os.environ["DOCWS_CODICE_AOO"],
        ruolo_docws=os.environ["RUOLO_DOCWS"],

        # Valuta correttamente i booleani confrontando la stringa
        verify_tls=os.getenv("VERIFY_TLS", "True").lower() in ("true", "1", "yes"),
        soap_version=os.getenv("SOAP_VERSION", "1.1"),
    )

settings = load_settings()