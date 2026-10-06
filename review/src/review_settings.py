"""Impostazioni specifiche del servizio di revisione (variabili d'ambiente)."""

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewSettings:
    secret_key: str
    session_hours: int
    file_token_minutes: int
    allowed_origins: list[str]


def _load() -> ReviewSettings:
    secret = os.environ.get("REVIEW_SECRET_KEY", "").strip()
    if len(secret) < 32:
        # Con una chiave debole o assente chiunque potrebbe forgiare un token
        # di sessione: meglio non partire che partire insicuri.
        print("CRITICAL: REVIEW_SECRET_KEY mancante o più corta di 32 caratteri.", file=sys.stderr)
        raise SystemExit(1)

    origins_raw = os.environ.get("REVIEW_ALLOWED_ORIGINS") or os.environ.get("ALLOWED_ORIGINS") or "*"
    return ReviewSettings(
        secret_key=secret,
        session_hours=int(os.environ.get("REVIEW_SESSION_HOURS") or 10),
        file_token_minutes=int(os.environ.get("REVIEW_FILE_TOKEN_MINUTES") or 30),
        allowed_origins=[o.strip() for o in origins_raw.split(",") if o.strip()],
    )


review_settings = _load()
