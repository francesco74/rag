-- =============================================================================
-- Revisione e chat: protezione del login dai tentativi ripetuti
-- =============================================================================
--
-- Nuova tabella, nessun dato esistente toccato. Da eseguire prima di
-- aggiornare l'API: il login la usa a ogni tentativo.
-- Parametri: LOGIN_MAX_FAILURES (5), LOGIN_LOCKOUT_SECONDS (900),
-- LOGIN_IP_MAX_FAILURES (20, solo con TRUSTED_PROXY_HOPS > 0).
-- =============================================================================

CREATE TABLE IF NOT EXISTS review_login_failures (
    -- Tentativi di login falliti di fila, per chiave: "user:<nome utente>"
    -- (anche per nomi inesistenti) e, con TRUSTED_PROXY_HOPS > 0,
    -- "ip:<indirizzo>". Arrivati al massimo la chiave è bloccata fino a
    -- locked_until (LOGIN_LOCKOUT_SECONDS). Un login riuscito cancella le
    -- righe; utils/manage_review_users.py unlock toglie un blocco a mano.
    lkey            VARCHAR(160) NOT NULL PRIMARY KEY,
    failed_count    INT          NOT NULL DEFAULT 0,
    last_failed_at  TIMESTAMP    NULL DEFAULT NULL,
    locked_until    TIMESTAMP    NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
