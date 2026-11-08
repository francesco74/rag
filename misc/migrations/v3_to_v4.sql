-- =============================================================================
-- Migrazione del database dalla v3 alla v4
-- =============================================================================
--
-- Procedura completa: misc/migrations/MIGRAZIONE_V3_V4.md (passo 4).
-- Da eseguire con l'API e i servizi fermi, dopo il dump del database:
--
--   kubectl -n rag exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag' \
--       < misc/migrations/v3_to_v4.sql
--
-- Parte 1: tabelle nuove (CREATE TABLE IF NOT EXISTS, nessun dato esistente toccato).
-- Parte 2: conversione dei ruoli della revisione, UNA SOLA VOLTA. Il segnaposto in
--          schema_migrations lo garantisce: se la conversione è già stata fatta,
--          l'INSERT fallisce con "Duplicate entry 'v4_ruoli_revisione'", il client
--          mysql si ferma e i ruoli restano come sono.
-- Parte 3 (solo lettura) è nel file a parte v3_to_v4_verifica.sql.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- PARTE 1 — Tabelle nuove della v4
-- -----------------------------------------------------------------------------

-- Siti della chat: il frontend è unico e riconosce il sito dal nome di dominio.
CREATE TABLE IF NOT EXISTS chat_sites (
    hostname           VARCHAR(255) NOT NULL PRIMARY KEY,
    topic_id           VARCHAR(100) NOT NULL,
    name               VARCHAR(255) NOT NULL,
    allow_date_filter  BOOLEAN      NOT NULL DEFAULT FALSE,
    allow_subtopic_selection BOOLEAN NOT NULL DEFAULT TRUE,
    active             BOOLEAN      NOT NULL DEFAULT TRUE,
    welcome_it         MEDIUMTEXT,
    welcome_en         MEDIUMTEXT,
    -- Pulsanti della home: JSON [{label_it, label_en, url}]
    home_buttons       TEXT,
    icon               MEDIUMBLOB,
    created_at         TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_chat_sites_topic UNIQUE (topic_id),
    CONSTRAINT fk_chat_sites_topic FOREIGN KEY (topic_id) REFERENCES topics (topic_id) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- File caricati dal sito di amministrazione e loro stato nella pipeline.
CREATE TABLE IF NOT EXISTS document_uploads (
    id            INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    topic_id      VARCHAR(100) NOT NULL,
    sub_topic_id  VARCHAR(100) NOT NULL,
    source        VARCHAR(700) NOT NULL,
    file_name     VARCHAR(255) NOT NULL,
    size_bytes    BIGINT       NOT NULL,
    sha256        CHAR(64)     NOT NULL,
    -- in_coda, da_inviare, in_lavorazione, indicizzato, errore
    status        VARCHAR(20)  NOT NULL,
    error         VARCHAR(1000),
    uploaded_by   VARCHAR(100) NOT NULL,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_uploads_source UNIQUE (source),
    KEY idx_document_uploads_hash (topic_id, sub_topic_id, sha256),
    KEY idx_document_uploads_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Pagine trascritte con l'OCR per ogni documento: le aggiorna il converter.
-- I documenti indicizzati in v3 non hanno riga: dato non noto.
CREATE TABLE IF NOT EXISTS document_ocr (
    source        VARCHAR(700) NOT NULL,
    topic_id      VARCHAR(100) NOT NULL,
    sub_topic_id  VARCHAR(100) NOT NULL,
    ocr_pages     INT          NOT NULL,
    total_pages   INT          NOT NULL,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_ocr UNIQUE (source(500), topic_id, sub_topic_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Credenziali dei sistemi esterni (python -m utils.manage_api_keys).
CREATE TABLE IF NOT EXISTS external_api_keys (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    key_hash      CHAR(64)     NOT NULL,
    key_prefix    VARCHAR(16)  NOT NULL,
    active        BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMP    NULL DEFAULT NULL,
    last_used_at  TIMESTAMP    NULL DEFAULT NULL,
    UNIQUE KEY uq_external_name (name),
    UNIQUE KEY uq_external_hash (key_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS external_api_key_topics (
    key_id    INT          NOT NULL,
    topic_id  VARCHAR(100) NOT NULL,
    PRIMARY KEY (key_id, topic_id),
    CONSTRAINT fk_external_key FOREIGN KEY (key_id)
        REFERENCES external_api_keys (id) ON DELETE CASCADE,
    CONSTRAINT fk_external_topic FOREIGN KEY (topic_id)
        REFERENCES topics (topic_id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Tentativi di login falliti (blocco dopo LOGIN_MAX_FAILURES).
CREATE TABLE IF NOT EXISTS review_login_failures (
    lkey            VARCHAR(160) NOT NULL PRIMARY KEY,
    failed_count    INT          NOT NULL DEFAULT 0,
    last_failed_at  TIMESTAMP    NULL DEFAULT NULL,
    locked_until    TIMESTAMP    NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Migrazioni di dati già eseguite: impediscono una seconda esecuzione.
CREATE TABLE IF NOT EXISTS schema_migrations (
    name         VARCHAR(100) NOT NULL PRIMARY KEY,
    executed_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- -----------------------------------------------------------------------------
-- PARTE 2 — Ruoli della revisione (una sola volta)
-- -----------------------------------------------------------------------------
--   revisore, catalogatore -> archivista   (il catalogatore guadagna il testo)
--   correttore, validatore -> revisore     (il validatore guadagna il testo)
--   lettore                -> nessun ruolo (non esiste più la sola lettura)
--   admin                  -> invariato
-- L'ordine conta: il vecchio "revisore" poteva già fare tutto, quindi diventa
-- archivista PRIMA che i correttori diventino revisori.

-- Prima della conversione: chi resterà senza ruoli (solo "lettore"). Dopo la
-- migrazione non potrà più entrare finché l'amministratore non gli assegna un ruolo.
SELECT u.username AS resta_senza_ruoli
FROM review_users u
WHERE EXISTS (SELECT 1 FROM review_user_roles r WHERE r.user_id = u.id)
  AND NOT EXISTS (SELECT 1 FROM review_user_roles r
                  WHERE r.user_id = u.id AND r.role <> 'lettore');

START TRANSACTION;

-- Si ferma qui (Duplicate entry) se la conversione è già stata fatta.
INSERT INTO schema_migrations (name) VALUES ('v4_ruoli_revisione');

INSERT IGNORE INTO review_user_roles (user_id, role)
    SELECT user_id, 'archivista' FROM review_user_roles WHERE role IN ('revisore', 'catalogatore');
DELETE FROM review_user_roles WHERE role = 'revisore';

INSERT IGNORE INTO review_user_roles (user_id, role)
    SELECT user_id, 'revisore' FROM review_user_roles WHERE role IN ('correttore', 'validatore');
DELETE FROM review_user_roles WHERE role IN ('lettore', 'correttore', 'catalogatore', 'validatore');

COMMIT;

-- Risultato: ruoli per utente (solo revisore, archivista, admin).
SELECT u.username, GROUP_CONCAT(r.role ORDER BY r.role) AS ruoli
FROM review_users u
LEFT JOIN review_user_roles r ON r.user_id = u.id
GROUP BY u.id, u.username
ORDER BY u.username;
