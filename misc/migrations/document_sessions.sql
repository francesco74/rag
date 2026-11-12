-- =============================================================================
-- Sessioni di documenti delle API esterne (api/src/external_sessions.py)
-- =============================================================================
-- Nuove tabelle, nessun dato esistente toccato: si può rilanciare.
-- I dati sono temporanei (scadono con la sessione): le tabelle restano fuori
-- dai backup, con mysqldump --ignore-table=<db>.session_parents ecc.
--
--   k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag' \
--       < misc/migrations/document_sessions.sql
-- =============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    -- Contenitore temporaneo di documenti di una chiave esterna. owner è scelto
    -- dal sistema esterno (atti-pa: impronta di ente e utente) e non si interpreta.
    id          CHAR(32)     NOT NULL PRIMARY KEY,
    key_id      INT          NOT NULL,
    owner       VARCHAR(128) NOT NULL DEFAULT '',
    created_at  DATETIME     NOT NULL,
    expires_at  DATETIME     NOT NULL,
    KEY ix_sessions_expires (expires_at),
    CONSTRAINT fk_sessions_key FOREIGN KEY (key_id)
        REFERENCES external_api_keys (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS session_documents (
    -- status: queued, converting, indexing, ready, failed (con error).
    -- tokens: stima del testo convertito (cl100k), per l'indicatore nell'app.
    id          CHAR(32)     NOT NULL PRIMARY KEY,
    session_id  CHAR(32)     NOT NULL,
    file_name   VARCHAR(255) NOT NULL,
    status      VARCHAR(16)  NOT NULL,
    error       VARCHAR(1000) NULL,
    tokens      INT          NULL,
    sections    INT          NULL,
    created_at  DATETIME     NOT NULL,
    KEY ix_session_documents_session (session_id),
    CONSTRAINT fk_session_documents_session FOREIGN KEY (session_id)
        REFERENCES sessions (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS session_parents (
    -- Sezioni (parent) dei documenti, in ordine: i pezzi (child) sono in Qdrant,
    -- collection session_chunks, con doc_id e parent_index.
    session_id    CHAR(32)    NOT NULL,
    doc_id        CHAR(32)    NOT NULL,
    parent_index  INT         NOT NULL,
    content       MEDIUMTEXT  NOT NULL,
    metadata      JSON        NULL,
    PRIMARY KEY (doc_id, parent_index),
    KEY ix_session_parents_session (session_id),
    CONSTRAINT fk_session_parents_document FOREIGN KEY (doc_id)
        REFERENCES session_documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Verifica
SELECT table_name, table_rows FROM information_schema.tables
WHERE table_schema = DATABASE() AND table_name IN ('sessions', 'session_documents', 'session_parents');
