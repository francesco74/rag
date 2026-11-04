-- =============================================================================
-- API per i sistemi esterni: credenziali e topic abilitati
-- =============================================================================
-- Nuove tabelle, nessun dato esistente toccato. Le chiavi si gestiscono con
-- python -m utils.manage_api_keys
-- =============================================================================

CREATE TABLE IF NOT EXISTS external_api_keys (
    -- Credenziali dei sistemi esterni (api/src/external_routes.py). Della chiave
    -- si conserva solo l'hash SHA-256: è casuale e lunga, e non si può rileggere.
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
