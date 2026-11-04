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
