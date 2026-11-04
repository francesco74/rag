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
    -- Ogni archivio ha il suo sito.
    CONSTRAINT uq_chat_sites_topic UNIQUE (topic_id),
    CONSTRAINT fk_chat_sites_topic FOREIGN KEY (topic_id) REFERENCES topics (topic_id) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
