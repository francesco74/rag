-- Pagine trascritte con l'OCR per ogni documento: le aggiorna il converter.
-- I documenti indicizzati prima di questa tabella non hanno riga: dato non noto.
CREATE TABLE IF NOT EXISTS document_ocr (
    source        VARCHAR(700) NOT NULL,
    topic_id      VARCHAR(100) NOT NULL,
    sub_topic_id  VARCHAR(100) NOT NULL,
    ocr_pages     INT          NOT NULL,
    total_pages   INT          NOT NULL,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_ocr UNIQUE (source(500), topic_id, sub_topic_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
