-- =============================================================================
-- Pulizia dei metadati dei parent: rimozione delle chiavi di sistema
-- =============================================================================
--
-- Problema: i parent migrati da Qdrant con utils/migrate/parent_documents_qdrant2mysql.py
-- hanno in parent_documents.metadata l'intero payload del punto Qdrant, comprese
-- le chiavi di sistema (content, source, topic_id, sub_topic_id, parent_index...).
-- Il testo è quindi salvato due volte e, soprattutto, la revisione le tratta
-- come metadati del documento: correggendo il testo vengono ricopiate nel
-- manifest e l'ingest blocca la re-indicizzazione ("Chiavi riservate trovate
-- nei metadati extra").
--
-- Le chiavi rimosse sono settings.protected_keys (common/config.py). I parent
-- creati dall'ingest non le hanno mai nei metadati: per loro la pulizia non
-- cambia nulla. Il payload su Qdrant NON si tocca: lì quelle chiavi servono.
--
-- Eseguire i passi uno alla volta, controllando il risultato di ciascuno.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- PASSO 1 — Solo lettura: quanti parent sono coinvolti, e dove
-- -----------------------------------------------------------------------------
SELECT topic_id, sub_topic_id, COUNT(*) AS parent_coinvolti
FROM parent_documents
WHERE JSON_CONTAINS_PATH(metadata, 'one',
        '$.content', '$.source', '$.topic_id', '$.sub_topic_id', '$.parent_id',
        '$.parent_index', '$.child_index', '$.file_name',
        '$._ingestion_error', '$._ingestion_id', '$.content_hash')
GROUP BY topic_id, sub_topic_id;

-- Facoltativo: quanto spazio occupano oggi i metadati coinvolti
SELECT COUNT(*) AS parent, ROUND(SUM(LENGTH(metadata)) / 1024 / 1024, 1) AS mb_metadati
FROM parent_documents
WHERE JSON_CONTAINS_PATH(metadata, 'one', '$.content', '$.source', '$.topic_id');


-- -----------------------------------------------------------------------------
-- PASSO 2 — Copia di sicurezza dei metadati che verranno modificati
-- (per tornare indietro: vedi in fondo)
-- -----------------------------------------------------------------------------
CREATE TABLE parent_documents_metadata_backup_20261008 AS
SELECT id, metadata
FROM parent_documents
WHERE JSON_CONTAINS_PATH(metadata, 'one',
        '$.content', '$.source', '$.topic_id', '$.sub_topic_id', '$.parent_id',
        '$.parent_index', '$.child_index', '$.file_name',
        '$._ingestion_error', '$._ingestion_id', '$.content_hash');

-- Deve coincidere con il totale del passo 1
SELECT COUNT(*) AS righe_salvate FROM parent_documents_metadata_backup_20261008;


-- -----------------------------------------------------------------------------
-- PASSO 3 — Pulizia (in transazione: verificare prima di COMMIT)
-- -----------------------------------------------------------------------------
START TRANSACTION;

UPDATE parent_documents
SET metadata = JSON_REMOVE(metadata,
        '$.content', '$.source', '$.topic_id', '$.sub_topic_id', '$.parent_id',
        '$.parent_index', '$.child_index', '$.file_name',
        '$._ingestion_error', '$._ingestion_id', '$.content_hash')
WHERE id IN (SELECT id FROM parent_documents_metadata_backup_20261008);

-- Verifica: deve restituire 0
SELECT COUNT(*) AS ancora_con_chiavi_di_sistema
FROM parent_documents
WHERE JSON_CONTAINS_PATH(metadata, 'one',
        '$.content', '$.source', '$.topic_id', '$.sub_topic_id', '$.parent_id',
        '$.parent_index', '$.child_index', '$.file_name',
        '$._ingestion_error', '$._ingestion_id', '$.content_hash');

-- Controllo a campione: restano solo i metadati veri (es. "Header 1")
SELECT id, metadata FROM parent_documents
WHERE id IN (SELECT id FROM parent_documents_metadata_backup_20261008)
LIMIT 5;

-- Se tutto è corretto:
COMMIT;
-- altrimenti: ROLLBACK;


-- -----------------------------------------------------------------------------
-- Ritorno allo stato precedente (solo se servisse, dopo il COMMIT)
-- -----------------------------------------------------------------------------
-- UPDATE parent_documents p
-- JOIN parent_documents_metadata_backup_20261008 b ON b.id = p.id
-- SET p.metadata = b.metadata;

-- Quando non serve più la copia di sicurezza:
-- DROP TABLE parent_documents_metadata_backup_20261008;
