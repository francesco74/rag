-- =============================================================================
-- Verifiche del database prima e dopo la migrazione dalla v3 alla v4
-- =============================================================================
--
-- Solo lettura: si può lanciare quante volte si vuole.
--
--   kubectl -n rag exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' \
--       < misc/migrations/v3_to_v4_verifica.sql
--
-- Procedura completa: misc/migrations/MIGRAZIONE_V3_V4.md.
-- =============================================================================


-- 1. Pulizia dei metadati dei parent (parent_metadata_cleanup.sql) — già fatta?
--    Nessuna riga = già fatta, o mai servita: non c'è niente da fare.
--    Righe presenti = eseguire misc/migrations/parent_metadata_cleanup.sql,
--    un passo alla volta, prima di align_dates. Senza la pulizia, correggendo
--    il testo in revisione la re-indicizzazione di questi documenti si blocca
--    ("Chiavi riservate trovate nei metadati extra").
SELECT topic_id, sub_topic_id, COUNT(*) AS parent_da_pulire
FROM parent_documents
WHERE JSON_CONTAINS_PATH(metadata, 'one',
        '$.content', '$.source', '$.topic_id', '$.sub_topic_id', '$.parent_id',
        '$.parent_index', '$.child_index', '$.file_name',
        '$._ingestion_error', '$._ingestion_id', '$.content_hash')
GROUP BY topic_id, sub_topic_id;


-- 2. Archivi esistenti: ogni sito della chat (utils/manage_sites.py add --topic)
--    deve usare uno di questi topic_id.
SELECT topic_id, description, public_access FROM topics ORDER BY topic_id;


-- 3. Ruoli della revisione. Prima della migrazione compaiono quelli della v3
--    (lettore, correttore, catalogatore, validatore, revisore, admin); dopo,
--    solo revisore, archivista e admin.
SELECT role, COUNT(*) AS utenti FROM review_user_roles GROUP BY role ORDER BY role;


-- 4. Tabelle della v4: dopo la migrazione devono esserci tutte e 7.
SELECT table_name AS tabelle_v4
FROM information_schema.tables
WHERE table_schema = DATABASE()
  AND table_name IN ('chat_sites', 'document_uploads', 'document_ocr', 'external_api_keys',
                     'external_api_key_topics', 'review_login_failures', 'schema_migrations')
ORDER BY table_name;


-- 5 e 6 leggono tabelle della v4: prima della migrazione il client si ferma qui con
-- "Table ... doesn't exist", ed è normale.

-- 5. Conversione dei ruoli già eseguita? (una riga = sì)
SELECT name, executed_at FROM schema_migrations WHERE name = 'v4_ruoli_revisione';


-- 6. Siti della chat configurati (dopo il passo 7 della procedura).
SELECT hostname, topic_id, name, allow_date_filter, allow_subtopic_selection, active
FROM chat_sites ORDER BY hostname;
