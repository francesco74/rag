    --
    -- Struttura della tabella `chat_feedback`
    --

    CREATE TABLE `chat_feedback` (
      `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
      `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
      `topic_id` varchar(255) DEFAULT NULL,
      `user_query` text,
      `ai_response` text,
      `rating` int DEFAULT NULL,
      `chat_history` json DEFAULT NULL,
      `comment` text DEFAULT NULL,
      `reviewed` tinyint(1) DEFAULT '0'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

    -- --------------------------------------------------------

    --
    -- Struttura della tabella `topics`
    --

    CREATE TABLE `topics` (
      `topic_id` varchar(100) NOT NULL PRIMARY KEY,
      `description` text NOT NULL,
      `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
      `aliases` text,
      `prompt` varchar(100) DEFAULT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

    CREATE TABLE IF NOT EXISTS `sub_topics` (
      `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
      `topic_id` varchar(100) NOT NULL,
      `sub_topic_id` varchar(100) NOT NULL,
      `description` varchar(255) NOT NULL,
      `description_long` varchar(255) NOT NULL,
      `chunk_size` int DEFAULT NULL,     
      `chunk_overlap` int DEFAULT 50,
      `parent_chunk_size` int DEFAULT 1500,
      `use_markdown_splitter` TINYINT(1) DEFAULT 1,
      UNIQUE KEY `unique_sub_topic` (`topic_id`, `sub_topic_id`),
      CONSTRAINT `fk_sub_topics_topic_id` 
        FOREIGN KEY (`topic_id`) 
        REFERENCES `topics` (`topic_id`) 
        ON DELETE CASCADE 
        ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

    CREATE TABLE IF NOT EXISTS system_logs (
        log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        log_level VARCHAR(15) NOT NULL,       -- e.g., 'ERROR', 'CRITICAL', 'WARNING'
        message TEXT NOT NULL,                -- The actual log message (TEXT to hold stack traces)
        file_name VARCHAR(255),               -- The python script name
        line_no INT,                          -- The line number where the error occurred
        pod_name VARCHAR(255) DEFAULT NULL,   -- The Kubernetes Pod name
        
        -- Indexes for faster querying when your log table grows
        INDEX idx_log_level (log_level),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

    CREATE TABLE IF NOT EXISTS parent_documents (
        id VARCHAR(36) PRIMARY KEY,
        topic_id VARCHAR(255) NOT NULL,
        sub_topic_id VARCHAR(255) NOT NULL,
        source VARCHAR(255) NOT NULL,
        file_name VARCHAR(255),
        parent_index INT,
        content LONGTEXT NOT NULL,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_search (source, topic_id, sub_topic_id)
    );

    -- Tabella metriche per il RAG: una riga per ogni task processato da process_rag_query.
-- Segue lo stesso pattern di chat_feedback già presente nel DB.

  CREATE TABLE IF NOT EXISTS rag_metrics (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,

      task_id VARCHAR(64) NOT NULL,
      topic_id VARCHAR(128),
      query_preview VARCHAR(255),

      cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
      is_satisfactory BOOLEAN,

      total_attempts TINYINT NOT NULL DEFAULT 0,
      duration_ms INT,

      -- Dettaglio per-tentativo: lista di oggetti con attempt, standalone_query,
      -- semantic_size, semantic_threshold, mmr_lambda, n_parent_candidates,
      -- n_mmr_candidates, n_mmr_penalized, n_parents_selected, n_parents_final.
      -- Vuoto ([]) per i cache hit, dato che non passano dal retrieval.
      attempts_detail JSON,

      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

      INDEX idx_task_id (task_id),
      INDEX idx_topic_created (topic_id, created_at),
      INDEX idx_created_at (created_at)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--  ==============================================================================
-- QUERY DI ESEMPIO PER L'ANALISI
-- ==============================================================================
 
-- 1. Quante query su 100 arrivano a ciascun numero di tentativi (cache hit esclusi)
-- SELECT total_attempts, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
-- FROM rag_metrics
-- WHERE cache_hit = FALSE AND created_at > NOW() - INTERVAL 7 DAY
-- GROUP BY total_attempts;
 
-- 2. Tasso di cache hit nel tempo
-- SELECT DATE(created_at) AS day, AVG(cache_hit) AS cache_hit_rate, COUNT(*) AS n
-- FROM rag_metrics
-- GROUP BY DATE(created_at)
-- ORDER BY day DESC;
 
-- 3. Quante volte MMR sceglie un parent "penalizzato" (segnale di redistribuzione
--    che sta comunque scegliendo qualcosa di simile a ciò che ha già in quota)
-- SELECT
--   JSON_EXTRACT(attempt, '$.attempt') AS attempt_n,
--   AVG(JSON_EXTRACT(attempt, '$.n_mmr_penalized') / NULLIF(JSON_EXTRACT(attempt, '$.n_mmr_candidates'), 0)) AS penalized_ratio
-- FROM rag_metrics, JSON_TABLE(attempts_detail, '$[*]' COLUMNS (attempt JSON PATH '$')) AS jt
-- GROUP BY attempt_n;
 
-- 4. Query non soddisfacenti anche dopo tutti i retry (candidate per audit manuale)
-- SELECT task_id, topic_id, query_preview, total_attempts, duration_ms, created_at
-- FROM rag_metrics
-- WHERE is_satisfactory = FALSE AND cache_hit = FALSE
-- ORDER BY created_at DESC
-- LIMIT 50;
 
 
 
-- 5. Motivi di fallimento piu' frequenti (richiede il verdetto strutturato)
-- SELECT JSON_UNQUOTE(JSON_EXTRACT(attempt, '$.grader_reason')) AS reason, COUNT(*) AS n
-- FROM rag_metrics, JSON_TABLE(attempts_detail, '$[*]' COLUMNS (attempt JSON PATH '$')) AS jt
-- WHERE cache_hit = FALSE AND created_at > NOW() - INTERVAL 30 DAY
-- GROUP BY reason ORDER BY n DESC;
 
-- 6. Entita' cronicamente scoperte: se una compare spesso, verificare se i suoi
--    documenti sono stati indicizzati (problema di ingestion) oppure se ci sono
--    ma il retrieval non li aggancia (problema di soglie).
-- SELECT target, COUNT(*) AS n
-- FROM rag_metrics,
--      JSON_TABLE(attempts_detail, '$[*]' COLUMNS (missing JSON PATH '$.missing_targets')) AS jt,
--      JSON_TABLE(jt.missing, '$[*]' COLUMNS (target VARCHAR(255) PATH '$')) AS t
-- WHERE created_at > NOW() - INTERVAL 30 DAY
-- GROUP BY target ORDER BY n DESC LIMIT 20;
 
