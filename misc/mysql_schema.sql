--
-- Struttura della tabella `chat_feedback`
--

CREATE TABLE `chat_feedback` (
  `id` int NOT NULL PRIMARY KEY,
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
-- Struttura della tabella `failed_queries`
--

CREATE TABLE `failed_queries` (
  `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `standalone_query` text NOT NULL,
  `failure_type` varchar(50) NOT NULL,
  `topic_id_routed` varchar(255) DEFAULT NULL
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
  `chunk_size` int DEFAULT NULL,     
  `chunk_overlap` int DEFAULT 50,
  `parent_chunk_size` int DEFAULT 1500,
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

