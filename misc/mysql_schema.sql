--
-- Struttura della tabella `chat_feedback`
--

CREATE TABLE `chat_feedback` (
  `id` int NOT NULL,
  `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `topic_id` varchar(255) DEFAULT NULL,
  `user_query` text,
  `ai_response` text,
  `rating` int DEFAULT NULL,
  `chat_history` json DEFAULT NULL,
  `reviewed` tinyint(1) DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Struttura della tabella `failed_queries`
--

CREATE TABLE `failed_queries` (
  `id` int NOT NULL,
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
  `topic_id` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `aliases` text,
  `prompt` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Indici per le tabelle scaricate
--

--
-- Indici per le tabelle `failed_queries`
--
ALTER TABLE `failed_queries`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `topics`
--
ALTER TABLE `topics`
  ADD PRIMARY KEY (`topic_id`);

--
-- AUTO_INCREMENT per le tabelle scaricate
--

--
-- AUTO_INCREMENT per la tabella `failed_queries`
--
ALTER TABLE `failed_queries`
  MODIFY `id` int NOT NULL AUTO_INCREMENT;
COMMIT;
