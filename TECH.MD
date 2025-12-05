📄 Documentazione Tecnica del Progetto RAG Avanzato
Titolo: Gemini RAG System - Technical Documentation
Versione: 1.0
Autore: AI Assistant (Basato su Specifiche Progetto)
1. Architettura del Sistema (Overview)
Il progetto è implementato come un'architettura a microservizi disaccoppiati, incentrata sul paradigma RAG (Retrieval-Augmented Generation) per fornire risposte basate su una knowledge base proprietaria.
1.1 Diagramma dei Componenti
Il sistema si compone di tre moduli applicativi principali e tre servizi di data storage:

| Modulo Applicativo | Funzione | Linguaggio/Framework | Modello AI |
|---|---|---|---|
| Ingestion Worker | Estrazione, Chunking, Vettorizzazione. | Python (Asincrono) | Gemini 2.5 Flash (OCR), Gemini Embedding 001 |
| RAG API (Backend) | Routing, Retrieval, Re-ranking, Generazione, Caching. | Python (Flask) | Gemini 2.5 Flash (Router, Transformer, Generator) |
| Frontend Client | Interfaccia Utente, Stato Chat, Feedback. | Flutter (Web) | N/A (Client) |
| Servizio Dati | Tipo | Tecnologia | Ruolo |
|---|---|---|---|
| Vector Database | Non-Relazionale | Qdrant | Archiviazione Chunks (Embedding) e Cache Semantica. |
| Relational Database | Relazionale | MySQL | Metadati (Topics, Aliases), Log (Failed Queries, Feedback). |
| Storage (Files) | File System | Dischi / Volume Montato | Archiviazione Documenti Sorgente (Watch/Processed/Error). |
2. RAG API (Backend) - app.py
Il backend espone l'API di chat principale e di feedback.
2.1 Pipeline di Risposta
Il processo di generazione della risposta è il seguente:
 * Contextualization: transform_query(history, query) usa gemini-2.5-flash per trasformare la query in una Standalone Query basata sulla cronologia.
 * Semantic Cache Check: La query_vector viene confrontata con la collezione Qdrant semantic_cache. Threshold: \ge 0.95.
 * Topic Routing: route_query_to_topic() usa gemini-2.5-flash per associare la query a un topic_id da MySQL.
 * Retrieval Ibrido: retrieve_chunks() esegue la ricerca vettoriale e la ricerca keyword (MatchText) sulla collezione document_chunks, filtrando per topic_id.
 * Re-ranking: I candidati vengono riordinati dal Cross-Encoder (BAAI/bge-reranker-v2-m3) per selezionare i top 15 chunks più rilevanti.
 * Generation: Il contesto aggregato e limitato a 30.000 caratteri viene inviato a GENERATOR_MODEL (gemini-2.5-flash) con un prompt specifico per il topic.
2.2 Endpoint API
| Metodo | Endpoint | Descrizione | Request Body (JSON) | Response Body (JSON) |
|---|---|---|---|---|
| POST | /chat | Gestisce l'intera pipeline RAG e restituisce la risposta AI. | {"query": str, "history": List[dict]} | {"answer": str, "sources": List[dict], "topic": str, "cached": bool} |
| POST | /feedback | Registra il feedback utente nel DB per HIL (Human-in-the-Loop). | {"query": str, "answer": str, "topic_id": str, "rating": int, "history": List[dict]} | {"status": "success", ...} |
2.3 Modelli e Configurazioni
 * Embedding: gemini-embedding-001 (Output: 768 dim).
 * LLMs: gemini-2.5-flash (Utilizzato per Transform, Route, Generate).
 * Re-ranker: BAAI/bge-reranker-v2-m3 (Multilingue, max_length=1024).
 * Parametri: QDRANT_SIZE=20, RERANK_SIZE=15, MAX_CONTEXT_CHARS=30000.
3. Ingestion Worker - ingestion_worker.py
Il Worker è un servizio di background asincrono per l'aggiornamento della Knowledge Base.
3.1 Flusso di Elaborazione Asincrona
Il Worker utilizza asyncio, aiolimiter e tenacity per l'elaborazione resiliente e parallela dei file.
 * Stabilità File: wait_for_file_stability() attende che il file sia completamente copiato prima dell'inizio dell'elaborazione.
 * Parsing Ibrido:
   * Per i PDF, extract_text_from_pdf_sync tenta l'estrazione del testo digitale.
   * Se il testo digitale è scarso (lunghezza <50 e varianza immagine >15), viene eseguito l'OCR asincrono (async_ocr_generate) tramite Gemini 2.5 Flash.
 * Chunking: I dati estratti vengono segmentati in chunks di 750 caratteri con 150 caratteri di overlap (Encoding: cl100k_base).
 * Vettorizzazione: I chunks vengono vettorizzati in batch (size 50) tramite async_embed_batch.
 * Upsert: I PointStruct sono caricati in Qdrant, con uuid.uuid5 per ID deterministici basati su topic, filename e indice del chunk.
 * Sincronizzazione DB: register_topic_safe assicura la presenza del topic_id in MySQL.
3.2 Struttura delle Directory
 * WATCH_FOLDER/<topic_id>/: Directory di input monitorata.
 * PROCESSED_FOLDER/<topic_id>/: Destinazione in caso di successo.
 * ERROR_FOLDER/<topic_id>/: Destinazione in caso di fallimento.
4. Frontend (Flutter Web App) - main.dart
Il client è un'applicazione Flutter che gestisce lo stato della sessione e l'interazione con l'utente.
4.1 Gestione dello Stato e Comunicazione API
 * Modello Dati: ChatMessage memorizza testo, sorgenti, topic, stato di errore e stato di feedback (FeedbackStatus).
 * Cronologia: La funzione _handleSendPressed estrae e filtra la cronologia (escludendo messaggi di sistema/errore) per inviarla all'endpoint /chat.
 * Visualizzazione HTML: Utilizza la libreria flutter_html per il rendering sicuro delle risposte formattate dall'LLM.
4.2 Interazione Utente e Trasparenza
 * Visualizzazione Fonti: Le sorgenti (file) sono presentate come Chip cliccabili.
   * Costruzione URL: Il link è costruito con la logica http://<base_url>/downloads/$topic/$fileName, richiedendo che i documenti siano serviti da un server web esterno (es. Nginx o altro CDN) per l'accesso diretto.
 * Raccolta Feedback: _sendFeedback invia la coppia (query, risposta) con la cronologia e il rating (1/-1) all'endpoint /feedback. Il feedback è essenziale per la metrica di prestazione del sistema.
4.3 Dipendenze Chiave (pubspec.yaml)
 * http: Comunicazione con l'API Flask.
 * flutter_html: Rendering HTML della risposta AI.
 * url_launcher: Apertura dei link ai documenti sorgente.
5. Setup dell'Ambiente e Dipendenze
Per avviare il progetto, sono necessari i seguenti servizi:
 * Python Environment: Python 3.9+ e le librerie elencate in requirements.txt (Flask, requests, google-genai, qdrant-client, mysql-connector, sentence-transformers, PyMuPDF, PIL, tenacity, aiolimiter).
 * Database Qdrant: Istanza running.
 * Database MySQL: Istanza running con lo schema del database pre-configurato (tabelle topics, failed_queries, chat_feedback).
 * Flutter SDK: Per buildare e servire l'applicazione web.
 * Variabili d'Ambiente: Configurazione del file .env con le chiavi API di Google, credenziali DB e indirizzi host (Qdrant/MySQL).
<!-- end list -->
# .env file example
GOOGLE_API_KEY="AIzaSy..."
DB_HOST="localhost"
DB_USER="rag_user"
DB_PASS="secure_pass"
DB_NAME="rag_system"
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
WATCH_FOLDER="./watch_folders/"
LOG_LEVEL="INFO"

