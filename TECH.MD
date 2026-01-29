# 📄 Documentazione Tecnica: Gemini RAG System

**Versione:** 1.0
**Autore:** AI Assistant (Basato su Specifiche Progetto)
**Descrizione:** Architettura RAG (Retrieval-Augmented Generation) avanzata basata su Google Gemini, Qdrant (Vector DB) e un'interfaccia utente Flutter.

---

## 1. Architettura del Sistema (Overview)

Il progetto è implementato come un'architettura a microservizi disaccoppiati, incentrata sul paradigma **RAG** per fornire risposte basate su una knowledge base proprietaria.

### 1.1 Diagramma dei Componenti

| Modulo Applicativo | Funzione | Tecnologia Chiave | Modello AI |
| :--- | :--- | :--- | :--- |
| **Ingestion Worker** | Estrazione, Chunking, Vettorizzazione. | Python (Asincrono) | Gemini 2.5 Flash (OCR), Gemini Embedding 001 |
| **RAG API (Backend)** | Routing, Retrieval, Re-ranking, Generazione, Caching. | Python (Flask) | Gemini 2.5 Flash (Router, Transformer, Generator) |
| **Frontend Client** | Interfaccia Utente, Stato Chat, Feedback. | Flutter (Web) | N/A (Client) |

| Servizio Dati | Tipo | Tecnologia | Ruolo |
| :--- | :--- | :--- | :--- |
| **Vector Database** | Non-Relazionale | Qdrant | Archiviazione Chunks (Embedding) e Cache Semantica. |
| **Relational Database** | Relazionale | MySQL | Metadati (Topics, Aliases), Log (Failed Queries, Feedback). |
| **Storage (Files)** | File System | Dischi / Volume Montato | Archiviazione Documenti Sorgente (Watch/Processed/Error). |

---

## 2. RAG API (Backend) - `app.py`

Il backend espone l'API di chat principale e di feedback.

### 2.1 Pipeline di Risposta

Il processo di generazione della risposta segue questa sequenza:

1.  **Contextualization:** La query viene trasformata in una **Standalone Query** (`gemini-2.5-flash`).
2.  **Semantic Cache Check:** La `query_vector` viene controllata nella collezione Qdrant `semantic_cache`. (Threshold: $\ge 0.95$).
3.  **Topic Routing:** La query viene associata a un `topic_id` (ottenuto da MySQL e processato da `gemini-2.5-flash`).
4.  **Retrieval Ibrido:** Ricerca vettoriale e keyword su Qdrant, filtrate per `topic_id`.
5.  **Re-ranking:** I **20** candidati recuperati vengono riordinati dal **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`) per selezionare i top **15** *chunks* più pertinenti.
6.  **Generation:** Il contesto aggregato (max **30.000 caratteri**) viene inviato al `GENERATOR_MODEL` (`gemini-2.5-flash`).

### 2.2 Endpoint API

| Metodo | Endpoint | Descrizione | Request Body (JSON) | Response Body (JSON) |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/chat` | Gestisce l'intera pipeline RAG e restituisce la risposta AI. | `{"query": str, "history": List[dict]}` | `{"answer": str, "sources": List[dict], "topic": str, "cached": bool}` |
| `POST` | `/feedback` | Registra il feedback utente per HIL (Human-in-the-Loop). | `{"query": str, "answer": str, "topic_id": str, "rating": int, "history": List[dict]}` | `{"status": "success", "message": str}` |

---

## 3. Ingestion Worker - `ingestion_worker.py`

Il Worker è un servizio di background asincrono per l'aggiornamento della Knowledge Base.

### 3.1 Flusso di Elaborazione Asincrona

* **Concurrency Control:** `asyncio.Semaphore(5)` e `AsyncLimiter` (per le chiamate Gemini) gestiscono il carico.
* **Parsing Ibrido:** Estrazione testo digitale e **OCR asincrono** (`async_ocr_generate`) con **Gemini 2.5 Flash** per PDF/immagini scansionate (attivato se lunghezza testo $<50$ e varianza immagine $>15$).
* **Chunking:** Utilizzo di `RecursiveCharacterTextSplitter` (encoding `cl100k_base`):
    * **Dimensione Chunk:** 750 caratteri.
    * **Overlap:** 150 caratteri.
* **Vettorizzazione:** I *chunks* sono vettorizzati in batch (size 50) tramite `async_embed_batch` (`gemini-embedding-001`).
* **Upsert Qdrant:** Caricamento in batch, con ID deterministico (`uuid.uuid5`) basato sui metadati.
* **Gestione File:** Spostamento automatico in cartelle `PROCESSED_FOLDER` o `ERROR_FOLDER`.

---

## 4. Frontend (Flutter Web App) - `main.dart`

Il client gestisce lo stato della sessione, la visualizzazione e la raccolta di metriche.

### 4.1 Caratteristiche del Client

* **Gestione del Contesto:** La cronologia della chat viene filtrata e inviata al backend ad ogni richiesta per la contextualization.
* **Rendering Risposte:** `flutter_html` supporta il rendering HTML per risposte AI riccamente formattate.
* **Trasparenza:** Le fonti (file) sono visualizzate come `Chip` cliccabili. Il frontend costruisce l'URL completo per consentire all'utente l'accesso diretto al documento sorgente.
* **Raccolta Dati:** I pulsanti di feedback (Like/Dislike) raccolgono il rating, la query, la risposta e la cronologia contestuale e li inviano all'endpoint `/feedback`.

---

## 5. Setup dell'Ambiente

### 5.1 Dipendenze

È necessario un ambiente Python 3.9+ e un Flutter SDK configurato. Le dipendenze Python sono gestite tramite `requirements.txt`.

### 5.2 Variabili d'Ambiente

Le credenziali devono essere fornite tramite un file `.env` (non committato).

```ini
# .env.example
# Credenziali API
GOOGLE_API_KEY="AIzaSy..."

# Configurazione Database MySQL
DB_HOST="localhost"
DB_USER="rag_user"
DB_PASS="secure_pass"
DB_NAME="rag_system"

# Configurazione Vector Database Qdrant
QDRANT_HOST="localhost"
QDRANT_PORT="6333"

# Configurazione Ingestion
WATCH_FOLDER="./watch_folders/"
LOG_LEVEL="INFO"
