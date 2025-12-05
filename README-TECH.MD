# 📚 Documentazione del Sistema RAG Avanzato

## 🎯 1. Panoramica del Sistema

Il sistema descritto è una soluzione di **Chatbot RAG (Retrieval-Augmented Generation)**, progettata per rispondere a domande complesse attingendo a una base di conoscenza documentale proprietaria. Sfrutta i modelli linguistici avanzati di Google Gemini, la gestione asincrona dei dati e un'architettura microservizi per garantire risposte accurate, contestuali e aggiornate.

### Componenti Chiave

| Componente | Tecnologia | Ruolo Principale |
| :--- | :--- | :--- |
| **Backend API** | Python (Flask), Go
Questo schema copre l'intero ciclo di vita dei dati: dal caricamento automatico (Ingestion), alla ricerca complessa e alla generazione della risposta (Backend RAG), fino all'interazione utente e alla valutazione della qualità (Frontend).ogle Gemini, Qdrant, MySQL | Gestione del flusso di chat, Reranking, Caching Semantico e Generazione della Risposta. |
| **Sistema di Ingestion** | Python (Asincrono), Gemini, Qdrant, MySQL | Estrazione, Chunking, OCR (per immagini/PDF scansionati) e Vettorializzazione dei documenti. |
| **Frontend** | Flutter (Web App) | Interfaccia Utente, gestione della Cronologia Conversazionale, Visualizzazione delle Fonti e Raccolta del Feedback. |

---

## ⚙️ 2. Pipeline del Backend API (Servizio di Chat)

Il servizio Flask (Python) è il cervello del sistema, gestendo la complessità della risposta a ogni query utente (`/chat`).

### 2.1 Pre-Processing e Ottimizzazione della Query

1.  **Trasformazione della Query (Query Rewriting):**
    * **Obiettivo:** Convertire una query dipendente dal contesto (es. "E per quello?") in una query auto-contenuta (standalone) utilizzando il modello **Gemini 2.5 Flash** (`TRANSFORM_MODEL`).
    * **Vantaggio:** Migliora drasticamente la precisione della ricerca vettoriale.
2.  **Embedding:**
    * La query standalone viene convertita in un vettore a 768 dimensioni utilizzando **`gemini-embedding-001`**.
3.  **Caching Semantico (Qdrant - `semantic_cache`):**
    * Il sistema controlla se la query (vettorizzata) è stata posta e risolta in precedenza con un'alta similarità ($\ge 95\%$ ). In caso di **HIT**, la risposta viene restituita immediatamente, riducendo latenza e costi di API.
4.  **Routing del Topic:**
    * Un LLM separato (`ROUTER_MODEL`) assegna la query a un **`topic_id`** specifico (es. `policy_hr`) basato sulla configurazione in **MySQL**. Questo restringe il campo di ricerca.

### 2.2 Retrieval Avanzato (Ricerca Ibrida con Re-ranking)

1.  **Ricerca Ibrida Fusa (Qdrant - `document_chunks`):**
    * Viene eseguita una ricerca combinata che include:
        * **Vector Search:** Basata sull'embedding della query.
        * **Keyword Search:** Basata sul testo della query.
    * Entrambi i risultati sono **filtrati** dal `topic_id` e vengono fusi in un unico set di **20** candidati (`QDRANT_SIZE`).
2.  **Re-ranking (Cross-Encoder):**
    * I candidati vengono riordinati per pertinenza contestuale da un modello **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`).
    * **Vantaggio:** Un re-ranker valuta la coerenza logica della coppia (Query, Chunk) e non solo la vicinanza spaziale, selezionando i migliori **15** *chunks* (`RERANK_SIZE`).
3.  **Generazione del Contesto:** I *chunks* selezionati vengono concatenati, con l'aggiunta di **tag di fonte** (`[Source: filename]`), e viene applicato un limite di caratteri (**30.000**) per ottimizzare la chiamata all'LLM.

### 2.3 Generazione della Risposta e Feedback

1.  **Generazione RAG:** Il contesto, il prompt specifico per il *topic* e la query vengono inviati al modello **Gemini 2.5 Flash** (`GENERATOR_MODEL`) per generare la risposta finale.
2.  **Gestione degli Errori/Log:** Le query che falliscono (es. non trovano *topic* o risposta nel contesto) vengono loggate in MySQL per analisi e miglioramento futuri.
3.  **Raccolta Feedback (`/feedback`):** Un endpoint separato consente di registrare il feedback dell'utente (rating, query, risposta, cronologia) in MySQL, alimentando un ciclo di **Human-in-the-Loop (HIL)** per la valutazione della qualità.

---

## 🏗️ 3. Pipeline di Ingestion Asincrona

Il servizio di Ingestion è un processo asincrono progettato per caricare e preparare automaticamente i dati non strutturati.

### 3.1 Flusso di Lavoro Controllato

1.  **Monitoraggio:** Il sistema monitora la **`WATCH_FOLDER`**, dove le sottocartelle definiscono il **`topic_id`** dei documenti.
2.  **Controlli di Concorrenza:** L'uso di `asyncio.Semaphore` e `AsyncLimiter` garantisce l'efficienza e il rispetto dei limiti di API (60 chiamate Gemini al minuto).
3.  **Gestione del File:** I file processati vengono spostati in **`PROCESSED_FOLDER`** o **`ERROR_FOLDER`** in base al successo dell'operazione.

### 3.2 Estrazione e Preparazione Avanzata dei Dati

1.  **Estrazione Ibrida PDF:**
    * Viene tentata l'estrazione del testo digitale.
    * Se il testo digitale è insufficiente, viene attivato l'**OCR** utilizzando **Gemini 2.5 Flash** per scansionare la pagina come immagine, gestendo documenti scansionati o basati su immagini.
2.  **Pulizia del Testo:** Viene applicata una funzione di pulizia per correggere l'uso del trattino a capo e lo spazio bianco eccessivo.
3.  **Segmentazione (Chunking):**
    * Utilizza **`RecursiveCharacterTextSplitter`** con codificatore Tiktoken.
    * **Parametri:** `CHUNK_SIZE=750` e `CHUNK_OVERLAP=150`, scelti per massimizzare il contesto e minimizzare la perdita di significato.
4.  **Vettorizzazione e Upsert:** I *chunks* vengono vettorizzati in batch (lotti da 50) e caricati in Qdrant (lotti da 100), assicurando che i metadati (`topic_id`, `source_file`) siano inclusi nel payload.
5.  **Sincronizzazione MySQL:** Il `topic_id` viene registrato o aggiornato nella tabella `topics` di MySQL.

---

## 📱 4. Frontend (Flutter Web App)

L'interfaccia utente è la porta d'accesso per l'utente finale, concentrandosi sull'usabilità e la raccolta di dati di qualità.

### 4.1 Caratteristiche Interfaccia Utente

* **Design:** Implementato con **Material 3** e un tema scuro per un'esperienza moderna.
* **Cronologia Conversazionale:** Gestisce lo stato e invia la cronologia filtrata al backend per il *query rewriting*.
* **Rendering HTML:** Utilizza `flutter_html` per interpretare la formattazione avanzata (es. paragrafi, liste) contenuta nella risposta AI.
* **Selezionabilità:** Il testo utente e la risposta AI sono completamente selezionabili e copiabili.

### 4.2 Trasparenza e Feedback

* **Visualizzazione delle Fonti:** Mostra i file di origine (`source_file`) come *chip* cliccabili.
* **Accesso ai Documenti:** I link delle fonti sono costruiti dinamicamente e indirizzati a un server di download (es. **Nginx**), permettendo all'utente di accedere al documento originale.
* **Sistema di Feedback:** I pulsanti **Like/Dislike** consentono agli utenti di valutare la qualità della risposta. Questo feedback viene trasmesso all'endpoint `/feedback` e registrato, chiudendo il ciclo di *Human-in-the-Loop* (HIL).
