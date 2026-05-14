# ⚙️ Architettura Tecnica - Progetto RAG

Questo repository implementa una soluzione di **Retrieval-Augmented Generation (RAG)** basata su un'architettura a microservizi. Il progetto è sviluppato principalmente in **Python (62.2%)** per il backend e **Dart (33.2%)** per il frontend Flutter, con una minima parte (2.5%) di HTML. 

Il codice è organizzato in moduli distinti:

## 1. Modulo `ingestion`: Pipeline dei Dati
Questo modulo gestisce l'acquisizione e l'indicizzazione dei documenti all'interno del sistema.
*   **Lettura Dati e OCR:** Integra i servizi Google OCR per effettuare il parsing di documenti complessi (come PDF scannerizzati e immagini), convertendoli in testo elaborabile.
*   **Chunking gerarchico:** I documenti estratti vengono frammentati in contesti semantici mirati (*child chunks*). Questi piccoli chunk sono ottimizzati per la ricerca vettoriale, ma mantengono una referenza diretta e un mapping univoco al documento originale più ampio (*Parent Document*).
*   **Vector Database:** L'archiviazione dei vettori e dei relativi metadati avviene su **Qdrant**, un engine vettoriale ad alte prestazioni.

## 2. Modulo `rest`: Backend e Core Logico
Costituisce le API (sviluppate in Python) che gestiscono l'interfaccia tra l'utente e i modelli di Intelligenza Artificiale.
*   **Query Transformation:** Per supportare le chat multi-turn, il sistema effettua un *Contextual Query Rewriting* che trasforma e disambigua le domande in base al contesto precedente.
*   **Semantic Cache:** Un layer di caching vettoriale che intercetta domande con similarità elevata a query passate.
*   **Retrieval, Reranking e Parent Document Retrieval:** Esegue una **Ricerca Ibrida** (vettoriale e keyword-based). Qui viene applicato il design pattern del **Parent Document Retrieval**: la ricerca di similarità viene eseguita sui chunk piccoli per garantire la massima precisione semantica (alta *accuracy* di matching). Una volta identificati i top-K chunks, il sistema fa un lookup e recupera i rispettivi *Parent Documents* completi. 
*   **LLM Generation:** I documenti "padre" espansi vengono forniti a **Google Gemini** con un prompt di grounding. Questo assicura che il modello abbia una visione d'insieme completa, superando i classici limiti di perdita di contesto dei sistemi RAG basati su chunking tradizionale.

## 3. Modulo `web`: Frontend in Flutter
Il client è un'applicazione web sviluppata in Dart tramite il framework Flutter.
*   **Gestione dello Stato e UI:** Gestisce in tempo reale lo storico delle chat e il rendering delle risposte.
*   **Reference Linking:** Estrae e mappa i metadati forniti dal backend trasformandoli in elementi UI interattivi (*Chip*). Questo permette all'utente di accedere in un clic al file sorgente originale salvato nel database.
*   **Modulo di Feedback:** Integra chiamate REST per raccogliere metriche utente tramite pulsanti *upvote/downvote* (👍/👎), pensate per future pipeline di fine-tuning.