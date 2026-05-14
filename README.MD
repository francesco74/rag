# 💡 Il Tuo Assistente AI: Come Funziona e Perché è Intelligente

Il sistema funziona come un **Assistente Personale altamente qualificato** in grado di rispondere a qualsiasi domanda, ma con una regola fondamentale: consulta solo ed esclusivamente i tuoi documenti e archivi interni. Non "inventa" le risposte, ma le **trova, le riassume e le cita** direttamente dalle fonti fornite.

Il progetto si compone di tre parti che lavorano insieme come una squadra: l'**Impiegato Addetto all'Archivio (Ingestion)**, il **Cervello del Ricercatore (Backend)** e l'**Interfaccia di Chat (Frontend)**.

## 1. 📂 L'Impiegato Addetto all'Archivio (Sistema di Ingestion)
Questa parte del sistema prende i tuoi documenti e li rende "cercabili" dall'intelligenza artificiale.
*   **Monitoraggio:** Controlla continuamente una cartella dedicata per rilevare l'inserimento di nuovi manuali o file.
*   **Comprensione (OCR):** Utilizza la tecnologia OCR di Google per "leggere" ed estrarre il testo anche da file scannerizzati o immagini.
*   **Chunking e Contesto Esteso (Parent Document Retrieval):** Il documento intero è troppo grande, quindi l'Impiegato lo divide in tanti piccoli paragrafi pertinenti (chiamati *chunks*) per trovarli con estrema precisione. Tuttavia, il sistema sa che una singola frase estrapolata può perdere significato. Per questo, collega ogni piccolo frammento al suo "documento padre" originale. 
*   **Vettorizzazione e Archiviazione:** Traduce i pezzetti di testo in "codici segreti" numerici (vettori) basati sul significato, e li archivia in un database super-veloce chiamato **Qdrant**, catalogandoli per argomento.

## 2. 🧠 Il Cervello del Ricercatore (Backend API)
È il motore intelligente del sistema che decide come rispondere alle domande.
*   **Analisi del Contesto:** Riscrive le domande per renderle chiare in base allo storico della conversazione.
*   **Cache Semantica:** Controlla se hai già posto in passato una domanda simile, restituendo subito la risposta per risparmiare tempo.
*   **Ricerca Ibrida, Re-ranking e Recupero del "Documento Padre":** Identifica l'argomento e cerca nei documenti sia per significato sia per parole chiave esatte. Una volta trovati i piccoli frammenti più rilevanti, grazie alla tecnica del **Parent Document Retrieval**, il sistema non si limita a usare solo quei frammenti isolati, ma **recupera l'intero documento originale (o sezioni molto più ampie) e lo passa all'Intelligenza Artificiale**.
*   **Generazione della Risposta:** Invia questo contesto completo e arricchito al modello **Gemini**, istruendolo a rispondere *solo* in base a tali informazioni e fornendo le fonti esatte.

## 3. 💬 L'Interfaccia di Chat (Frontend Flutter)
L'applicazione web progettata per rendere l'interazione semplice e trasparente.
*   **Chat Intelligente:** Mantiene traccia di tutte le domande che hai fatto per comprendere il contesto.
*   **Chiarezza e Fonti:** Espone dei "Chip" cliccabili alla fine della risposta con i nomi dei file originali. Cliccandoli, puoi aprire direttamente i documenti di origine per verificare le informazioni.
*   **Feedback:** Include pulsanti 👍/👎 per valutare se la risposta è stata ottima o sbagliata, aiutando il sistema a migliorarsi continuamente.
