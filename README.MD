## 💡 Il Tuo Assistente AI: Come Funziona e Perché è Intelligente

Il sistema è come un **Assistente Personale altamente qualificato** che può rispondere a qualsiasi domanda, ma solo ed esclusivamente consultando i tuoi documenti e archivi interni. Non "inventa" le risposte; le **trova, riassume e cita** dalle fonti fornite.

Si compone di tre parti che lavorano insieme come una squadra: l'**Impiegato Addetto all'Archivio (Ingestion)**, il **Cervello del Ricercatore (Backend)** e l'**Interfaccia di Chat (Frontend)**.

***

## 1. 📂 L'Impiegato Addetto all'Archivio (Sistema di Ingestion)

Questa parte del sistema è responsabile di prendere i tuoi documenti e renderli **"cercabili"** dall'AI.

### Cosa fa:
1.  **Monitora le Scartoffie:** L'Impiegato controlla continuamente una cartella dedicata. Quando metti un nuovo manuale o un file (anche PDF scannerizzati o immagini), lui lo nota.
2.  **Legge e Capisce:** Apre ogni documento e lo analizza. Se il file è un'immagine o un PDF difficile, usa la tecnologia **OCR (Riconoscimento Ottico dei Caratteri)** di Google per "leggere" e trascrivere il testo, proprio come un umano farebbe con una scansione.
3.  **Fa a Pezzetti Intelligenti (Chunking):** Il documento intero è troppo grande. L'Impiegato lo divide in tanti piccoli paragrafi pertinenti (chiamati *chunks*), assicurandosi che non venga perso il significato a metà frase.
4.  **Codifica in Codici Segreti (Vettorizzazione):** Assegna a ogni pezzetto di testo un "codice numerico" (chiamato **vettore**) che ne rappresenta il significato. Invece di cercare per parole chiave esatte, l'AI cercherà in base al **significato** del tuo codice.
5.  **Archiviazione:** I codici segreti e i pezzetti di testo vengono salvati in un archivio super-veloce (**Qdrant**), etichettati per argomento.

***

## 2. 🧠 Il Cervello del Ricercatore (Backend API)

Quando fai una domanda in chat, il Ricercatore si attiva. Questo è il cuore intelligente che decide come rispondere.

### I 5 Passaggi per una Risposta:

1.  **Capire il Contesto:** Se hai chiesto "Cos'è la Politica X?" e subito dopo "E il modulo da usare?", il Ricercatore usa l'AI per **riscrivere** la seconda domanda come "Qual è il modulo da usare per la Politica X?", in modo da avere una ricerca chiara.
2.  **Controllo Memoria (Cache Semantica):** Prima di cercare, controlla: "Ho già risposto a una domanda **identica o molto simile** prima?". Se sì, la risposta viene data subito, risparmiando tempo e risorse.
3.  **Identificare l'Argomento:** Capisce subito se la domanda riguarda le "Risorse Umane" o l'"Informatica", restringendo la ricerca solo all'archivio pertinente.
4.  **Cercare in Modo Ibrido (Re-ranking):** Non cerca solo per "codice segreto" (significato), ma anche per **parole chiave esatte**. Poi, usa un modello AI specializzato per **riordinare** i risultati trovati, selezionando i più utili in assoluto.
5.  **Generare la Risposta:** Prende i migliori pezzetti di testo, li passa a **Gemini** (il modello di intelligenza artificiale) e dice: "Rispondi a questa domanda usando **SOLO** queste informazioni e indicami le fonti." L'AI riassume e formula la risposta finale.

***

## 3. 💬 L'Interfaccia di Chat (Frontend Flutter)

Questa è l'applicazione web che usi. È progettata per rendere l'interazione il più semplice e utile possibile.

### Funzionalità Utili:

* **Chat Intelligente:** Mantiene traccia di tutte le domande che hai fatto, permettendo al Ricercatore di capire il contesto.
* **Chiarezza e Fonti:** La risposta viene visualizzata in modo chiaro. Ancora più importante, mostra dei pulsanti cliccabili (chiamati **Chip**) con i nomi dei file originali (le **Fonti**).
* **Trasparenza:** Cliccando sul nome del file, puoi **accedere al documento originale** per verificare che l'AI abbia risposto correttamente.
* **Pulsanti di Feedback:** Dopo ogni risposta, trovi un pollice in su 👍 e un pollice in giù 👎. Se la risposta è ottima o sbagliata, puoi dirlo all'AI.
    * **Perché è importante?** Questo feedback viene registrato e permette agli sviluppatori di identificare e correggere i punti deboli del sistema, rendendolo **continuamente più intelligente e preciso**.

In sintesi, il sistema garantisce che le risposte AI siano sempre basate sui **tuoi dati ufficiali**, offrendo velocità e trasparenza in ogni interazione.