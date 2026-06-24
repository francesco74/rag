
# Guida Semplice: Come Funziona il Nostro Assistente Intelligente?

Immagina di avere in azienda un assistente virtuale capace di **leggere migliaia di pagine di documenti, contratti, manuali e normative in pochi secondi** e di rispondere a qualsiasi tua domanda in modo preciso, sicuro e senza mai inventare nulla.

Questo è ciò che fa il nostro sistema, basato sulla tecnologia **RAG** (*Retrieval-Augmented Generation*).

Per capire come funziona, non serve essere programmatori. In questa guida vedremo insieme, passo dopo passo, il viaggio che fanno i tuoi documenti, come l'interfaccia ti aiuta a consultarli e come l'intelligenza artificiale riesce a darti sempre la risposta giusta.

## 1. La Filosofia di Base: "L'Esame a Libro Aperto"

Le intelligenze artificiali tradizionali (come ChatGPT o simili) a volte "allucinano", ovvero inventano risposte che sembrano credibili ma sono del tutto false. Questo accade perché rispondono basandosi solo sulla loro memoria generica.

Il nostro sistema funziona invece come uno studente brillante che affronta un **esame a libro aperto**:

1. Non si fida solo della sua memoria.
2. Quando gli fai una domanda, corre in biblioteca.
3. Cerca i libri e i documenti aziendali che parlano di quell'argomento.
4. Rilegge i paragrafi esatti.
5. Usa quelle informazioni ufficiali per scriverti una risposta precisa e documentata.

**Se un'informazione non è presente nei tuoi documenti, l'assistente te lo dice chiaramente, evitando di inventare.**


## 2. Il Viaggio dei Documenti: Dall'Ufficio alla Memoria dell'Assistente

Prima di poter rispondere alle tue domande, il sistema deve "studiare" i tuoi file (PDF, relazioni, immagini, foto di documenti). Questo processo avviene in tre passaggi automatici:


```

┌─────────────────┐       ┌────────────────────┐       ┌───────────────────┐
│ 1. Caricamento  │ ───►  │  2. Lettura e OCR  │ ───►  │  3. Scomposizione │
│   (I file raw)  │       │ (Anche foto brutte)│       │  (Parent-Child)   │
└─────────────────┘       └────────────────────┘       └───────────────────┘

```

### Passo 1: Il Caricamento veloce
Tu o i sistemi aziendali inserite i file in una cartella condivisa. Il sistema si accorge immediatamente dei nuovi arrivi e si prepara a elaborarli in background, senza rallentare il tuo lavoro.

### Passo 2: La Lettura Intelligente (OCR)
Il sistema apre i documenti e li traduce in un testo pulito e uniforme.
* **Cosa succede se il file è una scansione o una foto sbiadita?** Il sistema attiva una "vista digitale" (OCR) che riconosce i caratteri scritti nelle immagini.
* **Cosa succede se le regole di sicurezza bloccano la lettura?** Se un sistema di intelligenza artificiale si rifiuta di leggere una pagina per un falso allarme di sicurezza, l'assistente non si ferma: passa immediatamente il compito a un "occhio di riserva" (Google Vision) per essere sicuro di non perdere nemmeno una riga del tuo documento.

### Passo 3: La scomposizione "Genitore-Figlio" (L'Indice del Libro)
Per cercare velocemente nei documenti, il sistema usa una strategia geniale chiamata **Parent-Child** (Genitore-Figlio):
* **I "Figli" (I riassunti veloci):** Il sistema spezza il documento in piccolissimi frammenti (poche righe ciascuno). Questi frammenti servono al sistema come un *indice analitico super veloce* per capire in un millisecondo dove si parla di un certo argomento.
* **I "Genitori" (La pagina intera):** Il sistema conserva l'intera pagina o paragrafo originale (il Genitore) in un archivio sicuro.
* **L'analogia:** Quando fai una domanda, il sistema trova in un lampo il frammento "Figlio" nell'indice e poi va a prendere la pagina "Genitore" intera per avere tutto il contesto e capire il senso profondo del discorso.


## 3. L'Interfaccia Utente: La Tua Scrivania Digitale

Il frontend (l'applicazione con cui interagisci) è stato progettato per darti il massimo controllo, totale trasparenza e un'esperienza d'uso accessibile e senza stress. Ecco le sue caratteristiche principali:

### 🌟 Accoglienza Guidata (Il Mini-Tour)
Al tuo primissimo accesso all'applicazione, non sarai lasciato solo. Il sistema ti mostrerà un **tour guidato interattivo** che evidenzia e spiega brevemente le funzioni dei tasti principali (lingua, filtri, temi visivi, invio della domanda e pulizia della chat), così da farti sentire subito a casa.

### 🌐 Rilevamento della Lingua Automatico
Il sistema legge le impostazioni del tuo browser e si adatta automaticamente a te. Se il tuo computer è impostato in italiano, l'interfaccia ti parlerà in italiano; se è impostato in inglese, passerà all'inglese. In ogni caso, puoi cambiare la lingua in qualsiasi momento con un semplice tocco sulla bandierina nella barra in alto.

### 👓 Accessibilità e Benessere Visivo (3 Temi ed Alto Contrasto)
Per non affaticare la vista durante le lunghe sessioni di lavoro, puoi cambiare l'aspetto visivo dell'app con un solo clic (o usando la scorciatoia da tastiera `Alt + T`):
* **Tema Chiaro:** Pulito e solare, ideale per ambienti molto illuminati.
* **Tema Scuro:** Riposante, riduce l'emissione di luce blu dallo schermo.
* **Tema ad Alto Contrasto:** Sviluppato specificamente per ipovedenti o per l'uso sotto la luce diretta del sole. Utilizza uno sfondo nero assoluto con scritte bianche e dettagli in giallo acceso per massimizzare la leggibilità dei testi.

### 🔍 Il Filtro dei Sotto-Argomenti (Scegli tu cosa deve studiare)
A volte non hai bisogno di cercare in tutta la biblioteca aziendale, ma solo su un gruppo specifico di documenti (ad esempio, solo i contratti del 2026, o solo i manuali tecnici di una determinata macchina). Grazie al pulsante con l'icona del **filtro**, si aprirà una lista di controllo con cui potrai decidere quali categorie o fascicoli attivi includere o escludere dalla ricerca.

### ⏳ Trasparenza nei Tempi di Attesa (Nessun dubbio sul funzionamento)
Cercare tra migliaia di pagine e strutturare una risposta corretta richiede qualche secondo. Per rassicurarti sul fatto che il sistema stia lavorando attivamente dietro le quinte:
1. L'app mostra un indicatore visivo animato in tempo reale.
2. Se la domanda è particolarmente complessa e richiede **più di 20 secondi**, l'app si aggiorna automaticamente mostrando un messaggio di cortesia per informarti che l'elaborazione sta richiedendo un po' più di tempo del solito, garantendo trasparenza e tranquillità.

### 🔗 Fonti Cliccabili per la Massima Trasparenza
Ogni volta che l'assistente risponde, non devi credergli sulla parola. Sotto la risposta troverai un menu a tendina espandibile con il numero esatto di fonti utilizzate (es. "Fonti: 3"). Cliccandoci sopra, appariranno dei comodi pulsanti (chip) che mostrano i nomi dei file originali. **Cliccando sul nome del file, potrai scaricare o aprire direttamente il documento sorgente nel punto esatto**, per una verifica immediata al 100%.

### 💬 Sistema di Feedback con Commenti (Miglioramento Continuo)
Se una risposta è eccezionale o se noti qualcosa che non va, puoi dircelo subito usando i pulsanti "Mi piace" (Pollice in su) e "Non mi piace" (Pollice in giù). Al clic, il sistema ti mostrerà una finestra in cui potrai lasciare un commento facoltativo. This feedback viene inviato istantaneamente al database centrale, consentendo ai supervisori umani di correggere i documenti o affinare il sistema.

## 4. Come Risponde alle Tue Domande? (Il Cervello del Sistema)

Quando scrivi una domanda nella chat dell'assistente, si attiva una catena di controllo qualità unica nel suo genere.


```
   La tua domanda (es. "Ci sono sconti sulle auto?")
                           │
                           ▼
        [ Traduttore di Sinonimi e Concetti ]
         (Cerca anche "veicoli", "macchine")
                           │
                           ▼
        [ Ricerca Doppia: Significato + Codici ]
                           │
                           ▼
           [ Il Riordinatore di Priorità ]
          (Mette le pagine migliori in cima)
                           │
                           ▼
           [ Generazione della Risposta ]
                           │
                           ▼
           [ Il Revisore di Qualità (Grader) ]
         (Verifica che non ci siano invenzioni)
         /                                   \
  (Approvato!)                           (Bocciato...)
       ▼                                       ▼
Risposta in chat                      Riformula la ricerca
                                         e riprova da solo

```



1. **Capisce cosa intendi (anche se usi parole diverse):** Se chiedi *"Ci sono sconti sulle auto?"*, l'assistente capisce che parole come *"veicoli"*, *"vetture"* o *"macchine"* indicano lo stesso concetto ed espande la ricerca in automatico.
2. **Esegue una doppia ricerca:** Cerca contemporaneamente sia per significato (semantica) che per codici/parole esatte (sintattica), così da non mancare date o numeri di protocollo.
3. **Mette in ordine i risultati:** Se trova molti documenti, li analizza e seleziona solo i più pertinenti e aggiornati, escludendo i doppioni.
4. **Il controllo qualità (Il Revisore virtuale):** Un secondo controllore di intelligenza artificiale indipendente prende la risposta appena scritta dall'assistente e la confronta con i documenti originali. Se la risposta è corretta e documentata, ti viene mostrata; se contiene imprecisioni, viene bocciata e l'assistente riformula la ricerca in background per darti un risultato migliore.
5. **Ricorda le risposte (La scorciatoia):** Se un tuo collega ha fatto una domanda identica poco prima, il sistema recupera istantaneamente la risposta approvata e te te la mostra in meno di un secondo.

## 5. Gestione degli Imprevisti (La Modalità Manutenzione)

Sappiamo che nei contesti aziendali l'affidabilità è fondamentale. Se il sistema dovesse subire un'interruzione di rete o se il server dovesse essere temporaneamente spento per manutenzione o aggiornamenti, l'interfaccia utente non andrà in crash mostrando strani errori di programmazione. 

L'applicazione mostrerà una **schermata di cortesia molto chiara e rilassante** con un'icona di cantiere in corso, spiegando la situazione e offrendoti un comodo pulsante "Riprova connessione" per ricollegarti non appena i servizi torneranno attivi.
