# Chat: frontend unico per tutti gli archivi

Un solo codice, una sola immagine (`rag/web-chat`) e un solo deploy. Il sito lo decide il dominio: all'avvio il frontend chiede a `/api/site`, sullo stesso indirizzo, nome, archivio e home page. L'immagine non ha variabili d'ambiente.

## Aggiungere un sito

1. Crea l'archivio da riga di comando, se non esiste: `python -m utils.manage_topics add <archivio> --description "<descrizione>" --prompt general`. Le sottocategorie si aggiungono da **sito di amministrazione → Amministrazione → Archivi**.
2. Crea il sito da riga di comando, l'unico modo previsto: `python -m utils.manage_sites add <dominio> --topic <archivio> --name "<nome>"`, con `--dir` per caricare home page e icona.
3. Fai arrivare il dominio a nginx: record DNS e, in Kubernetes, ingress con certificato. Con un ingress e un certificato wildcard basta il DNS.

La home è composta, dall'alto, da testo descrittivo, statistiche sui documenti (calcolate dal sistema), pulsanti e limiti del sistema. Testo e pulsanti si modificano dalla scheda **Siti**; la cartella di `--dir` può contenere `welcome_it.html`, `welcome_en.html`, `favicon.png` e `buttons.json`. Senza testo si mostra il nome del sito; senza la versione inglese si usa quella italiana.

## Migrare i siti esistenti

Una sola volta, con le home in `sites/`:

```bash
kubectl cp web/chat/sites rag/<pod-api>:/tmp/sites
kubectl exec -n rag <pod-api> -- python -m utils.manage_sites add archiviogreenlees.provincia.lucca.it \
  --topic greenlees --name "Archivio Ian Greenlees" --dir /tmp/sites/greenlees
kubectl exec -n rag <pod-api> -- python -m utils.manage_sites add attiamministrativi.provincia.lucca.it \
  --topic attiprovincia --name "Atti amministrativi" --date-filter --dir /tmp/sites/attiamministrativi
```

## Build

```bash
docker build -t rag/web-chat web/chat
```

In locale il sistema completo è in `docker-compose.yml`: `http://greenlees.localhost:8081` e `http://atti.localhost:8081`.
