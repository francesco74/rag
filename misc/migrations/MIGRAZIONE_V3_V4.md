# Migrazione dalla v3 alla v4 (Kubernetes)

Procedura per portare un'installazione di produzione della v3, namespace `rag`, alla v4.
**Non serve re-indicizzare**: Qdrant mantiene collection, dimensione dei vettori e indici,
MySQL riceve solo tabelle nuove. Il fermo dura il tempo dei passi 3-9 (indicativamente
un'ora, più la durata di `align_dates` sull'archivio).

## Cosa cambia

| Componente | v3 | v4 |
|---|---|---|
| Siti della chat | un deployment per sito (`attiamministrativi`, `archiviogreenlees`), configurato con variabili d'ambiente e una ConfigMap `site-*` | un solo deployment `web-chat`: il sito lo riconosce l'API dal dominio (tabella `chat_sites`) |
| Revisione | `/revisione/` | sito di amministrazione `/amministrazione/` (i vecchi link sono rediretti) |
| Caricamento dei documenti | uploader (filebrowser) su `/uploader/` | dal sito di amministrazione (ruolo archivista): **l'uploader si toglie** |
| Ruoli della revisione | lettore, correttore, catalogatore, validatore, revisore, admin | revisore, archivista, admin |
| MySQL | | 6 tabelle nuove (`chat_sites`, `document_uploads`, `document_ocr`, `external_api_keys`, `external_api_key_topics`, `review_login_failures`) |
| Qdrant | date dei metadati in formati diversi | date in formato AAAA-MM-GG (`utils/align_dates.py`): un formato diverso fa sparire il documento dai filtri per data |
| RabbitMQ | | coda nuova `da-convertire-api` (conversioni chieste dai sistemi esterni) |
| Configurazione | `ALLOW_SUBTOPIC_SELECTION` | tolta: è un'impostazione di ogni sito; nuove `TRUSTED_PROXY_HOPS`, `UPLOAD_MAX_MB`, `OCR_PROVIDER` e altre, con valori predefiniti |
| Converter | | conversione con la libreria pa-common, DOCX e ODT con LibreOffice (immagine più grande) |

File usati: `misc/migrations/v3_to_v4.sql` (modifiche al database),
`misc/migrations/v3_to_v4_verifica.sql` (controlli in sola lettura) e i manifest in `k8s/`.

Nei comandi, `V4` è l'etichetta delle immagini nuove (es. `v4.0.0`) e `REG` il registry:

```bash
export REG=registry.intranet.provincia.lucca/rag V4=v4.0.0
alias k='kubectl -n rag'
mkdir -p ~/migrazione-v4 && cd ~/migrazione-v4     # backup e report finiscono qui
```

---

## 1. Preparazione (senza fermo)

**Immagini.** Dal branch `v4` del repository, costruire e pubblicare:

| Immagine | Dockerfile / contesto |
|---|---|
| `$REG/rest:$V4` | `api/Dockerfile` (API, worker celery, celery-beat) |
| `$REG/converter:$V4` | `converter/Dockerfile` |
| `$REG/ingestion:$V4` | `ingestion/Dockerfile` |
| `$REG/direct:$V4` | `direct/Dockerfile` |
| `$REG/saga_extract:$V4` | `saga_extract/Dockerfile` |
| `$REG/detect_boilerplate:$V4` | `detect_boilerplate/Dockerfile` |
| `$REG/web-chat:$V4` | contesto `web/chat` (sostituisce `web-greenlees` e `web-attiamministrativi`) |
| `$REG/web-revisione:$V4` | contesto `web/revisione` |

```bash
docker build -t $REG/rest:$V4 -f api/Dockerfile . && docker push $REG/rest:$V4
# ...stesso schema per le altre; per i frontend:
docker build -t $REG/web-chat:$V4 web/chat && docker push $REG/web-chat:$V4
```

**Manifest.** Aggiornare le immagini a `$V4` in `k8s/web-chat.yaml`, `k8s/revisione.yaml`,
`k8s/api.yaml`, `k8s/celery-beat.yaml`, `k8s/converter.yaml`, `k8s/ingestion.yaml`,
`k8s/saga_cron.yaml` e nel manifest del worker celery. `k8s/nginx.yaml` è già quello della v4:
inoltra tutto il traffico a `web-chat`, `/api/` all'API, `/amministrazione/` alla revisione,
non ha più `/uploader/` e redirige `/revisione/` su `/amministrazione/`.

**Nomi dei deployment.** Elenco con le immagini, da conservare (serve anche per tornare
indietro):

```bash
k get deploy,cronjob -o wide > deploy-v3.txt
k get deploy -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.replicas}{"\t"}{.spec.template.spec.containers[*].image}{"\n"}{end}' | tee repliche-v3.txt
```

Il worker celery è il deployment con l'immagine `rest` e il comando `celery ... worker`.

## 2. Controlli preliminari (sola lettura, senza fermo)

```bash
k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' \
    < misc/migrations/v3_to_v4_verifica.sql
```

Il client si ferma sulla verifica 5 con "Table ... doesn't exist": prima della migrazione è
normale. Guardare:

1. **Parent da pulire.** Se compaiono righe, la pulizia dei metadati
   (`parent_metadata_cleanup.sql`) non è stata fatta: va eseguita al passo 5.
2. **Archivi.** I `topic_id` dei siti (oggi `attiprovincia` e `greenlees`, dalla
   variabile `TOPIC_ID` dei vecchi deployment) devono esserci.
3. **Ruoli.** Gli utenti con il solo ruolo `lettore` resteranno senza ruoli: dopo la
   migrazione l'amministratore deve assegnargliene uno, altrimenti non entrano.

**Dimensione dei vettori.** Deve coincidere con `EMBEDDING_DIMENSION` (predefinita 768),
perché la v4 crea le collection con quella dimensione:

```bash
k exec deploy/api -- python -c "
from qdrant_client import QdrantClient; from common.config import settings
c = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
for n in ('document_chunks', 'semantic_cache', 'conceptual_dictionary'):
    i = c.get_collection(n); print(n, i.config.params.vectors.size, i.points_count)
print('EMBEDDING_DIMENSION =', settings.embedding_dimension)"
```

**Segreti.** `AUTH_SECRET_KEY` (almeno 32 caratteri) deve essere in `rag-secrets`, come in v3:
senza, il login della revisione e della chat è disabilitato.

## 3. Fermo

Prima si ferma ciò che porta documenti nuovi, poi si aspetta che le code si svuotino.

```bash
k patch cronjob saga-cron -p '{"spec":{"suspend":true}}'
k scale deploy/uploader --replicas=0
```

Code di RabbitMQ: aspettare che `messages` sia 0 su tutte (DLQ escluse):

```bash
k exec rabbitmq-server-0 -- rabbitmqctl list_queues name messages messages_unacknowledged
```

Poi si fermano i servizi applicativi (MySQL, Redis, RabbitMQ, Qdrant e nginx restano accesi;
la chat mostra la pagina di manutenzione):

```bash
k scale deploy/ingestion deploy/converter deploy/celery-beat deploy/api --replicas=0
k scale deploy/NOME-WORKER-CELERY --replicas=0
```

## 4. Backup

**MySQL:**

```bash
k exec deploy/mysql -- sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --single-transaction --routines rag' \
    | gzip > rag-v3.sql.gz
gunzip -t rag-v3.sql.gz && ls -lh rag-v3.sql.gz
```

**Qdrant** (snapshot di ogni collection, scaricati su questo computer):

```bash
k port-forward svc/qdrant 6333:6333 &
for c in document_chunks semantic_cache conceptual_dictionary; do
  nome=$(curl -s -X POST "http://localhost:6333/collections/$c/snapshots?wait=true" \
         | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')
  curl -s -o "qdrant-$c.snapshot" "http://localhost:6333/collections/$c/snapshots/$nome"
  ls -lh "qdrant-$c.snapshot"
done
kill %1
```

**Manifest dei documenti** (`align_dates` li modifica), dal pod nginx che monta il volume:

```bash
k exec deploy/nginx -- tar czf - --exclude='*.pdf' --exclude='*.png' --exclude='*.jpg' \
    --exclude='*.jpeg' -C /data processed > processed-v3.tgz
tar tzf processed-v3.tgz | tail -3
```

Sono esclusi i file originali, che la migrazione non tocca; restano solo `.json` e `.md`.

## 5. MySQL

Se il passo 2 ha trovato parent da pulire, eseguire prima `misc/migrations/parent_metadata_cleanup.sql`
un passo alla volta (ha la propria copia di sicurezza e il `COMMIT` manuale), da una sessione
interattiva:

```bash
k exec -it deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag'
```

Poi le modifiche della v4:

```bash
k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' \
    < misc/migrations/v3_to_v4.sql | tee migrazione-mysql.txt
```

Mostra chi resta senza ruoli e, alla fine, i ruoli di ogni utente. La conversione dei ruoli
avviene una sola volta: rilanciando lo script si ferma con
`Duplicate entry 'v4_ruoli_revisione'` e non cambia nulla.

## 6. Configurazione

In `k8s/rag-config.yaml`:

- togliere `ALLOW_SUBTOPIC_SELECTION` (ora è un'impostazione del sito, passo 8);
- aggiungere `TRUSTED_PROXY_HOPS: "2"` (ingress + nginx: serve al blocco dei login per
  indirizzo IP);
- aggiungere `UPLOAD_MAX_MB: "100"` (deve restare sotto `client_max_body_size` di `/api/`
  in nginx, 110m);
- controllare l'OCR: `OCR_PROVIDER` è `gemini` per default, con `OCR_MODEL_NAME` già presente.

Le altre variabili nuove hanno valori predefiniti adatti; l'elenco completo è nella tabella
della configurazione in `README-TECH.MD`.

```bash
k apply -f k8s/rag-config.yaml
```

## 7. Backend e strutture

Si avvia solo l'API, per preparare code e collection:

```bash
k apply -f k8s/api.yaml                  # immagine $V4
k scale deploy/api --replicas=1 && k rollout status deploy/api
k exec deploy/api -- python -m utils.setup_qdrant
k exec deploy/api -- env BROKER_PORT=15672 python -m utils.setup_rabbitmq
```

`setup_rabbitmq` usa l'API di gestione di RabbitMQ (porta 15672); entrambi gli script si
possono rilanciare senza danni. Nei log dell'API non devono comparire errori su
`AUTH_SECRET_KEY` o sul database:

```bash
k logs deploy/api --tail=50
```

## 8. Siti della chat e frontend

**Siti.** Si copiano home, pulsanti e icone dei siti nel pod e si creano i siti, con le
impostazioni dei vecchi deployment (`TOPIC_ID`, `PROJECT_NAME`, `ALLOW_DATE_FILTER`). La
selezione delle sottocategorie è attiva per default, come era `ALLOW_SUBTOPIC_SELECTION`.

```bash
POD=$(k get pod -l app=api -o name | head -1 | cut -d/ -f2)
k cp web/chat/sites/attiamministrativi $POD:/tmp/sito-atti
k cp web/chat/sites/greenlees $POD:/tmp/sito-greenlees

k exec deploy/api -- python -m utils.manage_sites add attiamministrativi.provincia.lucca.it \
    --topic attiprovincia --name "Atti amministrativi Provincia di Lucca" --date-filter --dir /tmp/sito-atti
k exec deploy/api -- python -m utils.manage_sites add archiviogreenlees.provincia.lucca.it \
    --topic greenlees --name "Archivio Ian Greenlees" --dir /tmp/sito-greenlees
k exec deploy/api -- python -m utils.manage_sites list
```

Se la home di un sito era stata cambiata direttamente nella ConfigMap del cluster, prima si
recupera quella (`k get configmap site-attiamministrativi -o yaml`). Le modifiche successive
si fanno dal sito di amministrazione.

**Frontend e proxy:**

```bash
k apply -f k8s/web-chat.yaml -f k8s/revisione.yaml -f k8s/nginx.yaml
k rollout restart deploy/nginx          # la ConfigMap di nginx non si ricarica da sola
k rollout status deploy/web-chat && k rollout status deploy/nginx
```

**Rimozione dei vecchi siti e dell'uploader:**

```bash
k delete deploy/attiamministrativi deploy/archiviogreenlees svc/attiamministrativi svc/archiviogreenlees
k delete configmap site-attiamministrativi site-greenlees --ignore-not-found
k delete deploy/uploader svc/uploader
# La configurazione di filebrowser (utenti) si può cancellare quando la v4 è confermata:
#   k delete pvc filebrowser-config
```

Gli ingress (`ingress-attiamministrativi`, `ingress-archiviogreenlees`) non cambiano:
puntano già a nginx. Un sito nuovo, in futuro, richiede solo DNS, ingress con certificato e
`manage_sites add`.

## 9. Dati: date e cache

**Date.** Prima la prova, che conta e scrive il report senza modificare nulla:

```bash
k exec deploy/api -- python -m utils.align_dates --dry-run
k exec deploy/api -- sh -c 'ls -t /data/date_alignment_report_*.csv | head -1'
k cp $POD:/data/date_alignment_report_AAAAMMGG_HHMMSS.csv ./report-align-dry.csv
```

Nel report ci sono i valori che verranno **tolti** (date parziali, sentinelle, formati non
riconosciuti): controllarli. Poi l'esecuzione vera:

```bash
k exec deploy/api -- python -m utils.align_dates
```

Se si interrompe, rilanciarla: riprende dai documenti non ancora fatti. I valori originali
restano nella tabella `date_alignment_backup`.

**Cache semantica.** Contiene risposte della v3: si svuota e si ricrea.

```bash
k exec deploy/api -- python -c "
from qdrant_client import QdrantClient; from common.config import settings
QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port).delete_collection('semantic_cache')"
k exec deploy/api -- python -m utils.setup_qdrant
```

## 10. Ripartenza

```bash
k apply -f k8s/converter.yaml -f k8s/ingestion.yaml -f k8s/celery-beat.yaml -f k8s/saga_cron.yaml
k apply -f MANIFEST-WORKER-CELERY.yaml
# repliche come in repliche-v3.txt
k scale deploy/converter deploy/ingestion deploy/celery-beat --replicas=1
k scale deploy/NOME-WORKER-CELERY --replicas=1
k patch cronjob saga-cron -p '{"spec":{"suspend":false}}'
k get pods
```

## 11. Controlli

- [ ] `https://attiamministrativi.provincia.lucca.it` e
      `https://archiviogreenlees.provincia.lucca.it`: home con il nome, i pulsanti e l'icona giusti.
- [ ] Una domanda su ciascun sito, con risposta e fonti; su atti amministrativi una ricerca con
      il filtro per data.
- [ ] Apertura di un documento originale dalle fonti (pubblico e riservato).
- [ ] `https://…/revisione/` porta a `/amministrazione/`; login di un amministratore.
- [ ] Ai vecchi "lettori" rimasti senza ruoli (`migrazione-mysql.txt`) l'amministratore
      assegna revisore o archivista, se devono ancora entrare.
- [ ] Caricamento di un documento di prova dal sito di amministrazione (ruolo archivista):
      deve arrivare a "indicizzato".
- [ ] Verifiche finali del database (tutte e 6, nessun errore):
      `k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' < misc/migrations/v3_to_v4_verifica.sql`
- [ ] Log senza errori: `k logs deploy/api`, `k logs deploy/converter`, `k logs deploy/ingestion`.

Facoltativo, con calma: la pulizia del testo della v4 toglie anche i tag `<u>` dell'OCR.
`python -m utils.reindex_topic ARCHIVIO` re-indicizza solo i documenti che li contengono.

## Ritorno alla v3

Ruoli e date sono stati modificati: si torna indietro con i backup del passo 4.

1. Fermare i servizi come al passo 3.
2. MySQL:
   ```bash
   gunzip -c rag-v3.sql.gz | k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag'
   ```
   Le tabelle nuove della v4 restano ma la v3 le ignora.
3. Qdrant, per ogni collection:
   ```bash
   k port-forward svc/qdrant 6333:6333 &
   for c in document_chunks semantic_cache conceptual_dictionary; do
     curl -s -X POST "http://localhost:6333/collections/$c/snapshots/upload?priority=snapshot" \
          -F "snapshot=@qdrant-$c.snapshot"
   done
   kill %1
   ```
4. Manifest dei documenti:
   `k exec -i deploy/converter -- tar xzf - -C /data < processed-v3.tgz` (con il converter v3
   avviato, che monta `/data` in scrittura).
5. Riapplicare i manifest e le immagini della v3 (`deploy-v3.txt`), compresi i deployment
   dei vecchi siti, l'uploader e la vecchia ConfigMap di nginx, poi le repliche di
   `repliche-v3.txt`.
