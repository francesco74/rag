# Migrazione dalla v3 alla v4 (Kubernetes)

Procedura per portare la produzione della v3, namespace `rag`, alla v4.
**Non serve re-indicizzare**: Qdrant mantiene collection, dimensione dei vettori e indici,
MySQL riceve solo tabelle nuove. Il fermo dura il tempo dei passi 3-10 (indicativamente
un'ora, più la durata di `align_dates` sull'archivio).

I manifest sono in `deploy/k8s/`: quelli della v4 nella cartella, quelli della produzione v3
in `deploy/k8s/v3/` (servono per tornare indietro). I segreti veri (`rag-secrets.yaml`,
`regcreds.yaml`) non sono nel repository: ci sono gli esempi `*.example.yaml`.

## Cosa cambia

| Componente | v3 | v4 |
|---|---|---|
| Siti della chat | un deployment per sito con immagine propria (`web-attiamministrativi`, `web-greenlees`), configurato con variabili d'ambiente | un solo deployment `web-chat`: il sito lo riconosce l'API dal dominio (tabella `chat_sites`) |
| nginx | un `server` per dominio, `/uploader/`, `/revisione/`, `/downloads/` | un `server` per tutti i domini, `/amministrazione/`, documenti originali via `/_protected_files/` (volume montato in sola lettura); `/downloads/` rimanda all'API |
| Revisione | `/revisione/` | sito di amministrazione `/amministrazione/` (i vecchi indirizzi non sono rediretti) |
| Caricamento dei documenti | uploader (filebrowser) | dal sito di amministrazione (ruolo archivista): **l'uploader si toglie** |
| API | utente root | utente 1000, come converter e ingestion: i file in `/data` passano all'utente 1000 (passo 6) |
| Ruoli della revisione | lettore, correttore, catalogatore, validatore, revisore, admin | revisore, archivista, admin |
| MySQL | | 6 tabelle nuove (`chat_sites`, `document_uploads`, `document_ocr`, `external_api_keys`, `external_api_key_topics`, `review_login_failures`) |
| Qdrant | date dei metadati in formati diversi | date in formato AAAA-MM-GG (`utils/align_dates.py`): un formato diverso fa sparire il documento dai filtri per data |
| RabbitMQ | | coda nuova `da-convertire-api` (conversioni chieste dai sistemi esterni) |
| Configurazione | `ALLOW_SUBTOPIC_SELECTION`, `WS_PASSWORD` nella ConfigMap | la prima è un'impostazione di ogni sito, la seconda passa in `rag-secrets`; nuove `FILES_X_ACCEL_PREFIX`, `TRUSTED_PROXY_HOPS`, `UPLOAD_MAX_MB`, `OCR_PROVIDER` |
| Converter | | conversione con la libreria pa-common, DOCX e ODT con LibreOffice (immagine più grande) |
| Reranking | ONNX `mmarco` su CPU, dentro il worker | `bge-reranker-v2-m3` servito da vLLM sul DGX (`RERANK_PROVIDER=vllm`); l'immagine `rest` non contiene più il reranker locale |

File usati: `misc/migrations/v3_to_v4.sql` (modifiche al database),
`misc/migrations/v3_to_v4_verifica.sql` (controlli in sola lettura), i manifest in `deploy/k8s/`,
`deploy/k8s/strumenti.yaml` (pod per gli script di migrazione, con più memoria dell'API) e
`deploy/k8s/permessi-dati.yaml` (proprietario dei file).

I comandi si lanciano dalla radice del repository, branch `v4`:

```bash
alias k='kubectl -n rag'
export BK=~/migrazione-v4 && mkdir -p $BK          # backup e report
SAGA="saga-determina-cronjob saga-decreto-deliberativo-cronjob saga-decreto-presidenziale-cronjob"
```

---

## 1. Preparazione (senza fermo)

**Immagini**, etichetta `v4.0.0` (quella dei manifest), con `deploy/immagini.sh`: costruisce
e pubblica `rest` (API, worker, celery-beat, pod strumenti), `converter`, `ingestion`,
`saga_extract`, `direct`, `detect_boilerplate`, `web-chat` (sostituisce
`web-attiamministrativi` e `web-greenlees`) e `web-revisione` nel registry
`registry.intranet.provincia.lucca/rag`.

```bash
docker login registry.intranet.provincia.lucca
deploy/immagini.sh v4.0.0
```

L'immagine `rest` è quella leggera, senza il reranker locale: il reranking lo fa il DGX.

**Reranker sul DGX.** Deve rispondere dal cluster con il nome configurato in `rag-config`
(`RERANK_MODEL_NAME`, cioè il `--served-model-name` di vLLM):

```bash
k exec deploy/api -- python -c "
import json, urllib.request
r = urllib.request.urlopen(urllib.request.Request('http://10.1.1.132:8001/v1/rerank',
    data=json.dumps({'model': 'bge-reranker-v2-m3', 'query': 'bilancio di previsione',
                     'documents': ['Approvazione del bilancio di previsione 2025', 'Lavori stradali']}).encode(),
    headers={'Content-Type': 'application/json'}), timeout=30)
print([(x['index'], round(x['relevance_score'], 3)) for x in json.load(r)['results']])"
```

Deve stampare due punteggi tra 0 e 1, più alto per il primo documento.

**`WS_PASSWORD` nel Secret.** Nella v4 non è più nella ConfigMap. Anche la v3 la legge dal
Secret (entrambi arrivano ai pod come variabili d'ambiente), quindi si può spostare subito:

```bash
k patch secret rag-secrets -p '{"stringData":{"WS_PASSWORD":"<password del web service>"}}'
```

Nel Secret deve esserci anche `AUTH_SECRET_KEY` (almeno 32 caratteri), come già in v3.

**Home dei siti.** `manage_sites` (passo 8) le legge da `web/chat/sites/attiamministrativi` e
`web/chat/sites/greenlees` (testi italiano e inglese, pulsanti, icone). Confrontarle con quelle
dei siti v3 (`web/provincia`, `web/greenlees`) se nel frattempo sono cambiate.

**Stato attuale**, da conservare:

```bash
k get deploy,cronjob,svc,ingress,configmap -o wide > $BK/risorse-v3.txt
```

## 2. Controlli preliminari (sola lettura, senza fermo)

```bash
k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' \
    < misc/migrations/v3_to_v4_verifica.sql
```

Il client si ferma sulla verifica 5 con "Table ... doesn't exist": prima della migrazione è
normale. Guardare:

1. **Parent da pulire.** Se compaiono righe, la pulizia dei metadati
   (`parent_metadata_cleanup.sql`) non è stata fatta: va eseguita al passo 5.
2. **Archivi.** Devono esserci `attiprovincia` e `greenlees` (i `TOPIC_ID` dei siti v3).
   `public_access` decide se i documenti originali si aprono senza login: vale come in v3,
   che già li serviva da `/api/files`.
3. **Ruoli.** Gli utenti con il solo ruolo `lettore` resteranno senza ruoli: dopo la
   migrazione l'amministratore deve assegnargliene uno, altrimenti non entrano.

**Dimensione dei vettori**: deve coincidere con `EMBEDDING_DIMENSION` (non impostata in
produzione: 768).

```bash
k exec deploy/api -- python -c "
from qdrant_client import QdrantClient; from common.config import settings
c = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
for n in ('document_chunks', 'semantic_cache', 'conceptual_dictionary'):
    i = c.get_collection(n); print(n, i.config.params.vectors.size, i.points_count)
print('EMBEDDING_DIMENSION =', settings.embedding_dimension)"
```

## 3. Fermo

Prima ciò che porta documenti nuovi, poi si aspetta che le code si svuotino:

```bash
for c in $SAGA; do k patch cronjob $c -p '{"spec":{"suspend":true}}'; done
k get jobs                                   # nessun job di saga ancora in esecuzione
k scale deploy/uploader --replicas=0
k exec rabbitmq-server-0 -- rabbitmqctl list_queues name messages messages_unacknowledged
```

Quando `messages` è 0 su tutte le code (DLQ escluse), si fermano i servizi; MySQL, Redis,
RabbitMQ e Qdrant restano accesi:

```bash
k scale deploy/nginx deploy/ingestion deploy/converter deploy/celery-beat deploy/worker deploy/api --replicas=0
```

Da qui i siti rispondono con un errore dell'ingress fino al passo 8.

## 4. Backup

**MySQL:**

```bash
k exec deploy/mysql -- sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --single-transaction --routines rag' \
    | gzip > $BK/rag-v3.sql.gz
gunzip -t $BK/rag-v3.sql.gz && ls -lh $BK/rag-v3.sql.gz
```

**Qdrant** (snapshot di ogni collection, scaricati su questo computer):

```bash
k port-forward svc/qdrant 6333:6333 &
for c in document_chunks semantic_cache conceptual_dictionary boilerplate_phrases; do
  nome=$(curl -s -X POST "http://localhost:6333/collections/$c/snapshots?wait=true" \
         | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')
  curl -s -o "$BK/qdrant-$c.snapshot" "http://localhost:6333/collections/$c/snapshots/$nome"
done
ls -lh $BK/*.snapshot; kill %1
```

**Manifest dei documenti** (`align_dates` li modifica), dal pod strumenti, che monta il
volume dei dati:

```bash
k apply -f deploy/k8s/strumenti.yaml && k wait --for=condition=Ready pod/strumenti --timeout=5m
k exec strumenti -- tar czf - --exclude='*.pdf' --exclude='*.png' --exclude='*.jpg' \
    --exclude='*.jpeg' -C /data processed > $BK/processed-v3.tgz
tar tzf $BK/processed-v3.tgz | tail -3
```

Restano solo `.json` e `.md`: i file originali la migrazione non li tocca.

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
    < misc/migrations/v3_to_v4.sql | tee $BK/migrazione-mysql.txt
```

Mostra chi resta senza ruoli e, alla fine, i ruoli di ogni utente. La conversione dei ruoli
avviene una sola volta: rilanciando lo script si ferma con
`Duplicate entry 'v4_ruoli_revisione'` e non cambia nulla.

## 6. Proprietario dei file

In v3 l'API girava come root; nella v4 gira come utente 1000 e deve poter riscrivere manifest e
file caricati scritti allora:

```bash
k apply -f deploy/k8s/permessi-dati.yaml
k wait --for=condition=complete job/permessi-dati --timeout=60m && k logs job/permessi-dati
k delete job permessi-dati
```

## 7. Configurazione, backend e strutture

```bash
k apply -f deploy/k8s/rag-config.yaml
k apply -f deploy/k8s/api.yaml && k rollout status deploy/api
# il pod strumenti legge la configurazione all'avvio: si ricrea
k delete pod strumenti && k apply -f deploy/k8s/strumenti.yaml && k wait --for=condition=Ready pod/strumenti --timeout=5m
k exec strumenti -- python -m utils.setup_qdrant
k exec strumenti -- env BROKER_PORT=15672 python -m utils.setup_rabbitmq
k logs deploy/api --tail=50                  # nessun errore su AUTH_SECRET_KEY o sul database
```

`setup_rabbitmq` usa l'API di gestione di RabbitMQ (porta 15672). Entrambi gli script si
possono rilanciare senza danni.

## 8. Siti della chat, frontend e nginx

**Siti**, con le impostazioni dei deployment v3 (`TOPIC_ID`, `PROJECT_NAME`,
`ALLOW_DATE_FILTER`). La selezione delle sottocategorie è attiva per default, come era
`ALLOW_SUBTOPIC_SELECTION`.

```bash
k cp web/chat/sites/attiamministrativi strumenti:/tmp/sito-atti
k cp web/chat/sites/greenlees strumenti:/tmp/sito-greenlees
k exec strumenti -- python -m utils.manage_sites add attiamministrativi.provincia.lucca.it \
    --topic attiprovincia --name "Atti amministrativi Provincia di Lucca" --date-filter --dir /tmp/sito-atti
k exec strumenti -- python -m utils.manage_sites add archiviogreenlees.provincia.lucca.it \
    --topic greenlees --name "Archivio Ian Greenlees" --dir /tmp/sito-greenlees
k exec strumenti -- python -m utils.manage_sites list
```

Le modifiche successive (testi, pulsanti, icona) si fanno dal sito di amministrazione.

**Frontend e proxy:**

```bash
k apply -f deploy/k8s/web-chat.yaml -f deploy/k8s/revisione.yaml -f deploy/k8s/nginx.yaml
k rollout restart deploy/nginx               # la ConfigMap di nginx non si ricarica da sola
k rollout status deploy/web-chat && k rollout status deploy/revisione && k rollout status deploy/nginx
```

**Rimozione dei vecchi siti e dell'uploader:**

```bash
k delete deploy/attiamministrativi deploy/archiviogreenlees svc/attiamministrativi svc/archiviogreenlees
k delete deploy/uploader svc/uploader
# Configurazione di filebrowser (utenti): da cancellare quando la v4 è confermata
#   k delete pvc filebrowser-config
```

Gli ingress non cambiano: puntano già a nginx. Un sito nuovo, in futuro, richiede solo DNS,
ingress con certificato e `manage_sites add`.

## 9. Dati: date e cache

**Date.** Prima la prova, che conta e scrive il report senza modificare nulla:

```bash
k exec strumenti -- python -m utils.align_dates --dry-run
r=$(k exec strumenti -- sh -c 'ls -t /data/date_alignment_report_*.csv | head -1')
k cp strumenti:$r $BK/report-align-prova.csv
```

Nel report ci sono i valori che verranno **tolti** (date parziali, sentinelle, formati non
riconosciuti): controllarli. Poi l'esecuzione vera:

```bash
k exec strumenti -- python -m utils.align_dates
```

Se si interrompe, rilanciarla: riprende dai documenti non ancora fatti. I valori originali
restano nella tabella `date_alignment_backup`.

**Cache semantica.** Contiene risposte della v3: si svuota e si ricrea.

```bash
k exec strumenti -- python -m utils.qdrant_semantic_cache_reset
```

## 10. Ripartenza

```bash
k apply -f deploy/k8s/worker.yaml -f deploy/k8s/celery-beat.yaml -f deploy/k8s/converter.yaml \
        -f deploy/k8s/ingestion.yaml
k apply -f deploy/k8s/saga-determine-cron.yaml -f deploy/k8s/saga-decreto_deliberativo.yaml \
        -f deploy/k8s/saga-decreto_presidenziale.yaml
# apply non toglie la sospensione del passo 3: va tolta a mano
for c in $SAGA; do k patch cronjob $c -p '{"spec":{"suspend":false}}'; done
k get pods
```

**Pagine OCR dei documenti v3.** La tabella `document_ocr` è nuova e il converter la compila
solo per i documenti che tratta da qui in avanti: per quelli della v3 il dato si stima dai
file archiviati in `processed/` (non trascrive nulla, non chiama modelli). Senza questo passo,
in revisione il filtro **Trascrizione** con OCR / senza OCR non mostra i documenti della v3.
Si lancia nel converter appena è ripartito; prima la prova:

```bash
k exec deploy/converter -- python rebuild_document_ocr.py --dry-run
k exec deploy/converter -- python rebuild_document_ocr.py
```

Non tocca le righe già presenti, quindi si può rilanciare o fare un archivio alla volta
(`--topic NOME`). Su tutto l'archivio è lungo: se la sessione può cadere, lancialo in
background nel pod con il log sul volume
(`k exec deploy/converter -- sh -c 'nohup python rebuild_document_ocr.py > /data/rebuild_ocr.log 2>&1 &'`).
Esce con 1 se resta qualche documento senza dato (file non più in `processed/`): l'elenco è
in fondo all'output.

Il pod strumenti si può tenere per il cambio degli embedding (`MIGRAZIONE_EMBEDDING.md`) o
eliminare: `k delete pod strumenti`.

## 11. Controlli

- [ ] `https://attiamministrativi.provincia.lucca.it` e
      `https://archiviogreenlees.provincia.lucca.it`: home con il nome, i pulsanti e l'icona giusti.
- [ ] Una domanda su ciascun sito, con risposta e fonti; su atti amministrativi una ricerca con
      il filtro per data.
- [ ] Apertura di un documento originale dalle fonti (archivio pubblico; per uno riservato,
      richiesta di login).
- [ ] `https://…/amministrazione/`: login di un amministratore.
- [ ] Ai vecchi "lettori" rimasti senza ruoli (`$BK/migrazione-mysql.txt`) l'amministratore
      assegna revisore o archivista, se devono ancora entrare.
- [ ] Caricamento di un documento di prova dal sito di amministrazione (ruolo archivista):
      deve arrivare a "indicizzato".
- [ ] Verifiche finali del database (tutte e 6, nessun errore):
      `k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -t rag' < misc/migrations/v3_to_v4_verifica.sql`
- [ ] Log senza errori: `k logs deploy/api`, `k logs deploy/worker`, `k logs deploy/converter`,
      `k logs deploy/ingestion`.
- [ ] Il giorno dopo: i tre job di saga sono andati a buon fine (`k get jobs`).
- [ ] Nei log del worker, all'avvio, `Reranker vLLM: modello bge-reranker-v2-m3` e nessun
      errore di reranking durante le domande di prova.

**Soglia del reranking.** `MIN_PROB_THRESHOLD` (predefinita 0.02) è la probabilità minima
perché un passaggio entri nel contesto: era tarata sui punteggi di `mmarco`, mentre
`bge-reranker-v2-m3` restituisce già probabilità 0..1 con una distribuzione diversa. Con le
domande di prova, se le risposte perdono passaggi pertinenti o ne includono di estranei, si
rivede, insieme a `EXTRA_PARENTS_MIN_PROB`, in `deploy/k8s/rag-config.yaml` (`k apply`, poi `k rollout restart deploy/worker`).

Facoltativo, con calma: la pulizia del testo della v4 toglie anche i tag `<u>` dell'OCR.
`python -m utils.reindex_topic ARCHIVIO` (nel pod strumenti) re-indicizza solo i documenti che
li contengono.

## Ritorno alla v3

Ruoli e date sono stati modificati: si torna indietro con i backup del passo 4.

1. Fermare i servizi come al passo 3 (compresi `web-chat` e `revisione`).
2. MySQL:
   ```bash
   gunzip -c $BK/rag-v3.sql.gz | k exec -i deploy/mysql -- sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" rag'
   ```
   Le tabelle nuove della v4 restano, ma la v3 le ignora.
3. Qdrant, per ogni collection:
   ```bash
   k port-forward svc/qdrant 6333:6333 &
   for c in document_chunks semantic_cache conceptual_dictionary boilerplate_phrases; do
     curl -s -X POST "http://localhost:6333/collections/$c/snapshots/upload?priority=snapshot" \
          -F "snapshot=@$BK/qdrant-$c.snapshot"
   done
   kill %1
   ```
4. Manifest dei documenti: `k exec -i strumenti -- tar xzf - -C /data < $BK/processed-v3.tgz`.
5. Manifest della v3, esclusi i segreti (restano quelli del cluster):
   ```bash
   for f in deploy/k8s/v3/*.yaml; do
     case "$f" in *rag-secrets.yaml|*regcreds.yaml) continue ;; esac
     k apply -f "$f"
   done
   k delete deploy/web-chat svc/web-chat
   k rollout restart deploy/nginx
   for c in $SAGA; do k patch cronjob $c -p '{"spec":{"suspend":false}}'; done
   ```
   I file in `/data` restano dell'utente 1000: la v3 li legge e li scrive comunque (l'API v3
   gira come root).
