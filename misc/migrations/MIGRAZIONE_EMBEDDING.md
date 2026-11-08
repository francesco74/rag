# Cambio del modello di embedding (Gemini → Qwen3-Embedding su vLLM)

Procedura per passare da `EMBEDDING_PROVIDER=gemini` (768 dimensioni) a un modello servito da
vLLM, qui Qwen3-Embedding-4B sul DGX (2560 dimensioni), in Kubernetes, namespace `rag`, a
**sistema fermo**. Da fare dopo la migrazione alla v4 (`MIGRAZIONE_V3_V4.md`), non insieme.

**Non serve re-indicizzare**: `utils/migrate_embeddings.py` ricalcola i vettori dal testo già
salvato in Qdrant, senza conversione né OCR. Le collection originali restano intatte fino
alla sostituzione, e dopo la sostituzione se ne tiene una copia per tornare indietro.

| Collection | Cosa succede |
|---|---|
| `document_chunks` | vettori ricalcolati; nuova collection con i vettori **su disco e una copia int8 in RAM** (`QDRANT_CHUNKS_QUANTIZATION`) |
| `conceptual_dictionary`, `boilerplate_phrases` | vettori ricalcolati (piccole, restano in RAM) |
| `semantic_cache` | ricreata vuota: contiene domande vettorializzate con Gemini |

Con la copia int8 la RAM di `document_chunks` è circa 2,5 KB per chunk (contro i ~3 KB di oggi
a 768 tutto in RAM): la ricerca scorre la copia compressa e riordina i migliori
`QDRANT_OVERSAMPLING` × candidati con i vettori pieni letti dal disco. Lo **spazio su disco**
del volume di Qdrant durante la migrazione: vecchie + nuove + copia delle vecchie, circa
10 KB × chunk per le nuove più 2 × 3 KB × chunk per le vecchie.

```bash
alias k='kubectl -n rag'
mkdir -p ~/migrazione-embedding && cd ~/migrazione-embedding
```

## Configurazione nuova

```
EMBEDDING_PROVIDER=vllm
LOCAL_VLLM_EMBEDDING_MODEL=Qwen3-Embedding-4B
LOCAL_VLLM_EMBEDDING_BASE_URL=http://10.1.1.132:8002/v1
EMBEDDING_DIMENSION=2560
EMBEDDING_QUERY_INSTRUCTION=Given a question about administrative documents, retrieve the passages that answer it
```

- `LOCAL_VLLM_EMBEDDING_MODEL` è il nome con cui vLLM espone il modello (`--served-model-name`).
- `EMBEDDING_DIMENSION` è la dimensione nativa: il rag non chiede vettori troncati, e controlla
  che ogni risposta abbia esattamente questa lunghezza.
- `EMBEDDING_QUERY_INSTRUCTION`: Qwen3-Embedding è asimmetrico, vuole le domande precedute da
  un'istruzione (in inglese) e i documenti senza. Il rag la aggiunge solo alle domande della
  ricerca. Si può cambiare in seguito senza ricalcolare i documenti (va solo svuotata la cache
  semantica).

## 1. Preparazione (senza fermo)

**Immagini.** Le modifiche al codice (fornitore `vllm` con controllo della dimensione e
istruzione, quantizzazione, `migrate_embeddings`, `openai` per `detect_boilerplate`) devono
essere già in produzione: immagini `rest`, `ingestion` e, se usata, `detect_boilerplate`.

**Il modello risponde dal cluster**, con la dimensione attesa:

```bash
k exec deploy/api -- python -c "
import json, urllib.request
r = urllib.request.urlopen(urllib.request.Request('http://10.1.1.132:8002/v1/embeddings',
    data=json.dumps({'model': 'Qwen3-Embedding-4B', 'input': ['prova']}).encode(),
    headers={'Content-Type': 'application/json'}), timeout=30)
print(len(json.load(r)['data'][0]['embedding']))"
```

Deve stampare `2560`. Se non risponde: vLLM avviato, porta 8002 aperta, nessuna regola di
rete che blocchi il traffico dai pod verso il DGX.

**Situazione di partenza:**

```bash
k exec deploy/api -- python -m utils.migrate_embeddings stato | tee stato-prima.txt
```

Dal numero di punti di `document_chunks` si stima la durata: una prova del ritmo si ha al
passo 4 (un messaggio ogni 5000 punti).

## 2. Fermo

Come nel passo 3 della migrazione v4: si ferma ciò che porta documenti nuovi, si aspetta che le
code si svuotino, poi si fermano i servizi. L'API resta accesa (serve per lanciare gli script),
ma nessuno la raggiunge perché si ferma nginx:

```bash
k patch cronjob saga-cron -p '{"spec":{"suspend":true}}'
# eventuale CronJob di detect_boilerplate: sospenderlo allo stesso modo
k exec rabbitmq-server-0 -- rabbitmqctl list_queues name messages    # attendere 0
k scale deploy/ingestion deploy/converter deploy/celery-beat deploy/NOME-WORKER-CELERY --replicas=0
k scale deploy/nginx --replicas=0        # i siti rispondono con errore finché non si riaccende
```

## 3. Backup di Qdrant

Lo script tiene già una copia delle collection, ma uno snapshot protegge anche da errori del
volume:

```bash
k port-forward svc/qdrant 6333:6333 &
for c in document_chunks semantic_cache conceptual_dictionary boilerplate_phrases; do
  nome=$(curl -s -X POST "http://localhost:6333/collections/$c/snapshots?wait=true" \
         | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')
  curl -s -o "qdrant-$c.snapshot" "http://localhost:6333/collections/$c/snapshots/$nome"
done
ls -lh *.snapshot; kill %1
```

## 4. Ricalcolo

Con la configurazione nuova passata solo a questo comando (la ConfigMap resta quella vecchia
fino al passo 5). Il ricalcolo può durare ore: gira in background dentro il pod e scrive il
log sul volume, così non dipende dalla connessione di `kubectl`.

```bash
k exec deploy/api -- sh -c 'nohup env EMBEDDING_PROVIDER=vllm \
    LOCAL_VLLM_EMBEDDING_MODEL=Qwen3-Embedding-4B \
    LOCAL_VLLM_EMBEDDING_BASE_URL=http://10.1.1.132:8002/v1 \
    EMBEDDING_DIMENSION=2560 \
    python -m utils.migrate_embeddings ricalcola > /data/migrate_embeddings.log 2>&1 &'
k exec deploy/api -- tail -f /data/migrate_embeddings.log
```

Alla fine: `Ricalcolo completato`. Per ogni collection il log riporta i punti ricalcolati su
quelli totali; quelli "senza testo" non hanno il campo di testo nel payload e non vengono
copiati (per `document_chunks` dovrebbero essere zero). Se il pod si riavvia o il comando si
interrompe, si rilancia: riparte da capo sulle collection `__nuovo`, le originali non sono mai
toccate.

```bash
k exec deploy/api -- python -m utils.migrate_embeddings stato | tee stato-ricalcolo.txt
```

Le collection `__nuovo` devono avere dimensione 2560 e `document_chunks__nuovo` vettori su
disco e quantizzazione int8.

## 5. Configurazione e sostituzione

In `k8s/rag-config.yaml` le variabili della configurazione nuova (sopra), poi:

```bash
k apply -f k8s/rag-config.yaml
k rollout restart deploy/api && k rollout status deploy/api
k exec deploy/api -- python -m utils.migrate_embeddings sostituisci
k exec deploy/api -- python -m utils.migrate_embeddings stato | tee stato-dopo.txt
```

`sostituisci` copia le collection attuali in `<nome>__vecchio`, mette le ricalcolate al loro
posto con gli stessi indici e ricrea vuota la cache. Ogni copia è verificata prima di
cancellare qualcosa; se si interrompe, si rilancia.

## 6. Soglie

Le soglie di similarità sono tarate sui valori di Gemini; con Qwen3 la distribuzione è diversa
(i testi senza relazione arrivano a ~0.55-0.58). Prima di riaprire, con una ventina di domande
di cui si conosce la risposta, si guardano i punteggi della ricerca vettoriale:

```bash
k exec -i deploy/api -- python - <<'EOF'
from qdrant_client import QdrantClient
from common.config import settings
from common.embedding import init_embedding, embed_query
init_embedding()
c = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
for q in ["Chi ha approvato il bilancio di previsione 2025?",
          "Quali lavori sono stati affidati per le scuole della Versilia?"]:
    hits = c.query_points("document_chunks", query=embed_query(q), limit=15).points
    print(f"\n{q}")
    for h in hits:
        print(f"  {h.score:.3f}  {h.payload.get('file_name')}  {h.payload.get('content', '')[:70]!r}")
EOF
```

| Soglia | Oggi | Cosa guardare |
|---|---|---|
| `QDRANT_SEMANTIC_THRESHOLD` | 0.60 | sotto il punteggio dei passaggi giusti, sopra quello dei passaggi estranei. Con Qwen3 di solito più alta che con Gemini |
| `QDRANT_CONCEPT_THRESHOLD` | 0.80 | i concetti del dizionario devono attivarsi sulle domande che li citano, non su quelle vicine |
| `BOILERPLATE_SIMILARITY_THRESHOLD` | 0.55 | le frasi ricorrenti (intestazioni, formule) devono ancora essere riconosciute |
| `MMR_SIMILARITY_THRESHOLD` | 0.92 | passaggi quasi uguali da considerare doppioni |

La soglia della cache semantica (0.97, in `api/src/rag/cache.py`) riconosce solo domande
praticamente identiche ed è di norma adeguata anche con Qwen3.

Le soglie si cambiano nella ConfigMap (`k apply` + `k rollout restart deploy/api`), senza
ricalcolare nulla.

## 7. Ripartenza

```bash
k scale deploy/NOME-WORKER-CELERY deploy/celery-beat deploy/converter deploy/ingestion --replicas=1
k scale deploy/nginx --replicas=1
k patch cronjob saga-cron -p '{"spec":{"suspend":false}}'
```

Controlli: domande sui due siti, con fonti pertinenti; una ricerca con il filtro per data;
un documento caricato di prova arriva a "indicizzato" (l'ingestione usa il modello nuovo); nei
log di API e ingestione nessun errore di embedding o di dimensione.

**Dopo qualche giorno**, a migrazione confermata, si eliminano le copie:

```bash
k exec deploy/api -- python -m utils.migrate_embeddings pulisci --conferma
```

## Ritorno a Gemini

Finché esistono le copie `__vecchio` (prima di `pulisci`):

1. Fermo come al passo 2.
2. ConfigMap con la configurazione precedente (`EMBEDDING_PROVIDER=gemini`,
   `EMBEDDING_DIMENSION=768`, senza `EMBEDDING_QUERY_INSTRUCTION`), `k apply` e
   `k rollout restart deploy/api`.
3. `k exec deploy/api -- python -m utils.migrate_embeddings ripristina`: rimette le collection
   precedenti (con i parametri di allora) e ricrea vuota la cache a 768.
4. Ripartenza come al passo 7.

I documenti indicizzati dopo il passaggio a Qwen3 non sono nelle copie `__vecchio`: vanno
re-indicizzati. Dopo `pulisci` si torna indietro solo dagli snapshot del passo 3.
