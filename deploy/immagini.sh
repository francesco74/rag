#!/usr/bin/env bash
# Costruisce le immagini del rag e le pubblica nel registry.
#
#   deploy/immagini.sh v4.0.0                  # costruisce e pubblica la versione v4.0.0
#   deploy/immagini.sh v4.0.0 --solo-build     # costruisce soltanto
#   COMPONENTI="rest web-chat" deploy/immagini.sh v4.0.1     # solo alcune immagini
#   REGISTRY=altro.registry/progetto deploy/immagini.sh v4.0.0
#   REGISTRY=francesco74 PREFISSO=rag- deploy/immagini.sh v4.0.0   # Docker Hub: francesco74/rag-rest...
#   MODELLI_LOCALI=1 deploy/immagini.sh v4.0.0  # rest con i modelli locali (vedi sotto)
#
# L'immagine rest, senza MODELLI_LOCALI, non contiene i modelli locali: reranking ed
# embedding vanno su un server (RERANK_PROVIDER=vllm, EMBEDDING_PROVIDER=vllm o esterno).
# Con MODELLI_LOCALI=1 contiene il reranker ONNX mmarco (RERANK_PROVIDER=onnx, modello da
# preparare in api/model_cache con build_fast_model.py) e sentence-transformers
# (EMBEDDING_PROVIDER=local): diversi GB in più.
#
# Il nome dell'immagine è REGISTRY/PREFISSO + componente + :versione, e decide dove viene
# pubblicata. Prima: `docker login` sul registry. Dopo la pubblicazione aggiornare le immagini
# nei manifest di deploy/k8s/ (i manifest della v4 usano v4.0.0) e applicarli.
set -euo pipefail

REGISTRY="${REGISTRY:-registry.intranet.provincia.lucca/rag}"
PREFISSO="${PREFISSO:-}"   # davanti al nome del componente, es. rag-
MODELLI_LOCALI="${MODELLI_LOCALI:-0}"

# componente -> "Dockerfile contesto"
declare -A BUILD=(
    [rest]="api/Dockerfile ."                       # API, worker celery, celery-beat, pod strumenti
    [converter]="converter/Dockerfile ."
    [ingestion]="ingestion/Dockerfile ."
    [saga_extract]="saga_extract/Dockerfile ."
    [direct]="direct/Dockerfile ."
    [detect_boilerplate]="detect_boilerplate/Dockerfile ."
    [web-chat]="web/chat/Dockerfile web/chat"       # chat di tutti i siti
    [web-revisione]="web/revisione/Dockerfile web/revisione"   # sito di amministrazione
)
ORDINE=(rest converter ingestion saga_extract direct detect_boilerplate web-chat web-revisione)
read -r -a COMPONENTI <<< "${COMPONENTI:-${ORDINE[*]}}"

versione="${1:-}"
if [[ -z "$versione" || "$versione" == -* ]]; then
    echo "Uso: $0 VERSIONE [--solo-build]     (es. $0 v4.0.0)" >&2
    exit 2
fi
pubblica=true
[[ "${2:-}" == "--solo-build" ]] && pubblica=false

# le immagini si costruiscono dalla radice del repository
cd "$(dirname "$0")/.."

for c in "${COMPONENTI[@]}"; do
    [[ -n "${BUILD[$c]:-}" ]] || { echo "Componente sconosciuto: $c (validi: ${ORDINE[*]})" >&2; exit 2; }
done

# Il modello di riordino non è nel repository: con i modelli locali l'immagine dell'API lo
# copia da api/model_cache.
if [[ "$MODELLI_LOCALI" == "1" && " ${COMPONENTI[*]} " == *" rest "* ]] \
        && ! ls -d api/model_cache/*/ >/dev/null 2>&1; then
    echo "Manca il modello di riordino in api/model_cache: prepararlo con" >&2
    echo "    cd api && python build_fast_model.py" >&2
    exit 1
fi

for c in "${COMPONENTI[@]}"; do
    read -r dockerfile contesto <<< "${BUILD[$c]}"
    immagine="$REGISTRY/$PREFISSO$c:$versione"
    echo "==> $immagine"
    extra=()
    [[ "$c" == "rest" ]] && extra=(--build-arg "MODELLI_LOCALI=$MODELLI_LOCALI")
    docker build -f "$dockerfile" -t "$immagine" "${extra[@]}" "$contesto"
done

if $pubblica; then
    for c in "${COMPONENTI[@]}"; do
        docker push "$REGISTRY/$PREFISSO$c:$versione"
    done
    echo "Pubblicate in $REGISTRY con la versione $versione."
    echo "Kubernetes: aggiornare le immagini in deploy/k8s/*.yaml se la versione è cambiata, poi kubectl apply."
fi
