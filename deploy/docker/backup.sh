#!/usr/bin/env bash
# Backup dell'installazione Docker del rag: database, snapshot delle collection di Qdrant e
# documenti (volume dati, senza le cartelle di lavoro temporanee).
#
#   deploy/docker/backup.sh [CARTELLA]     # predefinita /var/backups/rag
#   deploy/docker/backup.sh --dry-run      # mostra cosa farebbe
#
# Da pianificare con il cron del server, es. ogni notte alle 01:30:
#   30 1 * * * /opt/rag/deploy/docker/backup.sh >> /var/log/rag-backup.log 2>&1
# Tiene GIORNI giorni di copie (predefinito 14). Le copie vanno poi portate fuori dal server.
#
# Non si salvano (dati temporanei o ricostruibili): tabelle sessions, session_documents,
# session_parents; collection session_chunks e semantic_cache; code di RabbitMQ; Redis.
#
# Ripristino: deploy/docker/README.md, "Backup e ripristino".
set -euo pipefail

prova=false
if [[ "${1:-}" == "--dry-run" ]]; then
    prova=true
    shift
fi
CARTELLA="${1:-/var/backups/rag}"
GIORNI="${GIORNI:-14}"
COLLECTION=(document_chunks conceptual_dictionary boilerplate_phrases)
data="$(date +%Y%m%d-%H%M)"

cd "$(dirname "$0")"

if $prova; then
    echo "Verrebbero scritti in $CARTELLA:"
    echo "  rag-$data.sql.gz (senza sessions, session_documents, session_parents)"
    for c in "${COLLECTION[@]}"; do echo "  qdrant-$c-$data.snapshot"; done
    echo "  dati-$data.tar.gz (volume /data senza conversions/ e sessions/)"
    echo "e cancellate le copie più vecchie di $GIORNI giorni."
    exit 0
fi

mkdir -p "$CARTELLA"
umask 077
echo "$(date -Is) backup in $CARTELLA"

docker compose exec -T mysql sh -c \
    'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --single-transaction --routines --triggers rag \
     --ignore-table=rag.sessions --ignore-table=rag.session_documents --ignore-table=rag.session_parents' \
    | gzip > "$CARTELLA/rag-$data.sql.gz.parziale"
mv "$CARTELLA/rag-$data.sql.gz.parziale" "$CARTELLA/rag-$data.sql.gz"

# Snapshot creato da Qdrant, scaricato attraverso l'API (che vede Qdrant) e poi cancellato.
for c in "${COLLECTION[@]}"; do
    docker compose exec -T api python - "$c" > "$CARTELLA/qdrant-$c-$data.snapshot.parziale" <<'EOF'
import shutil, sys, urllib.request, json
c = sys.argv[1]
base = f"http://qdrant:6333/collections/{c}/snapshots"
nome = json.load(urllib.request.urlopen(urllib.request.Request(base + "?wait=true", method="POST")))["result"]["name"]
try:
    with urllib.request.urlopen(f"{base}/{nome}") as r:
        shutil.copyfileobj(r, sys.stdout.buffer)
finally:
    urllib.request.urlopen(urllib.request.Request(f"{base}/{nome}", method="DELETE"))
EOF
    mv "$CARTELLA/qdrant-$c-$data.snapshot.parziale" "$CARTELLA/qdrant-$c-$data.snapshot"
done

docker compose exec -T api tar czf - -C /data --exclude=./conversions --exclude=./sessions . \
    > "$CARTELLA/dati-$data.tar.gz.parziale"
mv "$CARTELLA/dati-$data.tar.gz.parziale" "$CARTELLA/dati-$data.tar.gz"

find "$CARTELLA" -maxdepth 1 \( -name 'rag-*.sql.gz' -o -name 'qdrant-*.snapshot' -o -name 'dati-*.tar.gz' \
    -o -name '*.parziale' \) -mtime +"$GIORNI" -delete
echo "$(date -Is) fatto: $(du -sh "$CARTELLA" | cut -f1) in totale"
