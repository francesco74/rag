#!/bin/sh
set -e

HTML=/usr/share/nginx/html

# Percorso sotto cui il reverse proxy pubblica l'app (es. /revisione/ con la
# location /revisione/ di k8s/nginx.yaml). Diventa il <base href> di
# index.html: da lì il browser carica main.dart.js e gli altri file. Se non
# corrisponde alla location del proxy, si apre l'applicazione sbagliata.
BASE_HREF="${BASE_HREF:-/}"
case "$BASE_HREF" in /*) ;; *) BASE_HREF="/$BASE_HREF" ;; esac
case "$BASE_HREF" in */) ;; *) BASE_HREF="$BASE_HREF/" ;; esac
case "$BASE_HREF" in
  *[!A-Za-z0-9/._~-]*)
    echo "BASE_HREF non valido: '$BASE_HREF' (ammessi lettere, cifre e / . _ ~ -)" >&2
    exit 1 ;;
esac
# Sostituisce qualunque valore precedente: funziona anche al riavvio.
sed -i "s|<base href=\"[^\"]*\">|<base href=\"$BASE_HREF\">|" "$HTML/index.html"
echo "Base href: $BASE_HREF"

# Configurazione a runtime letta da settings.dart
cat <<EOC > "$HTML/env-config.js"
window.ENV_CONFIG = {
  REVIEW_API_URL: "${REVIEW_API_URL:-http://localhost:5000/review}",
  PROJECT_NAME: "${PROJECT_NAME:-Revisione documenti}",
  FILES_URL: "${FILES_URL:-}"
};
EOC

echo "Environment configuration created:"
cat "$HTML/env-config.js"

exec nginx -g "daemon off;"
