#!/bin/sh

# Configurazione a runtime letta da settings.dart
cat <<EOC > /usr/share/nginx/html/env-config.js
window.ENV_CONFIG = {
  REVIEW_API_URL: "${REVIEW_API_URL:-http://localhost:5001}",
  PROJECT_NAME: "${PROJECT_NAME:-Revisione documenti}"
};
EOC

echo "Environment configuration created:"
cat /usr/share/nginx/html/env-config.js

exec nginx -g "daemon off;"
