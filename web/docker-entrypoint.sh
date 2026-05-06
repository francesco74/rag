#!/bin/sh

# Create env-config.js with environment variables
cat <<EOF > /usr/share/nginx/html/env-config.js
window.ENV_CONFIG = {
  DOCUMENT_URL: "${DOCUMENT_URL:-http://localhost:8080}",
  REST_URL: "${REST_URL:-http://localhost:8080}",
  API_SECRET_KEY: "${API_SECRET_KEY:-no_secret_key_submitted}",
  PROJECT_NAME: "${PROJECT_NAME:-BiblioLucca - Progetti innovativi}"
};
EOF

echo "Environment configuration created:"
cat /usr/share/nginx/html/env-config.js

# Execute the CMD
exec nginx -g "daemon off;"