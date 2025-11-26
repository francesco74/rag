#!/bin/sh

# Create env-config.js with environment variables
cat <<EOF > /usr/share/nginx/html/env-config.js
window.ENV_CONFIG = {
  CLIENT_DESCRIPTION: "${CLIENT_DESCRIPTION:-Default Client}",
  REST_URL: "${REST_URL:-http://localhost:8080}",
  REST_AI_DOC: "${REST_URL:-http://localhost:8080}/ai_doc",
  REST_AI_URL: "${REST_URL:-http://localhost:8080}/ai_url",
  REST_PDF_URL: "${REST_URL:-http://localhost:8080}/pdf_create"
};
EOF

echo "Environment configuration created:"
cat /usr/share/nginx/html/env-config.js

# Execute the CMD
exec nginx -g "daemon off;"