#!/usr/bin/env bash
# Startup script for Render. Builds the Chroma index if it doesn't exist yet,
# then launches the API server. On the free tier the index is rebuilt on each
# new deployment; with a persistent disk it's rebuilt only on the first deploy.
#!/usr/bin/env bash
set -e

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
