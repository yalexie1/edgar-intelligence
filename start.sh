#!/usr/bin/env bash
# Startup script for Render.
# The Pinecone index is persistent in the cloud — no rebuild needed on deploy.
set -e

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
