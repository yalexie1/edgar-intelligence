#!/usr/bin/env bash
# Startup script for Render. Start the API server immediately so Render can
# detect the open web port. Do not build/rebuild the Chroma index in the web
# startup path, because that can delay port binding and cause deploy failures.
set -e

if [ ! -d "data/chroma" ]; then
  echo "==> WARNING: data/chroma does not exist. The vector index may be missing."
  echo "==> Build it separately with: python embed_and_search.py --rebuild"
else
  echo "==> Found existing vector index at data/chroma."
fi

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
