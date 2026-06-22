#!/usr/bin/env bash
# Startup script for Render.
#
# Render needs the web server to bind to $PORT quickly. If the Chroma vector
# index is missing, build it in the background while the API starts immediately.
# During that background build, the frontend may temporarily show "Index not
# built". Once the background job finishes, subsequent requests should work.
set -e

if [ ! -d "data/chroma" ]; then
  echo "==> WARNING: data/chroma does not exist. Starting background index build..."
  (
    python - <<'EOF'
from embed_and_search import load_corpus, build_index, CORPUS_PATH
print("==> Background index build started.")
build_index(load_corpus(CORPUS_PATH))
print("==> Background index build finished. Index ready.")
EOF
  ) &
else
  echo "==> Found existing vector index at data/chroma."
fi

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
