#!/bin/sh
set -eu

python -m uvicorn kawaneen.demo.runtime:create_demo_app --factory --host 127.0.0.1 --port 8000 &
api_pid=$!
cleanup() {
  kill "$api_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

i=0
while [ "$i" -lt 60 ]; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2)" >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$i" -ge 60 ]; then
  echo "FastAPI demo service did not become ready" >&2
  exit 1
fi

streamlit run src/kawaneen/ui/app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true &
ui_pid=$!
wait "$ui_pid"
