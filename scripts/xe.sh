#!/usr/bin/env bash
# AXE-Diffusion KSL-XE — lifecycle for the W4A16 diffusion server
# (Diffusion Studio UI + OpenAI-compatible API in one process).
#
#   ./scripts/xe.sh up        build image if missing, fetch the AXE
#                             checkpoint, start + wait for /health
#   ./scripts/xe.sh fetch     prefetch srswti/axe-diffusion-ksl-xe into the
#                             HF cache (revision ${AXE_REVISION:-main})
#   ./scripts/xe.sh down      stop and remove the container
#   ./scripts/xe.sh restart   down + up
#   ./scripts/xe.sh status    container + GPU memory + health + /v1/models
#   ./scripts/xe.sh logs      docker logs -f
#   ./scripts/xe.sh gpu       xpu-smi snapshot
#   ./scripts/xe.sh test      python3 scripts/test_api.py  (needs server up)
#
# Env overrides: AXE_PORT (default 8080), AXE_IMAGE, AXE_BASE_IMAGE,
# AXE_REVISION (default main), SERVED_MODEL_NAME, HF_HOME.
# GPU pinning: the app binds xpu:0 (= level_zero device 0, the FIRST Intel GPU).

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${AXE_IMAGE:-local/axe-diffusion-ksl-xe:triton38}"
BASE_IMAGE="${AXE_BASE_IMAGE:-local/gemma-w4-intel:tested}"
PORT="${AXE_PORT:-8080}"
NAME="axe-diffusion-ksl-xe"
AXE_REVISION="${AXE_REVISION:-main}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

ensure_image() {
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "== image $IMAGE present"
    return
  fi
  echo "== building $IMAGE from $BASE_IMAGE (Triton 3.8.0 XPU upgrade)"
  docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .
  echo "== built $IMAGE"
}

fetch() {
  echo "== fetching srswti/axe-diffusion-ksl-xe (revision ${AXE_REVISION}) =="
  # Weights + config only — local kernels/ are never overwritten from remote.
  hf download srswti/axe-diffusion-ksl-xe \
    --revision "${AXE_REVISION}" \
    --include '*.safetensors' --include '*.json' --include '*.jinja' \
    --cache-dir "${HF_CACHE_DIR}"
}

up() {
  ensure_image
  fetch
  if [ "${AXE_CHECKS:-1}" = "1" ]; then
    python3 scripts/edit_bench.py start
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker compose up -d
  ./scripts/xe.sh wait
}

down() { docker compose down; python3 scripts/edit_bench.py stop; }
status() {
  echo "== container =="
  docker ps -a --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo "== health =="
  curl -s -m 3 "http://127.0.0.1:${PORT}/health" || echo "unreachable"
  echo
  echo "== openai models =="
  curl -s -m 3 "http://127.0.0.1:${PORT}/v1/models" || true
  echo
  ./scripts/xe.sh gpu
}
logs()   { docker logs -f "$NAME"; }
gpu()    { xpu-smi 2>/dev/null | grep 'MiB /' | sed 's/^ *//'; }
wait() {
  echo "== waiting for $NAME on :${PORT} =="
  for _ in $(seq 1 60); do
    curl -s -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ready":true' && {
      echo "READY"; return
    }
    sleep 5
  done
  echo "TIMEOUT waiting for $NAME" >&2
  docker logs --tail 20 "$NAME" || true
  exit 1
}

case "${1:-}" in
  up) up ;;
  fetch) fetch ;;
  down) down ;;
  restart) down; up ;;
  status) status ;;
  logs) logs ;;
  gpu) gpu ;;
  wait) wait ;;
  test) python3 scripts/test_api.py --base "http://127.0.0.1:${PORT}" ;;
  checks) python3 scripts/edit_bench.py start ;;
  eval) python3 scripts/edit_bench.py run --base "http://127.0.0.1:${PORT}" ;;
  *) echo "usage: $0 {up|fetch|down|restart|status|logs|wait|gpu|test|checks|eval}"; exit 2 ;;
esac