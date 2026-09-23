#!/usr/bin/env bash
# AXE-Diffusion KSL-XE — lifecycle for the GoedelMachines W4A16 diffusion
# server (Diffusion Studio UI + OpenAI-compatible API in one process).
#
#   ./scripts/xe.sh up        start + wait for /health (model load ~18 s)
#   ./scripts/xe.sh down      stop and remove the container
#   ./scripts/xe.sh restart   down + up
#   ./scripts/xe.sh status    container + GPU memory + health + /v1/models
#   ./scripts/xe.sh logs      docker logs -f
#   ./scripts/xe.sh gpu       xpu-smi snapshot
#   ./scripts/xe.sh test      python3 scripts/test_api.py  (needs server up)
#
# Env overrides: AXE_PORT (default 8080), AXE_IMAGE, SERVED_MODEL_NAME.
# GPU pinning: the app binds xpu:0 (= level_zero device 0, the FIRST Intel GPU).

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${AXE_IMAGE:-local/gemma-w4-intel:triton38}"
BASE_IMAGE="${AXE_BASE_IMAGE:-local/gemma-w4-intel:tested}"
PORT="${AXE_PORT:-8080}"
NAME="axe-diffusion-ksl-xe"

ensure_image() {
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "== image $IMAGE present"
    return
  fi
  echo "== building $IMAGE from $BASE_IMAGE (Triton 3.8.0 XPU upgrade)"
  docker rm -f xe-imgbuild >/dev/null 2>&1 || true
  docker run --name xe-imgbuild --entrypoint bash "$BASE_IMAGE" -c \
    "pip install --no-cache-dir 'triton-xpu==3.8.0' --index-url https://download.pytorch.org/whl/xpu"
  docker commit xe-imgbuild "$IMAGE"
  docker rm xe-imgbuild >/dev/null
  echo "== committed $IMAGE"
}

up() {
  ensure_image
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker compose up -d
  ./scripts/xe.sh wait
}

down()   { docker compose down; }
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
  down) down ;;
  restart) down; up ;;
  status) status ;;
  logs) logs ;;
  gpu) gpu ;;
  wait) wait ;;
  test) python3 scripts/test_api.py --base "http://127.0.0.1:${PORT}" ;;
  *) echo "usage: $0 {up|down|restart|status|logs|wait|gpu|test}"; exit 2 ;;
esac