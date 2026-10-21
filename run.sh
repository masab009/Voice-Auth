#!/usr/bin/env bash
# Start the voice authentication demo on 0.0.0.0:8781, reachable from the whole network (log: app.log).
#   PYTHON=/path/to/python bash run.sh [--port N] [--host H]
# PYTHON defaults to .venv/bin/python in this folder (see README: Install).
V="$(cd "$(dirname "$0")" && pwd)"
cd "$V" || exit 1
PY="${PYTHON:-$V/.venv/bin/python}"
PORT=8781; HOST_SET=0
for ((i = 1; i <= $#; i++)); do
  [ "${!i}" = "--port" ] && { j=$((i + 1)); PORT="${!j}"; }
  [ "${!i}" = "--host" ] && HOST_SET=1
done
[ $HOST_SET = 1 ] || set -- "$@" --host 0.0.0.0
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"; IP="${IP:-127.0.0.1}"
[ -x "$PY" ] || { echo "no python at $PY: create .venv (README: Install) or set PYTHON=..."; exit 1; }
[ -f models/w2v-bert-2.0_SV/model_lmft_0.14.pth ] || { echo "model missing: run bash download_models.sh"; exit 1; }
if ss -ltn | grep -q ":$PORT "; then echo "already running: http://$IP:$PORT"; exit 0; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TMPDIR="$V/tmp"
mkdir -p "$TMPDIR"
setsid nohup env -u VIRTUAL_ENV -u PYTHONPATH "$PY" app.py "$@" >> app.log 2>&1 < /dev/null &
for i in $(seq 1 120); do ss -ltn | grep -q ":$PORT " && { echo "up: http://$IP:$PORT"; exit 0; }; sleep 1; done
echo "not up after 120 s; see $V/app.log"; tail -5 app.log
