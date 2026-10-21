#!/usr/bin/env bash
# Public HTTPS link for the demo through a Cloudflare quick tunnel (needs `cloudflared`; no account).
# Browsers only allow the microphone on https or localhost, so use this link from other machines.
# Quick-tunnel URLs change on every start, and anyone with the link can use the page.
#   bash tunnel.sh [port]        # default 8781
V="$(cd "$(dirname "$0")" && pwd)"; PORT="${1:-8781}"; LOG="$V/cloudflared.log"
running() { ps -eo args | grep -q "[c]loudflared tunnel --no-autoupdate --edge-ip-version 4 --protocol http2 --url http://127.0.0.1:$PORT$"; }
link() { grep -v api.trycloudflare "$LOG" 2>/dev/null | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | head -1; }
if ! running; then
  # the trycloudflare API can be slow or refuse; retry until it hands out a URL
  for attempt in 1 2 3 4 5 6; do
    : > "$LOG"
    setsid nohup cloudflared tunnel --no-autoupdate --edge-ip-version 4 --protocol http2 --url "http://127.0.0.1:$PORT" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    for t in $(seq 1 40); do [ -n "$(link)" ] && break 2; kill -0 $pid 2>/dev/null || break; sleep 1; done
    kill $pid 2>/dev/null
  done
fi
echo "Voice auth page: $(link)"
