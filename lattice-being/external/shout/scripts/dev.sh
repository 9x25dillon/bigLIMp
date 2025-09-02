#!/usr/bin/env bash
set -euo pipefail

AL_ULS_URL=${AL_ULS_URL:-http://localhost:8000}

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s limpsbridge \
    'python -m uvicorn python.mock_al_uls_server:app --host 0.0.0.0 --port 8000'
  tmux split-window -h \
    "cd julia && AL_ULS_URL=$AL_ULS_URL julia --project=. -e 'using LIMPSBridge; LIMPSBridge.serve()'"
  tmux attach -t limpsbridge
else
  (python -m uvicorn python.mock_al_uls_server:app --host 0.0.0.0 --port 8000 &)
  (cd julia && AL_ULS_URL=$AL_ULS_URL julia --project=. -e 'using LIMPSBridge; LIMPSBridge.serve()')
fi