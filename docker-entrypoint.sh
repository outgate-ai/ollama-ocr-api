#!/usr/bin/env bash
set -euo pipefail

# If we are root (fresh named volume is mounted root-owned), fix ownership of
# the HF model cache and drop privileges to the unprivileged `app` user.

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R app:app /home/app/.cache 2>/dev/null || true
    exec gosu app ocr-api "$@"
fi

exec ocr-api "$@"
