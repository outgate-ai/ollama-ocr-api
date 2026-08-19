#!/usr/bin/env bash
set -euo pipefail

# A fresh named volume mounts root-owned, so fix the cache before dropping privileges.
if [[ "$(id -u)" -eq 0 ]]; then
    chown -R app:app /home/app/.cache 2>/dev/null || true
    exec gosu app ocr-api "$@"
fi

exec ocr-api "$@"
