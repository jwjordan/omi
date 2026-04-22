#!/usr/bin/env bash
set -e
if [ -f /usr/local/share/ca-certificates/pendant-ca.crt ]; then
    update-ca-certificates --fresh >/dev/null 2>&1 || true
fi
exec "$@"
