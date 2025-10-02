#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-"http://ds-v3.api.corp.local:80"}

echo "Running smoke test against $HOST"
curl -sS -X POST "$HOST/healthz" -H "Content-Type: application/json" -d '{}' || true
echo
#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-"http://ds-v3.api.corp.local:80"}

echo "Running smoke test against $HOST"
curl -sS -X POST "$HOST/healthz" -H "Content-Type: application/json" -d '{}' || true
echo
