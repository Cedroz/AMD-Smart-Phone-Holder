#!/usr/bin/env bash
set -euo pipefail
followme_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$followme_root"
exec "$followme_root/.venv/bin/python" -m followme.phase5 run --live --perception native --config "$followme_root/config/phase5/live-trial.yaml" "$@"
