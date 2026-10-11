#!/usr/bin/env bash
# Start the phone app. Scan the QR code it prints, or open the address it shows.
# Usage: ./app
cd "$(dirname "$0")"

# Free the app's port if an older copy is still running.
pkill -f '^python3 (remote/)?server\.py' && sleep 1

exec python3 remote/server.py "$@"
