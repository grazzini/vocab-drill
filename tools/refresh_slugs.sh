#!/bin/sh
# Rebuild index.html every 10 minutes while tools/slugs.py runs, then once more.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f tools/slugs.py >/dev/null; do
  sleep 600
  python3 -I tools/build.py
done
python3 -I tools/build.py
