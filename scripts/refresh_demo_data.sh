#!/usr/bin/env sh
# Copy a finished run into the dashboard's bundled sample data, so the UI works with no backend.
# usage: scripts/refresh_demo_data.sh [run-id]   (default: latest)
set -eu
RUNS="${RUNS_DIR:-runs}"
RUN="${1:-$(cat "$RUNS/LATEST")}"
DEST="dashboard/public/demo"
mkdir -p "$DEST"
cp "$RUNS/$RUN/metrics.jsonl"     "$DEST/metrics.jsonl"
cp "$RUNS/$RUN/segmentation.json" "$DEST/segmentation.json"
cp "$RUNS/$RUN/baseline.json"     "$DEST/baseline.json"
cp "$RUNS/$RUN/audit.json"        "$DEST/audit.json"
echo "sample data refreshed from $RUNS/$RUN"
