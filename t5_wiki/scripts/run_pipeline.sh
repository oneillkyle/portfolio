#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/5] Ingest"
python -m t5_wiki.src.ingest --config "$CONFIG"

echo "[2/5] Transform"
python -m t5_wiki.src.transform --config "$CONFIG"

echo "[3/5] Train"
python -m t5_wiki.src.train --config "$CONFIG"

echo "[4/5] Evaluate"
python -m t5_wiki.src.evaluate --config "$CONFIG" || true

echo "[5/5] Export"
python -m t5_wiki.src.export --config "$CONFIG"
