#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/5] Ingest"
python -m my_pipeline.src.ingest --config "$CONFIG"

echo "[2/5] Transform"
python -m my_pipeline.src.transform --config "$CONFIG"

echo "[3/5] Train"
python -m my_pipeline.src.train --config "$CONFIG"

echo "[4/5] Evaluate"
python -m my_pipeline.src.evaluate --config "$CONFIG" || true

echo "[5/5] Export"
python -m my_pipeline.src.export --config "$CONFIG"
