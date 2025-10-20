#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/4] Ingest"
python -m t5_xsum.src.ingest --config "$CONFIG"

echo "[2/4] Create Test Split"
python -m t5_xsum.scripts.make_test_split_pt --config "$CONFIG" --num_lines 1000

echo "[3/4] Train"
python -m t5_xsum.src.train_pt --config "$CONFIG"

echo "[4/4] Evaluate & Test"
echo "  Evaluating..."
python -m t5_xsum.src.advanced_eval_pt --config "$CONFIG" || true
echo "  Testing..."
python -m t5_xsum.src.test_pt --config "$CONFIG" || true

echo "[Export] Saving model..."
python -m t5_xsum.src.export_pt --config "$CONFIG"

echo "Pipeline complete! 🎉"
