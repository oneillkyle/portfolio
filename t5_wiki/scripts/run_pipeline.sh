#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/5] Ingest"
python -m t5_wiki.src.ingest --config "$CONFIG"

echo "[2/5] Transform"
python -m t5_wiki.src.transform --config "$CONFIG"

echo "[3/5] Train"
	echo "[3/5] Train (PyTorch)"
	python -m t5_wiki.src.train_pt --config "$CONFIG"

echo "[4/5] Evaluate"
	echo "[4/5] Evaluate (PyTorch)"
	python -m t5_wiki.src.advanced_eval_pt --config "$CONFIG" || true

echo "[5/5] Export"
	echo "[5/5] Export (PyTorch)"
	python -m t5_wiki.src.export_pt --config "$CONFIG"
