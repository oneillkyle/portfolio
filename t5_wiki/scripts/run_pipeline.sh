#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/5] Ingest"
python -m t5_wiki.src.ingest --config "$CONFIG"

echo "[2/5] Transform"
python -m t5_wiki.src.transform --config "$CONFIG"

echo "[2.5/6] Create Test Split"
python -m t5_wiki.scripts.make_test_split_pt --config "$CONFIG" --num_lines 1000

echo "[3/6] Train"
	echo "[3/5] Train (PyTorch)"
	python -m t5_wiki.src.train_pt --config "$CONFIG"

echo "[4/6] Evaluate"
	echo "[4/6] Evaluate (PyTorch)"
	python -m t5_wiki.src.advanced_eval_pt --config "$CONFIG" || true

echo "[5/6] Test"
	echo "[5/6] Test (PyTorch)"
	python -m t5_wiki.src.test_pt --config "$CONFIG" || true

echo "[6/6] Export"
	echo "[6/6] Export (PyTorch)"
	python -m t5_wiki.src.export_pt --config "$CONFIG"
