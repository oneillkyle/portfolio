# PyTorch T5 Pipeline

Pure PyTorch implementation using Hugging Face Transformers and Datasets with streaming support for large-scale training:

## Pipeline Components

- `src/ingest.py` — Data ingestion and setup
- `src/train_pt.py` — PyTorch training (span corruption, streaming datasets)
- `src/advanced_eval_pt.py` — Evaluation (ROUGE scores, sample outputs, file logging)
- `src/test_pt.py` — Test on held-out data
- `src/export_pt.py` — Export trained model and tokenizer
- `scripts/make_test_split_pt.py` — Create test split from raw data
- `scripts/tune_pt.py` — Hyperparameter tuning (grid search)
- `scripts/visualize_results.py` — Generate training visualizations

## Quick Start

Run the complete pipeline:
```bash
# Full pipeline (ingest → test split → train → evaluate → test → export)
bash t5_wiki/scripts/run_pipeline.sh t5_wiki/configs/default.yaml

# Or run with Python orchestration
python -m t5_wiki.scripts.run_all --config t5_wiki/configs/default.yaml
```

## Individual Components

Create test split:
```bash
python -m t5_wiki.scripts.make_test_split_pt --config t5_wiki/configs/default.yaml --num_lines 1000
```

Train (with streaming for large datasets):
```bash
python -m t5_wiki.src.train_pt --config t5_wiki/configs/default.yaml
```

Evaluate:
```bash
python -m t5_wiki.src.advanced_eval_pt --config t5_wiki/configs/default.yaml
```

Test:
```bash
python -m t5_wiki.src.test_pt --config t5_wiki/configs/default.yaml
```

Export:
```bash
python -m t5_wiki.src.export_pt --config t5_wiki/configs/default.yaml
```

Hyperparameter tuning:
```bash
python t5_wiki/scripts/tune_pt.py
```

## Visualization & Monitoring

### Real-time Training Monitoring (TensorBoard)
```bash
# Start TensorBoard (runs in background)
tensorboard --logdir t5_wiki/logs/tensorboard --host 0.0.0.0 --port 6006

# View at: http://localhost:6006
# Shows: training loss, learning rate, GPU utilization, system metrics
```

### Advanced Experiment Tracking (Weights & Biases)
```bash
# One-time setup
pip install wandb
wandb login

# Enable in config
echo "use_wandb: true" >> t5_wiki/configs/default.yaml

# Training will auto-sync to wandb.ai with:
# - Hyperparameters, metrics, system stats
# - Model architecture, gradients
# - Sample outputs and comparisons
```

### Comprehensive Visualization Dashboard
```bash
# Generate plots and HTML dashboard
python t5_wiki/scripts/visualize_results.py

# Creates:
# - t5_wiki/logs/training_metrics.png (loss curves, metrics)
# - t5_wiki/logs/rouge_scores.png (evaluation metrics over time)
# - t5_wiki/logs/training_report.html (comprehensive dashboard)
# - t5_wiki/logs/tuning_results.png (hyperparameter comparison)
```

### Quick Results Check
```bash
# View latest evaluation results (includes sample outputs)
cat t5_wiki/logs/advanced_eval_results.txt

# View training progress
tail -f t5_wiki/logs/tensorboard/events.out.tfevents.*
```

## Key Features

- **Streaming Dataset Loading**: Handles large datasets without memory issues
- **Advanced Evaluation**: ROUGE scores + sample text generation inspection
- **Comprehensive Logging**: Results saved to files + console output
- **Multiple Visualization Options**: TensorBoard, W&B, custom plots
- **Memory Efficient**: Uses Hugging Face Datasets streaming mode
- **GPU Optimized**: Automatic device detection and mixed precision support

## Prerequisites

- Python 3.10+
- GPU recommended for training
- Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r t5_wiki/requirements.txt
```

## Project Structure

```
t5_wiki/
├── data/
│   ├── raw/               # Raw source files (from ingest)
│   └── processed/         # Test split (test.txt)
├── src/
│   ├── ingest.py         # Data ingestion
│   ├── train_pt.py       # PyTorch training with streaming
│   ├── advanced_eval_pt.py # Evaluation with ROUGE & samples
│   ├── test_pt.py        # Test on held-out data
│   ├── export_pt.py      # Model export
│   └── utils.py          # Shared utilities
├── scripts/
│   ├── run_pipeline.sh   # Complete pipeline script
│   ├── run_all.py        # Python pipeline orchestration
│   ├── make_test_split_pt.py # Create test split
│   ├── tune_pt.py        # Hyperparameter tuning
│   └── visualize_results.py # Training visualizations
├── configs/
│   └── default.yaml      # Configuration file
└── logs/                 # Training outputs & checkpoints
```

## Configuration

Key settings in `t5_wiki/configs/default.yaml`:

### Data & Model
```yaml
raw_data_path: ai/datasets/wiki_corpus.subsample.txt  # Input text file
model_name: t5-small                                  # HF model to fine-tune
block_size: 256                                       # Sequence length
val_size: 2000                                        # Validation split size
```

### Training
```yaml
batch_size: 8           # Per-device batch size
learning_rate: 3e-4     # Learning rate
mixed_precision: true   # Enable FP16 for faster training
max_steps: 10000        # Training steps (for streaming datasets)
```

### Visualization & Tracking
```yaml
use_wandb: true                    # Enable W&B tracking
run_name: t5-wiki-pytorch         # Experiment name
log_dir: t5_wiki/logs             # Local log directory
```

## Memory Management

For large datasets, the pipeline uses **streaming mode** to avoid loading all data into RAM:

- Datasets are loaded incrementally during training
- Only `val_size` examples are kept in memory for validation
- Supports datasets of any size without OOM errors

## Output Files & Logs

### Training Outputs
- **Model**: `t5_wiki/logs/` (trained model + tokenizer)
- **TensorBoard**: `t5_wiki/logs/tensorboard/` (training metrics)
- **Checkpoints**: Automatic saving every epoch

### Evaluation Outputs
- **Advanced Results**: `t5_wiki/logs/advanced_eval_results.txt`
  - Sample input/output pairs
  - ROUGE scores
  - Evaluation metrics
- **Visualizations**: 
  - `training_metrics.png` (loss curves)
  - `rouge_scores.png` (evaluation over time)
  - `training_report.html` (comprehensive dashboard)

### Hyperparameter Tuning
- **Results**: `t5_wiki/logs/tuning_pt/tuning_results.csv`
- **Configs**: `t5_wiki/logs/tuning_pt/*.yaml` (per experiment)
- **Plots**: `tuning_results.png` (performance comparison)

## Pipeline Flow

1. **Ingest**: Copy raw data to pipeline directory
2. **Test Split**: Create held-out test set from raw data  
3. **Train**: PyTorch training with streaming datasets and span corruption
4. **Evaluate**: Generate sample outputs and compute ROUGE scores
5. **Test**: Evaluate on held-out test data
6. **Export**: Save trained model and tokenizer

## Notes

- **Streaming**: Handles large datasets without loading all data into memory
- **Span Corruption**: T5-style pretraining with masked span prediction
- **No Transform Step**: Preprocessing happens during training for efficiency
- **Memory Efficient**: Uses Hugging Face Datasets streaming mode