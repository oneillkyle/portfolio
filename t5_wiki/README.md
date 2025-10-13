# PyTorch T5 Pipeline (Recommended)

The pipeline is now fully supported in PyTorch using Hugging Face Transformers and Datasets. All major steps have a `_pt.py` version:

- `src/train_pt.py` — PyTorch training (span corruption, Trainer API, streaming datasets)
- `src/evaluate_pt.py` — Basic evaluation (perplexity)
- `src/advanced_eval_pt.py` — Advanced evaluation (ROUGE scores, sample outputs)
- `src/test_pt.py` — Evaluate on test set
- `src/export_pt.py` — Export model and tokenizer
- `scripts/tune_pt.py` — Hyperparameter tuning (grid search)
- `scripts/visualize_results.py` — Generate comprehensive visualizations

## Quick Start

Run the complete pipeline:
```bash
# Full pipeline (ingest → transform → train → evaluate → export)
bash t5_wiki/scripts/run_pipeline.sh t5_wiki/configs/default.yaml

# Or run individual components (orchestrated)
python -m t5_wiki.scripts.run_all --config t5_wiki/configs/default.yaml
```

## Individual Components

Train (with streaming for large datasets):
```bash
python -m t5_wiki.src.train_pt --config t5_wiki/configs/default.yaml
```

Evaluate:
```bash
# Basic evaluation (perplexity)
python -m t5_wiki.src.evaluate_pt --config t5_wiki/configs/default.yaml

# Advanced evaluation (ROUGE + sample outputs + file logging)
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

---

# (Legacy) TensorFlow pipeline

End-to-end pipeline to pre-train a T5-style seq2seq model on a raw text corpus using TensorFlow + Hugging Face Transformers.

## layout

t5_wiki/
- data/
	- raw/               # raw source files (symlinked/copied during ingest)
	- processed/         # TFRecord shards written by transform
- src/
	- ingest.py          # bring raw text into data/raw
	- transform.py       # tokenize, chunk, create labels (-100 mask), write TFRecords
	- model.py           # HF TF model wrapped for Keras (returns logits)
	- train.py           # tf.data input, masked loss, TensorBoard, checkpoints
	- evaluate.py        # loss/perplexity on val or train
	- export.py          # export latest trained checkpoint
- configs/
	- default.yaml       # default hyperparameters and paths
- scripts/
	- run_pipeline.sh    # ingest -> transform -> train -> evaluate -> export
	- smoke_test.py      # tiny E2E sanity check
- logs/                # {experiment}/{timestamp}/{tensorboard,checkpoints}

## prerequisites

- Python 3.10+
- GPU optional (recommended)
- Install dependencies:

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r t5_wiki/requirements.txt
```

## configuration

`t5_wiki/configs/default.yaml` keys:

- data
	- `raw_data_path`: path to your raw text (one document per line)
	- `raw_dir`, `processed_dir`, `output_dir`
- model/tokenization
	- `model_name`: e.g. `t5-small`
	- `max_length`: per-line tokenizer max length
	- `block_size`: final sequence length in TFRecords
- training
	- `batch_size`, `epochs`, `learning_rate`, `mixed_precision`
- logging
	- `experiment_name`, `log_dir`

## run the pipeline

Run all steps:

```
bash t5_wiki/scripts/run_pipeline.sh --config t5_wiki/configs/default.yaml
// OR

bash python3 -m t5_wiki.scripts.run_all
```

Or step-by-step:

```
python -m t5_wiki.src.ingest --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.transform --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.train --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.evaluate --config t5_wiki/configs/default.yaml
python -m t5_wiki.src.export --config t5_wiki/configs/default.yaml
```

python3 -m t5_wiki.scripts.make_test_split --num_lines 1000

View TensorBoard:

```
tensorboard --logdir t5_wiki/logs
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

## notes

- Transform creates labels by shifting input blocks and masking pad positions with -100; training uses `sample_weight` to ignore those.
- To mimic T5 span corruption, we can add a span-masking step in `transform.py` and adjust labels accordingly.