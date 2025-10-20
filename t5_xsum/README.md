# T5_Xsum Pipeline

T5 (Text-to-Text Transfer Transformer) for sequence-to-sequence tasks

**Model**: `t5-small`  
**Task**: seq2seq  
**Dataset**: `xsum`

## Quick Start

```bash
# Run complete pipeline
bash scripts/run_pipeline.sh configs/default.yaml

# Or run step by step
python -m t5_xsum.scripts.run_all --config configs/default.yaml
```

## Individual Components

```bash
# Train
python -m t5_xsum.src.train_pt --config configs/default.yaml

# Evaluate (with sample outputs and rouge, bleu)
python -m t5_xsum.src.advanced_eval_pt --config configs/default.yaml

# Export trained model
python -m t5_xsum.src.export_pt --config configs/default.yaml

# Hyperparameter tuning
python scripts/tune_pt.py
```

## Monitoring & Visualization

```bash
# TensorBoard (real-time training metrics)
tensorboard --logdir logs/tensorboard --port 6006

# Generate visualization dashboard
python scripts/visualize_results.py

# View evaluation results
cat logs/advanced_eval_results.txt
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Model**: Change `model_name` to try different seq2seq models
- **Data**: Update `raw_data_path` to use your dataset
- **Training**: Adjust `batch_size`, `learning_rate`, `epochs`
- **Tracking**: Enable W&B with `use_wandb: true`

## Output Files

- **Trained Model**: `logs/` (model + tokenizer)
- **Metrics**: `logs/tensorboard/` (TensorBoard logs)  
- **Evaluation**: `logs/advanced_eval_results.txt` (rouge, bleu scores + samples)
- **Visualizations**: `logs/*.png` (training plots)

---

Generated with the Model Pipeline Generator 🚀
