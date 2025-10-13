#!/usr/bin/env python3
"""
PyTorch-based hyperparameter tuning for t5_wiki using Hugging Face Trainer.
"""
import os
import copy
import yaml
from itertools import product
import subprocess

CONFIG_PATH = "t5_wiki/configs/default.yaml"
LOG_DIR = "t5_wiki/logs/tuning_pt"

# Define hyperparameter grid
param_grid = {
    "learning_rate": [3e-4, 1e-4],
    "batch_size": [8, 16],
    "noise_density": [0.15, 0.25],
    "mean_span_length": [3, 5],
}

def run_train(config):
    run_name = f"lr{config['learning_rate']}_bs{config['batch_size']}_nd{config['noise_density']}_msl{config['mean_span_length']}"
    tmp_cfg = os.path.join(LOG_DIR, f"{run_name}.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(config, f)
    # Run PyTorch training
    subprocess.run([
        "python3", "-m", "t5_wiki.src.train_pt", "--config", tmp_cfg
    ], check=True)
    # Evaluate
    eval_out = subprocess.run([
        "python3", "-m", "t5_wiki.src.evaluate_pt", "--config", tmp_cfg
    ], capture_output=True, text=True)
    # Log results
    with open(os.path.join(LOG_DIR, "tuning_results.csv"), "a") as rf:
        rf.write(f"{run_name},{eval_out.stdout.strip()}\n")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    keys, values = zip(*param_grid.items())
    # Write header
    with open(os.path.join(LOG_DIR, "tuning_results.csv"), "w") as rf:
        rf.write("run_name,metrics\n")
    for combo in product(*values):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            cfg[k] = v
        run_train(cfg)

if __name__ == "__main__":
    main()
