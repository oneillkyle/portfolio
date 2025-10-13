#!/usr/bin/env python3
"""
Simple hyperparameter tuning script for t5_wiki pipeline.
Runs multiple training jobs with different configs and logs results.
"""
import os
import copy
import subprocess
import yaml
from itertools import product

CONFIG_PATH = "t5_wiki/configs/default.yaml"
LOG_DIR = "t5_wiki/logs/tuning"

# Define hyperparameter grid
param_grid = {
    "learning_rate": [3e-4, 1e-4],
    "batch_size": [8, 16],
    "noise_density": [0.15, 0.25],
    "mean_span_length": [3, 5],
}


def run_train_and_eval(config, results_file):
    import json
    run_name = f"lr{config['learning_rate']}_bs{config['batch_size']}_nd{config['noise_density']}_msl{config['mean_span_length']}"
    tmp_cfg = os.path.join(LOG_DIR, f"{run_name}.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(config, f)
    # Train
    subprocess.run([
        "python3", "-m", "t5_wiki.src.train", "--config", tmp_cfg
    ], check=True)
    # Evaluate (validation)
    eval_out = subprocess.run([
        "python3", "-m", "t5_wiki.src.evaluate", "--config", tmp_cfg
    ], capture_output=True, text=True)
    # Parse and log results
    try:
        metrics = json.loads(eval_out.stdout.strip().split("\n")[-1].replace("'", '"'))
    except Exception:
        metrics = {"loss": None, "perplexity": None}
    with open(results_file, "a") as rf:
        row = {**config, **metrics}
        rf.write(",".join(str(row[k]) for k in row.keys()) + "\n")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    keys, values = zip(*param_grid.items())
    results_file = os.path.join(LOG_DIR, "tuning_results.csv")
    # Write header
    with open(results_file, "w") as rf:
        header = list(keys) + ["loss", "perplexity"]
        rf.write(",".join(header) + "\n")
    for combo in product(*values):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            cfg[k] = v
        run_train_and_eval(cfg, results_file)


if __name__ == "__main__":
    main()
