#!/usr/bin/env python3
"""
Plot tuning results from tuning_results.csv for the PyTorch pipeline.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = "t5_wiki/logs/tuning_pt/tuning_results.csv"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No tuning results found at {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    # Attempt to extract metrics from string if needed
    if 'metrics' in df.columns:
        import ast
        metrics_df = df['metrics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
        metrics_df = pd.json_normalize(metrics_df)
        df = pd.concat([df.drop(columns=['metrics']), metrics_df], axis=1)
    # Drop rows with missing metrics
    metric_cols = [c for c in df.columns if 'loss' in c or 'perplexity' in c]
    for metric in metric_cols:
        df = df[df[metric].notna()]
    # Plot loss vs. each hyperparameter
    for param in [c for c in df.columns if c not in metric_cols and c != 'run_name']:
        for metric in metric_cols:
            plt.figure()
            sns.scatterplot(x=param, y=metric, data=df)
            plt.title(f"{metric} vs. {param}")
            plt.savefig(f"t5_wiki/logs/tuning_pt/{metric}_vs_{param}.png")
    # Pairplot for all params and metrics
    sns.pairplot(df)
    plt.savefig("t5_wiki/logs/tuning_pt/tuning_pairplot.png")
    print("Plots saved to t5_wiki/logs/tuning_pt/")

if __name__ == "__main__":
    main()
