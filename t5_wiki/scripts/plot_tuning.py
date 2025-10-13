#!/usr/bin/env python3
"""
Plot tuning results from tuning_results.csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = "t5_wiki/logs/tuning/tuning_results.csv"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No tuning results found at {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    # Drop rows with missing metrics
    df = df.dropna(subset=["loss", "perplexity"])
    # Plot loss vs. each hyperparameter
    for param in [c for c in df.columns if c not in ("loss", "perplexity")]:
        plt.figure()
        sns.scatterplot(x=param, y="loss", data=df)
        plt.title(f"Validation Loss vs. {param}")
        plt.savefig(f"t5_wiki/logs/tuning/loss_vs_{param}.png")
    # Pairplot for all params and metrics
    sns.pairplot(df)
    plt.savefig("t5_wiki/logs/tuning/tuning_pairplot.png")
    print("Plots saved to t5_wiki/logs/tuning/")

if __name__ == "__main__":
    main()
