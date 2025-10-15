#!/usr/bin/env python3
"""
Analyze and rank hyperparameter tuning results from tune_pt.py.
Parses tuning_results.csv and displays the best configurations.
"""
import os
import sys
import pandas as pd
import re
import json
from pathlib import Path

def parse_metrics(metrics_str):
    """Parse metrics from the eval output string."""
    metrics = {}
    
    # Try to extract ROUGE scores
    rouge_match = re.search(r"ROUGE Scores: \{([^}]+)\}", metrics_str)
    if rouge_match:
        try:
            rouge_dict_str = "{" + rouge_match.group(1) + "}"
            rouge_dict = eval(rouge_dict_str)
            metrics.update(rouge_dict)
        except:
            pass
    
    # Try to extract eval_loss if present
    loss_match = re.search(r"'eval_loss':\s*([\d.]+)", metrics_str)
    if loss_match:
        metrics['eval_loss'] = float(loss_match.group(1))
    
    # Try to extract perplexity if present
    ppl_match = re.search(r"'eval_perplexity':\s*([\d.]+)", metrics_str)
    if ppl_match:
        metrics['eval_perplexity'] = float(ppl_match.group(1))
    
    return metrics

def parse_run_name(run_name):
    """Parse hyperparameters from run name."""
    params = {}
    
    # Extract learning rate
    lr_match = re.search(r"lr([\d.e-]+)", run_name)
    if lr_match:
        params['learning_rate'] = float(lr_match.group(1))
    
    # Extract batch size
    bs_match = re.search(r"bs(\d+)", run_name)
    if bs_match:
        params['batch_size'] = int(bs_match.group(1))
    
    # Extract noise density
    nd_match = re.search(r"nd([\d.]+)", run_name)
    if nd_match:
        params['noise_density'] = float(nd_match.group(1))
    
    # Extract mean span length
    msl_match = re.search(r"msl(\d+)", run_name)
    if msl_match:
        params['mean_span_length'] = int(msl_match.group(1))
    
    return params

def analyze_tuning_results(results_file="t5_wiki/logs/tuning_pt/tuning_results.csv"):
    """Analyze tuning results and display ranked configurations."""
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Have you run tune_pt.py yet?")
        return
    
    print("📊 Analyzing Hyperparameter Tuning Results")
    print("=" * 70)
    
    # Read results - handle multiline and comma-heavy metrics
    try:
        # Try reading with error handling for malformed CSV
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        # Parse manually since the metrics contain commas and newlines
        results_data = []
        current_run = None
        current_metrics = []
        
        for line in lines[1:]:  # Skip header
            if line.strip().startswith('lr'):
                # Save previous run if exists
                if current_run:
                    results_data.append({
                        'run_name': current_run,
                        'metrics': '\n'.join(current_metrics)
                    })
                # Start new run
                parts = line.split(',', 1)
                current_run = parts[0].strip()
                current_metrics = [parts[1].strip()] if len(parts) > 1 else []
            else:
                # Continue accumulating metrics
                if line.strip():
                    current_metrics.append(line.strip())
        
        # Save last run
        if current_run:
            results_data.append({
                'run_name': current_run,
                'metrics': '\n'.join(current_metrics)
            })
        
        df = pd.DataFrame(results_data)
        
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return
    
    if df.empty:
        print("❌ No results found in the file.")
        return
    
    # Parse run names and metrics
    results = []
    for idx, row in df.iterrows():
        run_name = row['run_name']
        metrics_str = str(row['metrics'])
        
        params = parse_run_name(run_name)
        metrics = parse_metrics(metrics_str)
        
        result = {
            'run_name': run_name,
            **params,
            **metrics
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Display summary
    print(f"\n📈 Total Runs: {len(results_df)}")
    print(f"📊 Metrics Available: {[col for col in results_df.columns if col not in ['run_name', 'learning_rate', 'batch_size', 'noise_density', 'mean_span_length']]}")
    
    # Rank by different metrics
    print("\n" + "=" * 70)
    print("🏆 TOP 5 CONFIGURATIONS BY METRIC")
    print("=" * 70)
    
    # Rank by eval_loss (lower is better)
    if 'eval_loss' in results_df.columns:
        print("\n📉 Best by Eval Loss (lower is better):")
        print("-" * 70)
        top_loss = results_df.nsmallest(5, 'eval_loss')
        for i, (idx, row) in enumerate(top_loss.iterrows(), 1):
            print(f"  {i}. {row['run_name']}")
            print(f"     Loss: {row.get('eval_loss', 'N/A'):.4f}")
            print(f"     LR: {row.get('learning_rate', 'N/A')}, BS: {row.get('batch_size', 'N/A')}, "
                  f"ND: {row.get('noise_density', 'N/A')}, MSL: {row.get('mean_span_length', 'N/A')}")
            print()
    
    # Rank by ROUGE-1 (higher is better)
    if 'rouge1' in results_df.columns:
        print("\n📈 Best by ROUGE-1 (higher is better):")
        print("-" * 70)
        top_rouge1 = results_df.nlargest(5, 'rouge1')
        for i, (idx, row) in enumerate(top_rouge1.iterrows(), 1):
            print(f"  {i}. {row['run_name']}")
            print(f"     ROUGE-1: {row.get('rouge1', 'N/A'):.4f}")
            print(f"     LR: {row.get('learning_rate', 'N/A')}, BS: {row.get('batch_size', 'N/A')}, "
                  f"ND: {row.get('noise_density', 'N/A')}, MSL: {row.get('mean_span_length', 'N/A')}")
            print()
    
    # Rank by ROUGE-L (higher is better)
    if 'rougeL' in results_df.columns:
        print("\n📈 Best by ROUGE-L (higher is better):")
        print("-" * 70)
        top_rougeL = results_df.nlargest(5, 'rougeL')
        for i, (idx, row) in enumerate(top_rougeL.iterrows(), 1):
            print(f"  {i}. {row['run_name']}")
            print(f"     ROUGE-L: {row.get('rougeL', 'N/A'):.4f}")
            print(f"     LR: {row.get('learning_rate', 'N/A')}, BS: {row.get('batch_size', 'N/A')}, "
                  f"ND: {row.get('noise_density', 'N/A')}, MSL: {row.get('mean_span_length', 'N/A')}")
            print()
    
    # Overall recommendation (lowest loss or highest ROUGE-L)
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDED CONFIGURATION")
    print("=" * 70)
    
    if 'eval_loss' in results_df.columns:
        best_idx = results_df['eval_loss'].idxmin()
        best = results_df.loc[best_idx]
        print(f"\n✨ Best Overall: {best['run_name']}")
        print(f"   (Based on lowest eval_loss)")
        print(f"\n   Hyperparameters:")
        print(f"     - learning_rate: {best.get('learning_rate', 'N/A')}")
        print(f"     - batch_size: {best.get('batch_size', 'N/A')}")
        print(f"     - noise_density: {best.get('noise_density', 'N/A')}")
        print(f"     - mean_span_length: {best.get('mean_span_length', 'N/A')}")
        print(f"\n   Metrics:")
        for col in results_df.columns:
            if col not in ['run_name', 'learning_rate', 'batch_size', 'noise_density', 'mean_span_length']:
                val = best.get(col)
                if pd.notna(val):
                    print(f"     - {col}: {val:.4f}")
    
    # Save detailed results
    output_file = "t5_wiki/logs/tuning_pt/analysis_summary.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Display full table
    print("\n" + "=" * 70)
    print("📋 FULL RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results")
    parser.add_argument('--results', type=str, default='t5_wiki/logs/tuning_pt/tuning_results.csv',
                       help='Path to tuning results CSV')
    args = parser.parse_args()
    
    analyze_tuning_results(args.results)

if __name__ == "__main__":
    main()
