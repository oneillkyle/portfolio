#!/usr/bin/env python3
"""
Visualization dashboard for T5 training results.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

def plot_training_metrics(log_dir="t5_wiki/logs"):
    """Plot training metrics from TensorBoard logs."""
    # Look for training logs
    tb_logs = glob.glob(f"{log_dir}/tensorboard/events.out.tfevents.*")
    if not tb_logs:
        print("No TensorBoard logs found")
        return
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('T5 Training Metrics', fontsize=16)
        
        for log_file in tb_logs:
            ea = EventAccumulator(log_file)
            ea.Reload()
            
            # Plot training loss
            if 'train_loss' in ea.scalars.Keys():
                train_loss = ea.scalars.Items('train_loss')
                steps = [x.step for x in train_loss]
                values = [x.value for x in train_loss]
                axes[0,0].plot(steps, values, label='Train Loss')
                axes[0,0].set_title('Training Loss')
                axes[0,0].set_xlabel('Steps')
                axes[0,0].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
        print(f"Training metrics saved to {log_dir}/training_metrics.png")
        
    except ImportError:
        print("Install tensorboard for training metric visualization: pip install tensorboard")

def plot_evaluation_results(log_dir="t5_wiki/logs"):
    """Plot evaluation results from advanced_eval_results.txt."""
    eval_file = f"{log_dir}/advanced_eval_results.txt"
    if not os.path.exists(eval_file):
        print(f"No evaluation results found at {eval_file}")
        return
    
    # Parse evaluation results
    rouge_scores = []
    with open(eval_file, 'r') as f:
        content = f.read()
        # Extract ROUGE scores using regex
        rouge_matches = re.findall(r"ROUGE Scores: \{([^}]+)\}", content)
        for match in rouge_matches:
            # Parse the dictionary string
            try:
                rouge_dict = eval("{" + match + "}")
                rouge_scores.append(rouge_dict)
            except:
                continue
    
    if rouge_scores:
        df = pd.DataFrame(rouge_scores)
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar', ax=plt.gca())
        plt.title('ROUGE Scores Over Time')
        plt.ylabel('Score')
        plt.xlabel('Evaluation Run')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{log_dir}/rouge_scores.png", dpi=300, bbox_inches='tight')
        print(f"ROUGE scores saved to {log_dir}/rouge_scores.png")

def plot_hyperparameter_tuning(tuning_dir="t5_wiki/logs/tuning_pt"):
    """Plot hyperparameter tuning results."""
    results_file = f"{tuning_dir}/tuning_results.csv"
    if not os.path.exists(results_file):
        print(f"No tuning results found at {results_file}")
        return
    
    try:
        df = pd.read_csv(results_file)
        
        # Extract hyperparameters from run names
        for idx, row in df.iterrows():
            run_name = row['run_name']
            # Parse lr0.0003_bs16_nd0.15_msl3 format
            parts = run_name.split('_')
            for part in parts:
                if part.startswith('lr'):
                    df.loc[idx, 'learning_rate'] = float(part[2:])
                elif part.startswith('bs'):
                    df.loc[idx, 'batch_size'] = int(part[2:])
                elif part.startswith('nd'):
                    df.loc[idx, 'noise_density'] = float(part[2:])
                elif part.startswith('msl'):
                    df.loc[idx, 'mean_span_length'] = int(part[3:])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16)
        
        # You'll need to extract actual metrics from the metrics column
        # This is a template - adjust based on your actual metrics format
        
        plt.tight_layout()
        plt.savefig(f"{tuning_dir}/tuning_results.png", dpi=300, bbox_inches='tight')
        print(f"Tuning results saved to {tuning_dir}/tuning_results.png")
        
    except Exception as e:
        print(f"Error plotting tuning results: {e}")

def create_summary_report(log_dir="t5_wiki/logs"):
    """Create a comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>T5 Wiki Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>T5 Wiki Training Report</h1>
        <div class="section">
            <h2>Training Metrics</h2>
            <img src="training_metrics.png" alt="Training Metrics">
        </div>
        <div class="section">
            <h2>Evaluation Results</h2>
            <img src="rouge_scores.png" alt="ROUGE Scores">
        </div>
        <div class="section">
            <h2>Sample Outputs</h2>
            <div class="metric">
                <h3>Latest Evaluation Samples:</h3>
                <pre id="samples"></pre>
            </div>
        </div>
        <div class="section">
            <h2>Configuration</h2>
            <div class="metric">
                <pre id="config"></pre>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(f"{log_dir}/training_report.html", 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {log_dir}/training_report.html")
    print("Open it in your browser to view the dashboard")

def main():
    """Generate all visualizations."""
    print("Generating T5 training visualizations...")
    
    plot_training_metrics()
    plot_evaluation_results()
    plot_hyperparameter_tuning()
    create_summary_report()
    
    print("\nVisualization complete! Check the following:")
    print("- TensorBoard: tensorboard --logdir t5_wiki/logs/tensorboard")
    print("- Training plots: t5_wiki/logs/training_metrics.png")
    print("- ROUGE plots: t5_wiki/logs/rouge_scores.png")
    print("- HTML report: t5_wiki/logs/training_report.html")

if __name__ == "__main__":
    main()