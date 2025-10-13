#!/usr/bin/env python3
"""
Model Pipeline Generator

Creates new model training pipelines based on the t5_wiki template.
Supports different model architectures, datasets, and training configurations.

Usage:
    python scripts/generate_pipeline.py --name my_model --model-type t5 --dataset my_data.txt
    python scripts/generate_pipeline.py --interactive
"""

import os
import sys
import argparse
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List
import re

# Model type configurations
MODEL_CONFIGS = {
    "t5": {
        "model_name": "t5-small",
        "task_type": "seq2seq",
        "preprocessing": "span_corruption",
        "metrics": ["rouge", "bleu"],
        "description": "T5 (Text-to-Text Transfer Transformer) for sequence-to-sequence tasks"
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "task_type": "masked_lm",
        "preprocessing": "masked_language_modeling",
        "metrics": ["perplexity", "accuracy"],
        "description": "BERT for masked language modeling and text understanding"
    },
    "gpt": {
        "model_name": "gpt2",
        "task_type": "causal_lm",
        "preprocessing": "causal_language_modeling",
        "metrics": ["perplexity", "bleu"],
        "description": "GPT for causal language modeling and text generation"
    },
    "roberta": {
        "model_name": "roberta-base",
        "task_type": "masked_lm",
        "preprocessing": "masked_language_modeling",
        "metrics": ["perplexity", "accuracy"],
        "description": "RoBERTa for robust masked language modeling"
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "task_type": "masked_lm",
        "preprocessing": "masked_language_modeling",
        "metrics": ["perplexity", "accuracy"],
        "description": "DistilBERT - lighter, faster BERT variant"
    }
}

class PipelineGenerator:
    def __init__(self, template_dir: str = "t5_wiki"):
        self.template_dir = Path(template_dir)
        self.project_root = Path.cwd()
        
    def generate_pipeline(self, config: Dict[str, Any]) -> Path:
        """Generate a new model pipeline from template."""
        pipeline_name = config["name"]
        pipeline_dir = self.project_root / pipeline_name
        
        print(f"🚀 Generating pipeline: {pipeline_name}")
        print(f"📁 Directory: {pipeline_dir}")
        print(f"🤖 Model type: {config['model_type']}")
        print(f"📊 Dataset: {config['dataset']}")
        
        # Create directory structure
        self._create_directory_structure(pipeline_dir)
        
        # Copy and customize source files
        self._copy_source_files(pipeline_dir, config)
        
        # Generate configuration
        self._generate_config(pipeline_dir, config)
        
        # Generate scripts
        self._generate_scripts(pipeline_dir, config)
        
        # Generate README
        self._generate_readme(pipeline_dir, config)
        
        print(f"✅ Pipeline generated successfully at: {pipeline_dir}")
        return pipeline_dir
    
    def _create_directory_structure(self, pipeline_dir: Path):
        """Create the directory structure for the new pipeline."""
        directories = [
            "src",
            "scripts", 
            "configs",
            "data/raw",
            "data/processed",
            "logs",
            "logs/tensorboard",
            "logs/checkpoints"
        ]
        
        for dir_path in directories:
            (pipeline_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        (pipeline_dir / "__init__.py").touch()
        (pipeline_dir / "src" / "__init__.py").touch()
    
    def _copy_source_files(self, pipeline_dir: Path, config: Dict[str, Any]):
        """Copy and customize source files from template."""
        src_dir = pipeline_dir / "src"
        template_src = self.template_dir / "src"
        
        # Files to copy and customize  
        source_files = [
            "train_pt.py",
            "evaluate_pt.py", 
            "advanced_eval_pt.py",
            "test_pt.py",
            "export_pt.py",
            "ingest.py",
            "utils.py"
        ]
        
        for file_name in source_files:
            if (template_src / file_name).exists():
                self._copy_and_customize_file(
                    template_src / file_name,
                    src_dir / file_name,
                    config
                )
    
    def _copy_and_customize_file(self, src_path: Path, dst_path: Path, config: Dict[str, Any]):
        """Copy a file and customize it for the new pipeline."""
        with open(src_path, 'r') as f:
            content = f.read()
        
        # Replace template placeholders
        replacements = {
            "t5_wiki": config["name"],
            "from .utils": f"from .utils",
            "from t5_wiki.src.utils": f"from {config['name']}.src.utils",
        }
        
        # Apply model-specific customizations
        if config["model_type"] != "t5":
            content = self._customize_for_model_type(content, config)
        
        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        with open(dst_path, 'w') as f:
            f.write(content)
    
    def _customize_for_model_type(self, content: str, config: Dict[str, Any]) -> str:
        """Customize code for different model types."""
        model_type = config["model_type"]
        model_config = MODEL_CONFIGS[model_type]
        
        if model_type in ["bert", "roberta", "distilbert"]:
            # Replace T5 specific code with BERT-style code
            content = content.replace(
                "AutoModelForSeq2SeqLM", 
                "AutoModelForMaskedLM"
            )
            content = content.replace(
                "span_corrupt_example",
                "mask_tokens_example"
            )
        
        elif model_type == "gpt":
            # Replace with GPT-style code
            content = content.replace(
                "AutoModelForSeq2SeqLM",
                "AutoModelForCausalLM"
            )
            content = content.replace(
                "span_corrupt_example",
                "causal_lm_example"
            )
        
        return content
    
    def _generate_config(self, pipeline_dir: Path, config: Dict[str, Any]):
        """Generate configuration file for the new pipeline."""
        model_config = MODEL_CONFIGS[config["model_type"]]
        
        config_data = {
            "experiment_name": f"{config['name']}_pretrain",
            "random_seed": 42,
            
            # Data
            "raw_data_path": config["dataset"],
            "output_dir": config["name"],
            "processed_dir": f"{config['name']}/data/processed",
            "raw_dir": f"{config['name']}/data/raw",
            
            # Model
            "model_name": model_config["model_name"],
            "max_length": 512 if model_config["task_type"] == "masked_lm" else 256,
            "block_size": 512 if model_config["task_type"] == "masked_lm" else 256,
            
            # Training
            "batch_size": 8,
            "epochs": 1,
            "learning_rate": 3e-4,
            "mixed_precision": True,
            "val_size": 2000,
            
            # Strategy
            "strategy": "auto",
            
            # Logging
            "log_dir": f"{config['name']}/logs",
            "ckpt_dir": f"{config['name']}/logs/checkpoints", 
            "save_interval": 1000,
            
            # Visualization
            "use_wandb": True,
            "run_name": f"{config['name']}-{model_config['task_type']}"
        }
        
        # Add model-specific settings
        if model_config["task_type"] == "seq2seq":
            config_data.update({
                "noise_density": 0.15,
                "mean_span_length": 3
            })
        elif model_config["task_type"] == "masked_lm":
            config_data.update({
                "mlm_probability": 0.15
            })
        
        config_path = pipeline_dir / "configs" / "default.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    def _generate_scripts(self, pipeline_dir: Path, config: Dict[str, Any]):
        """Generate utility scripts for the new pipeline."""
        scripts_dir = pipeline_dir / "scripts"
        
        # Generate run_pipeline.sh
        self._generate_run_script(scripts_dir, config)
        
        # Generate run_all.py
        self._generate_run_all_script(scripts_dir, config)
        
        # Generate tuning script
        self._generate_tune_script(scripts_dir, config)
        
        # Copy visualization script
        if (self.template_dir / "scripts" / "visualize_results.py").exists():
            shutil.copy(
                self.template_dir / "scripts" / "visualize_results.py",
                scripts_dir / "visualize_results.py"
            )
        
        # Generate test split script
        self._generate_test_split_script(scripts_dir, config)
    
    def _generate_run_script(self, scripts_dir: Path, config: Dict[str, Any]):
        """Generate run_pipeline.sh script."""
        script_content = f'''#!/usr/bin/env bash
set -euo pipefail

CONFIG=${{1:-configs/default.yaml}}

echo "[1/4] Ingest"
python -m {config["name"]}.src.ingest --config "$CONFIG"

echo "[2/4] Create Test Split"
python -m {config["name"]}.scripts.make_test_split_pt --config "$CONFIG" --num_lines 1000

echo "[3/4] Train"
python -m {config["name"]}.src.train_pt --config "$CONFIG"

echo "[4/4] Evaluate & Test"
echo "  Evaluating..."
python -m {config["name"]}.src.advanced_eval_pt --config "$CONFIG" || true
echo "  Testing..."
python -m {config["name"]}.src.test_pt --config "$CONFIG" || true

echo "[Export] Saving model..."
python -m {config["name"]}.src.export_pt --config "$CONFIG"

echo "Pipeline complete! 🎉"
'''
        
        script_path = scripts_dir / "run_pipeline.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
    
    def _generate_run_all_script(self, scripts_dir: Path, config: Dict[str, Any]):
        """Generate run_all.py orchestration script."""
        script_content = f'''#!/usr/bin/env python3
"""
Run the complete {config["name"]} pipeline.
"""
import subprocess
import sys
from pathlib import Path
import argparse

def run(cmd):
    print(f"🚀 Running: {{cmd}}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {{cmd}}")
        print(f"Error: {{result.stderr}}")
        sys.exit(1)
    print(f"✅ Completed: {{cmd}}")

def main():
    parser = argparse.ArgumentParser(description="Run {config['name']} pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    steps = [
        f"python3 -m {config['name']}.src.ingest --config {{args.config}}",
        f"python3 -m {config['name']}.scripts.make_test_split_pt --config {{args.config}} --num_lines 1000",
        f"python3 -m {config['name']}.src.train_pt --config {{args.config}}",
        f"python3 -m {config['name']}.src.advanced_eval_pt --config {{args.config}}",
        f"python3 -m {config['name']}.src.test_pt --config {{args.config}}",
        f"python3 -m {config['name']}.src.export_pt --config {{args.config}}"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\\n[{{i}}/{{len(steps)}}] {{step.split()[2].split('.')[-1].upper()}}")
        try:
            run(step)
        except SystemExit:
            if i <= 3:  # Critical steps: ingest, test split, train
                raise
            else:  # Optional steps: evaluate, test, export
                print(f"⚠️  Step {{i}} failed but continuing...")
    
    print("\\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        script_path = scripts_dir / "run_all.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
    
    def _generate_tune_script(self, scripts_dir: Path, config: Dict[str, Any]):
        """Generate hyperparameter tuning script.""" 
        model_config = MODEL_CONFIGS[config["model_type"]]
        
        # Different param grids for different model types
        if model_config["task_type"] == "seq2seq":
            param_grid = '''param_grid = {
    "learning_rate": [3e-4, 1e-4],
    "batch_size": [8, 16],
    "noise_density": [0.15, 0.25],
    "mean_span_length": [3, 5],
}'''
        elif model_config["task_type"] == "masked_lm":
            param_grid = '''param_grid = {
    "learning_rate": [3e-4, 1e-4, 5e-5],
    "batch_size": [8, 16, 32],
    "mlm_probability": [0.15, 0.20],
}'''
        else:  # causal_lm
            param_grid = '''param_grid = {
    "learning_rate": [3e-4, 1e-4, 5e-5],
    "batch_size": [4, 8, 16],
    "block_size": [256, 512],
}'''
        
        script_content = f'''#!/usr/bin/env python3
"""
PyTorch-based hyperparameter tuning for {config["name"]}.
"""
import os
import copy
import yaml
from itertools import product
import subprocess

CONFIG_PATH = "{config['name']}/configs/default.yaml"
LOG_DIR = "{config['name']}/logs/tuning_pt"

# Define hyperparameter grid
{param_grid}

def run_train(config):
    params = "_".join([f"{{k}}{{v}}" for k, v in config.items() if k in param_grid.keys()])
    run_name = f"{{params}}"
    tmp_cfg = os.path.join(LOG_DIR, f"{{run_name}}.yaml")
    
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(config, f)
    
    # Run PyTorch training
    subprocess.run([
        "python3", "-m", "{config['name']}.src.train_pt", "--config", tmp_cfg
    ], check=True)
    
    # Evaluate
    eval_out = subprocess.run([
        "python3", "-m", "{config['name']}.src.advanced_eval_pt", "--config", tmp_cfg
    ], capture_output=True, text=True)
    
    # Log results
    with open(os.path.join(LOG_DIR, "tuning_results.csv"), "a") as rf:
        rf.write(f"{{run_name}},{{eval_out.stdout.strip()}}\\n")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    
    keys, values = zip(*param_grid.items())
    
    # Write header
    with open(os.path.join(LOG_DIR, "tuning_results.csv"), "w") as rf:
        rf.write("run_name,metrics\\n")
    
    for combo in product(*values):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            cfg[k] = v
        run_train(cfg)

if __name__ == "__main__":
    main()
'''
        
        script_path = scripts_dir / "tune_pt.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
    
    def _generate_test_split_script(self, scripts_dir: Path, config: Dict[str, Any]):
        """Generate test split creation script."""
        script_content = f'''#!/usr/bin/env python3
"""
Create a test split from raw data for PyTorch pipeline.
"""
import os
import argparse
from {config["name"]}.src.utils import load_config, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='{config["name"]}/configs/default.yaml')
    parser.add_argument('--num_lines', type=int, default=1000, help='Number of lines to use for test set')
    parser.add_argument('--start_line', type=int, default=0, help='Starting line for test set')
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
    processed_dir = cfg["processed_dir"]
    ensure_dir(processed_dir)

    print(f"Creating test split from {{raw_path}}")
    print(f"Lines: {{args.start_line}} to {{args.start_line + args.num_lines}}")
    
    # Read specific lines for test set
    test_lines = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.start_line and i < args.start_line + args.num_lines:
                test_lines.append(line.strip())
            elif i >= args.start_line + args.num_lines:
                break
    
    # Write test.txt
    test_path = os.path.join(processed_dir, "test.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        for line in test_lines:
            if line:  # Skip empty lines
                f.write(line + "\\n")
    
    print(f"✅ Created test split: {{test_path}}")
    print(f"📊 Test samples: {{len(test_lines)}}")

if __name__ == "__main__":
    main()
'''
        
        script_path = scripts_dir / "make_test_split_pt.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
    
    def _generate_readme(self, pipeline_dir: Path, config: Dict[str, Any]):
        """Generate README for the new pipeline."""
        model_config = MODEL_CONFIGS[config["model_type"]]
        
        readme_content = f'''# {config["name"].title()} Pipeline

{model_config["description"]}

**Model**: `{model_config["model_name"]}`  
**Task**: {model_config["task_type"]}  
**Dataset**: `{config["dataset"]}`

## Quick Start

```bash
# Run complete pipeline
bash scripts/run_pipeline.sh configs/default.yaml

# Or run step by step
python -m {config["name"]}.scripts.run_all --config configs/default.yaml
```

## Individual Components

```bash
# Train
python -m {config["name"]}.src.train_pt --config configs/default.yaml

# Evaluate (with sample outputs and {", ".join(model_config["metrics"])})
python -m {config["name"]}.src.advanced_eval_pt --config configs/default.yaml

# Export trained model
python -m {config["name"]}.src.export_pt --config configs/default.yaml

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

- **Model**: Change `model_name` to try different {model_config["task_type"]} models
- **Data**: Update `raw_data_path` to use your dataset
- **Training**: Adjust `batch_size`, `learning_rate`, `epochs`
- **Tracking**: Enable W&B with `use_wandb: true`

## Output Files

- **Trained Model**: `logs/` (model + tokenizer)
- **Metrics**: `logs/tensorboard/` (TensorBoard logs)  
- **Evaluation**: `logs/advanced_eval_results.txt` ({", ".join(model_config["metrics"])} scores + samples)
- **Visualizations**: `logs/*.png` (training plots)

---

Generated with the Model Pipeline Generator 🚀
'''
        
        readme_path = pipeline_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

def interactive_mode():
    """Interactive CLI for pipeline generation."""
    print("🚀 Model Pipeline Generator")
    print("=" * 50)
    
    # Show available model types
    print("\\nAvailable model types:")
    for i, (key, value) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"  {i}. {key.upper()} - {value['description']}")
    
    # Get user input
    while True:
        try:
            choice = int(input("\\nSelect model type (1-5): ")) - 1
            model_types = list(MODEL_CONFIGS.keys())
            if 0 <= choice < len(model_types):
                model_type = model_types[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    name = input("\\nPipeline name: ").strip()
    if not name:
        print("Pipeline name is required!")
        return None
        
    dataset = input("Dataset path (relative to project root): ").strip()
    if not dataset:
        dataset = "data/my_dataset.txt"
        print(f"Using default: {dataset}")
    
    config = {
        "name": name,
        "model_type": model_type,
        "dataset": dataset
    }
    
    print(f"\\n📋 Configuration Summary:")
    print(f"  Name: {config['name']}")
    print(f"  Model: {MODEL_CONFIGS[model_type]['model_name']} ({model_type})")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Task: {MODEL_CONFIGS[model_type]['task_type']}")
    
    confirm = input("\\nProceed? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        return config
    else:
        print("Cancelled.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate model training pipeline")
    parser.add_argument("--name", help="Pipeline name")
    parser.add_argument("--model-type", choices=list(MODEL_CONFIGS.keys()), help="Model type")
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--template-dir", default="t5_wiki", help="Template directory")
    
    args = parser.parse_args()
    
    if args.interactive:
        config = interactive_mode()
        if not config:
            return
    else:
        if not all([args.name, args.model_type, args.dataset]):
            print("Error: --name, --model-type, and --dataset are required (or use --interactive)")
            parser.print_help()
            return
        
        config = {
            "name": args.name,
            "model_type": args.model_type,
            "dataset": args.dataset
        }
    
    # Generate pipeline
    generator = PipelineGenerator(args.template_dir)
    pipeline_dir = generator.generate_pipeline(config)
    
    print(f"\\n🎉 Success! Your new pipeline is ready at: {pipeline_dir}")
    print(f"\\n📚 Next steps:")
    print(f"  1. cd {pipeline_dir}")
    print(f"  2. Place your dataset at: {config['dataset']}")
    print(f"  3. Run: bash scripts/run_pipeline.sh")
    print(f"\\n📖 See README.md for detailed usage instructions.")

if __name__ == "__main__":
    main()