#!/usr/bin/env python
"""Compare multiple trained models"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    results = {}
    
    # Load config
    config_path = exp_dir / 'config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            results['config'] = yaml.safe_load(f)
    
    # Load metrics
    for split in ['train', 'val', 'test']:
        metrics_file = exp_dir / f"{split}_results_model_best.pt.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results[f'{split}_metrics'] = json.load(f)
    
    # Load training history
    train_metrics_file = exp_dir / 'train_metrics.json'
    if train_metrics_file.exists():
        with open(train_metrics_file, 'r') as f:
            results['train_history'] = [json.loads(line) for line in f]
    
    val_metrics_file = exp_dir / 'val_metrics.json'
    if val_metrics_file.exists():
        with open(val_metrics_file, 'r') as f:
            results['val_history'] = [json.loads(line) for line in f]
    
    return results

def create_comparison_table(experiments_data):
    """Create a comparison table of experiments"""
    rows = []
    
    for exp_name, data in experiments_data.items():
        row = {
            'Experiment': exp_name,
            'Model': data['config'].get('model_type', 'unknown'),
            'Learning Rate': data['config'].get('learning_rate', 'N/A'),
            'Batch Size': data['config'].get('batch_size', 'N/A'),
            'Dropout': data['config'].get('dropout', 'N/A'),
        }
        
        # Add metrics
        for split in ['val', 'test']:
            if f'{split}_metrics' in data:
                metrics = data[f'{split}_metrics']
                row[f'{split.capitalize()} AUPRC'] = f"{metrics['auprc']:.4f}"
                row[f'{split.capitalize()} Acc'] = f"{metrics['avg_accuracy']:.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_training_curves(experiments_data, output_dir):
    """Plot training curves for all experiments"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for exp_name, data in experiments_data.items():
        if 'val_history' in data:
            history = pd.DataFrame(data['val_history'])
            if 'auprc' in history.columns:
                axes[0].plot(history.index, history['auprc'], label=exp_name)
            if 'avg_loss' in history.columns:
                axes[1].plot(history.index, history['avg_loss'], label=exp_name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AUPRC')
    axes[0].set_title('Validation AUPRC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                        help='List of experiment directories to compare')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiment results
    experiments_data = {}
    for exp_path in args.experiments:
        exp_dir = Path(exp_path)
        if exp_dir.exists():
            exp_name = exp_dir.name
            print(f"Loading experiment: {exp_name}")
            experiments_data[exp_name] = load_experiment_results(exp_dir)
        else:
            print(f"Warning: Experiment directory not found: {exp_path}")
