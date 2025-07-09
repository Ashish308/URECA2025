import os
import subprocess
import argparse
import torch
from datetime import datetime
import itertools

def get_device():
    """Auto-select the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def run_experiment(config, experiment_id, device_type):
    """Run a single training experiment with given hyperparameters"""
    temp_output_path = f"./saved_models/temp_model_{experiment_id}.pth"
    os.makedirs("./saved_models", exist_ok=True)

    cmd = [
        "python", "train_local.py",
        f"--train_features={config['train_features']}",
        f"--val_features={config['val_features']}",
        f"--train_edges={config['train_edges']}",
        f"--val_edges={config['val_edges']}",
        f"--train_labels={config['train_labels']}",
        f"--val_labels={config['val_labels']}",
        f"--weights={config['weights']}",
        f"--epochs={config['epochs']}",
        f"--lr={config['lr']}",
        f"--hidden_dim={config['hidden_dim']}",
        f"--num_heads={config['num_heads']}",
        f"--output_path={temp_output_path}",
        f"--device={device_type}"
    ]

    print(f"\n Starting experiment {experiment_id} with config:")
    for name, value in config.items():
        print(f"  {name}: {value}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        # Parse best validation accuracy
        best_val_acc = None
        for line in output.split('\n'):
            if "Best validation accuracy:" in line:
                best_val_acc = float(line.split(":")[-1].strip())
                break

        return {
            'experiment_id': experiment_id,
            'best_val_acc': best_val_acc,
            'model_path': temp_output_path,
            'status': 'success'
        }

    except subprocess.CalledProcessError as e:
        print(f" Experiment {experiment_id} failed:")
        print(e.stderr)
        return {
            'experiment_id': experiment_id,
            'status': 'failed',
            'error': str(e)
        }

def grid_search(param_grid, base_config):
    """Perform grid search over hyperparameters"""
    device_type = get_device()
    print(f" Detected device: {device_type.upper()}")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    best_val_acc = 0.0
    best_model_path = None
    final_model_path = "./saved_models/best_model_val_accuracy.pth"

    for i, values in enumerate(itertools.product(*param_values)):
        config = base_config.copy()
        for name, value in zip(param_names, values):
            config[name] = value

        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        result = run_experiment(config, experiment_id, device_type)

        # Check if this experiment has the best validation accuracy
        if result['status'] == 'success' and result.get('best_val_acc') is not None:
            if result['best_val_acc'] > best_val_acc:
                best_val_acc = result['best_val_acc']
                # Move the temporary model file to the final path
                import shutil
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)  # Remove previous best model
                shutil.move(result['model_path'], final_model_path)
                best_model_path = final_model_path
                print(f"New best model saved with validation accuracy: {best_val_acc:.4f} at {best_model_path}")
            else:
                # Remove temporary model file if not the best
                if os.path.exists(result['model_path']):
                    os.remove(result['model_path'])

    if best_model_path:
        print(f"\nBest model saved at {best_model_path} with validation accuracy: {best_val_acc:.4f}")
    else:
        print("\nNo successful experiments completed; no model saved.")
    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for GAT')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with .npy files')
    args = parser.parse_args()

    base_config = {
        'train_features': os.path.join(args.data_dir, 'train_features.npy'),
        'val_features': os.path.join(args.data_dir, 'val_features.npy'),
        'train_edges': os.path.join(args.data_dir, 'train_edges.npy'),
        'val_edges': os.path.join(args.data_dir, 'val_edges.npy'),
        'train_labels': os.path.join(args.data_dir, 'train_labels.npy'),
        'val_labels': os.path.join(args.data_dir, 'val_labels.npy'),
        'weights': os.path.join(args.data_dir, 'sample_weights.npy'),
        'epochs': 100
    }

    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4],
        'hidden_dim': [64, 128, 256],
        'num_heads': [4, 8, 16]
    }

    print(" Starting hyperparameter tuning...")
    best_val_acc = grid_search(param_grid, base_config)
    print(f" Tuning complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()