#!/usr/bin/env python
"""Unified training script for FNN and GAT models"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from config.gat_config import GATConfig
from config.fnn_config import FNNConfig
from trainers.gat_trainer import GATTrainer
from trainers.fnn_trainer import FNNTrainer
from utils.seed import set_seed
from utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description='Train FNN or GAT model')
    parser.add_argument('--model', type=str, choices=['fnn', 'gat'], required=True,
                        help='Model type to train')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment')
    
    # Override config parameters
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Model-specific arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', help='Hidden dimensions for FNN')
    parser.add_argument('--hidden_channels', type=int, help='Hidden channels for GAT')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads for GAT')
    parser.add_argument('--num_layers', type=int, help='Number of GAT layers')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.model == 'fnn':
        config = FNNConfig.load(args.config) if args.config else FNNConfig()
        trainer_class = FNNTrainer
    else:
        config = GATConfig.load(args.config) if args.config else GATConfig()
        trainer_class = GATTrainer
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.dropout:
        config.dropout = args.dropout
    if args.seed:
        config.seed = args.seed
    
    # Model-specific overrides
    if args.model == 'fnn' and args.hidden_dims:
        config.hidden_dims = args.hidden_dims
    elif args.model == 'gat':
        if args.hidden_channels:
            config.hidden_channels = args.hidden_channels
        if args.num_heads:
            config.num_heads = args.num_heads
        if args.num_layers:
            config.num_layers = args.num_layers
    
    config.experiment_name = args.experiment_name
    
    # Set device
    if config.device == "auto":
        config.device = get_device()
    
    # Set random seed
    set_seed(config.seed)
    
    # Create experiment directory
    exp_dir = config.save_dir / args.model / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(exp_dir / 'config.yaml')
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Model: {args.model}")
    print(f"Device: {config.device}")
    print(f"Experiment directory: {exp_dir}")
    
    # Initialize trainer and train
    trainer = trainer_class(config, exp_dir)
    trainer.train()
    
    print(f"Training completed. Results saved to: {exp_dir}")

if __name__ == '__main__':
    main()
