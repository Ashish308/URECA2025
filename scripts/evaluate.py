#!/usr/bin/env python
"""Evaluate trained models"""

import argparse
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

import torch
from evaluation.metrics import MetricsTracker
from config import FNNConfig, GATConfig
from dataloaders import FNNDataLoader, GATDataLoader
from utils.device import get_device

def load_model_and_config(model_path, config_path):
    """Load model and configuration"""
    # Load config
    config_dict = torch.load(model_path, map_location='cpu')
    
    # Determine model type from config
    if 'model_type' in config_dict['config'].__dict__:
        model_type = config_dict['config'].model_type
    else:
        # Infer from model architecture
        model_type = 'gat' if 'convs' in config_dict['model_state_dict'] else 'fnn'
    
    # Load appropriate config
    if model_type == 'fnn':
        config = FNNConfig.load(config_path)
        from models.fnn_model import FNN_Model
        model = FNN_Model(
            input_size=config.input_size,
            num_classes=config.num_classes,
            dropout_rate=config.dropout,
            hidden_dims=config.hidden_dims
        )
    else:
        config = GATConfig.load(config_path)
        from models.gat_model import GATModel
        model = GATModel(
            in_channels=config.input_size,
            hidden_channels=config.hidden_channels,
            out_channels=config.num_classes,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    
    model.load_state_dict(config_dict['model_state_dict'])
    return model, config, model_type

def evaluate_model(model, dataloader, device, num_classes, model_type):
    """Evaluate model on a dataset"""
    model.eval()
    metrics = MetricsTracker(num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'fnn':
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            else:  # GAT
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index)
                labels = batch.y
            
            metrics.update(outputs, labels)
    
    return metrics.compute()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--checkpoint', type=str, default='model_best.pt',
                        help='Checkpoint name to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    model_path = exp_dir / args.checkpoint
    config_path = exp_dir / 'config.yaml'
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    
    # Load model and config
    model, config, model_type = load_model_and_config(model_path, config_path)
    
    # Set device
    device = get_device() if config.device == 'auto' else config.device
    device = torch.device(device)
    model = model.to(device)
    
    # Load data
    if model_type == 'fnn':
        dataloader = FNNDataLoader(config)
    else:
        dataloader = GATDataLoader(config)
    
    dataloader.load_data()
    
    # Get appropriate dataloader
    if args.split == 'train':
        loader = dataloader.get_train_loader(shuffle=False)
    elif args.split == 'val':
        loader = dataloader.get_val_loader()
    else:
        loader = dataloader.get_test_loader()
    
    # Evaluate
    print(f"Evaluating {model_type.upper()} model on {args.split} set...")
    results = evaluate_model(model, loader, device, config.num_classes, model_type)
    
    # Print results
    print(f"\n{args.split.capitalize()} Results:")
    print(f"AUPRC: {results['auprc']:.4f}")
    print(f"Accuracy: {results['avg_accuracy']:.4f}")
    print(f"Average Loss: {results['avg_loss']:.4f}")
    
    # Save results
    results_path = exp_dir / f"{args.split}_results_{args.checkpoint}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()
