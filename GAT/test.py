import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from GAT import GATModel


def load_model(model_path, metadata, args, device):
    """Load trained model"""
    model = GATModel(
        in_channels=metadata['n_features'],
        hidden_channels=args.hidden_channels,
        out_channels=metadata['n_classes'],
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0  # No dropout during testing
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def evaluate_model(model, test_data, device):
    """Evaluate model on test data"""
    test_data = test_data.to(device)
    
    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index)
        probs = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
    
    return predictions.cpu().numpy(), probs.cpu().numpy()


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate various evaluation metrics"""
    results = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, average='weighted'),
        'f1_score': metrics.f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Try to calculate ROC AUC
    try:
        results['roc_auc_ovr'] = metrics.roc_auc_score(y_true, y_probs, multi_class='ovr')
        results['roc_auc_ovo'] = metrics.roc_auc_score(y_true, y_probs, multi_class='ovo')
    except:
        results['roc_auc_ovr'] = None
        results['roc_auc_ovo'] = None
    
    return results


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), 
                yticklabels=np.unique(y_true))
    plt.title('GAT Model - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_distribution(y_true, y_pred, save_path=None):
    """Plot true vs predicted class distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True distribution
    unique, counts = np.unique(y_true, return_counts=True)
    ax1.bar(unique, counts, color='skyblue')
    ax1.set_title('True Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    
    # Predicted distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax2.bar(unique_pred, counts_pred, color='lightcoral')
    ax2.set_title('Predicted Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_attention_weights(model, data, node_idx=0, device='cuda'):
    """Visualize attention weights for a specific node"""
    model.eval()
    data = data.to(device)
    
    # Get attention weights from the last GAT layer
    with torch.no_grad():
        # Forward pass with attention weights
        x = model.bn_input(data.x)
        
        attention_weights = []
        for i in range(model.num_layers):
            if i < model.num_layers - 1:
                x, (edge_index, alpha) = model.convs[i](x, data.edge_index, return_attention_weights=True)
                attention_weights.append((edge_index, alpha))
                x = model.bns[i](x)
                x = torch.nn.functional.elu(x)
            else:
                x, (edge_index, alpha) = model.convs[i](x, data.edge_index, return_attention_weights=True)
                attention_weights.append((edge_index, alpha))
    
    # Get attention weights for the specified node from the last layer
    edge_index, alpha = attention_weights[-1]
    
    # Find edges where the target is our node of interest
    mask = (edge_index[1] == node_idx).cpu()
    source_nodes = edge_index[0][mask].cpu().numpy()
    attn_values = alpha[mask].cpu().numpy().squeeze()
    
    if len(source_nodes) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(source_nodes)), attn_values)
        plt.xlabel('Neighbor Index')
        plt.ylabel('Attention Weight')
        plt.title(f'Attention Weights for Node {node_idx}')
        plt.show()
    else:
        print(f"No incoming edges found for node {node_idx}")


def main():
    parser = argparse.ArgumentParser(description='Test GAT model')
    parser.add_argument('--data_dir', type=str, default='../Arrays', help='Data directory')
    parser.add_argument('--model_path', type=str, default='/gpfs/scratch/blukacsy/granulomas_gat_v1.pt', 
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for plots')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GAT layers')
    parser.add_argument('--visualize_attention', action='store_true', help='Visualize attention weights')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    test_features = np.load(os.path.join(args.data_dir, "test_features.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "test_labels.npy"))
    test_edge_index = np.load(os.path.join(args.data_dir, "test_edge_index.npy"))
    
    # Load metadata
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Create test data
    test_data = Data(
        x=torch.FloatTensor(test_features),
        edge_index=torch.LongTensor(test_edge_index),
        y=torch.LongTensor(test_labels)
    )
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, metadata, args, device)
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, probabilities = evaluate_model(model, test_data, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(test_labels, predictions, probabilities)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for metric, value in results.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: N/A")
    
    # Classification report
    print("\nClassification Report:")
    print(metrics.classification_report(test_labels, predictions, zero_division=0))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(test_labels, predictions, 
                         save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot class distributions
    print("Plotting class distributions...")
    plot_class_distribution(test_labels, predictions,
                           save_path=os.path.join(args.output_dir, 'class_distribution.png'))
    
    # Visualize attention weights for a few nodes
    if args.visualize_attention:
        print("\nVisualizing attention weights...")
        for i in range(min(3, len(test_labels))):
            visualize_attention_weights(model, test_data, node_idx=i, device=device)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()
