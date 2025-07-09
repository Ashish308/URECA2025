import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np

from GAT import GATModel

class GraphDataset(Dataset):
    """Dataset for loading graph data"""
    def __init__(self, features, edge_index, labels, sample_weights=None):
        super().__init__()
        self.data = Data(
            x=torch.FloatTensor(features),
            edge_index=torch.LongTensor(edge_index),
            y=torch.LongTensor(labels)
        )
        if sample_weights is not None:
            self.data.sample_weight = torch.FloatTensor(sample_weights)

    def len(self):
        return 1  # Single large graph

    def get(self, idx):
        return self.data
    
def get_device():
    """Auto-select the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def validate(model, val_data, criterion, device):
    """Validation function"""
    model.eval()
    
    with torch.no_grad():
        val_data = val_data.to(device)
        output = model(val_data.x, val_data.edge_index)
        val_loss = criterion(output, val_data.y).item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct = (pred == val_data.y).sum().item()
        accuracy = correct / val_data.y.size(0)
    
    return val_loss, accuracy

def train(model, train_data, optimizer, criterion, device):
    """Training function"""
    model.train()
    train_data = train_data.to(device)
    
    # Forward pass
    output = model(train_data.x, train_data.edge_index)
    
    # Calculate loss with sample weights
    losses = criterion(output, train_data.y)
    if hasattr(train_data, 'sample_weight'):
        losses = losses * train_data.sample_weight
    loss = losses.mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    pred = output.argmax(dim=1)
    correct = (pred == train_data.y).sum().item()
    train_acc = correct / train_data.y.size(0)
    
    return loss.item(), train_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to training features .npy')
    parser.add_argument('--train_features', type=str, required=True, help='Path to training features .npy')
    parser.add_argument('--val_features', type=str, required=True, help='Path to validation features .npy')
    parser.add_argument('--train_edges', type=str, required=True, help='Path to training edge index .npy')
    parser.add_argument('--val_edges', type=str, required=True, help='Path to validation edge index .npy')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels .npy')
    parser.add_argument('--val_labels', type=str, required=True, help='Path to validation labels .npy')
    parser.add_argument('--weights', type=str, default=None, help='Path to sample weights .npy')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--output_path', type=str, default='model.pt', help='Path to save the best model')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    train_features = np.load(args.train_features)
    val_features = np.load(args.val_features)
    train_edge_index = np.load(args.train_edges)
    val_edge_index = np.load(args.val_edges)
    train_labels = np.load(args.train_labels)
    val_labels = np.load(args.val_labels)
    sample_weights = np.load(args.weights) if args.weights else None

    # Create datasets
    train_dataset = GraphDataset(train_features, train_edge_index, train_labels, sample_weights)
    val_dataset = GraphDataset(val_features, val_edge_index, val_labels)
    
    # Get single graph data
    train_data = train_dataset[0]
    val_data = val_dataset[0]

    # Create model
    model = GATModel(
        in_channels=train_features.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=int(train_labels.max()) + 1,
        heads=args.num_heads
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Training
        train_loss, train_acc = train(model, train_data, optimizer, criterion, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_data, nn.CrossEntropyLoss(), device)
        
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()