import os
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def validate(model, val_data, criterion, device, rank):
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


def train(rank, world_size, args):
    """Main training function"""
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Load data
    input_path = args.data_dir
    train_features = np.load(os.path.join(input_path, "train_features.npy"))
    val_features = np.load(os.path.join(input_path, "val_features.npy"))
    train_labels = np.load(os.path.join(input_path, "train_labels.npy"))
    val_labels = np.load(os.path.join(input_path, "val_labels.npy"))
    train_edge_index = np.load(os.path.join(input_path, "train_edge_index.npy"))
    val_edge_index = np.load(os.path.join(input_path, "val_edge_index.npy"))
    sample_weights = np.load(os.path.join(input_path, "sample_weights.npy"))
    
    # Load metadata
    with open(os.path.join(input_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Create datasets
    train_dataset = GraphDataset(train_features, train_edge_index, train_labels, sample_weights)
    val_dataset = GraphDataset(val_features, val_edge_index, val_labels)
    
    # Get single graph data
    train_data = train_dataset.get(0)
    val_data = val_dataset.get(0)
    
    # Create model
    model = GATModel(
        in_channels=metadata['n_features'],
        hidden_channels=args.hidden_channels,
        out_channels=metadata['n_classes'],
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Convert batch norm to sync batch norm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    train_data = train_data.to(device)
    
    for epoch in range(args.epochs):
        ddp_model.train()
        
        # Forward pass
        output = ddp_model(train_data.x, train_data.edge_index)
        
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
        
        # Validation
        val_loss, val_acc = validate(ddp_model, val_data, nn.CrossEntropyLoss(), device, rank)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Logging (only on rank 0)
        if rank == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] "
                  f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.output_path)
                print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
    
    if rank == 0:
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train GAT model with DDP')
    parser.add_argument('--data_dir', type=str, default='../Arrays', help='Data directory')
    parser.add_argument('--output_path', type=str, default='/gpfs/scratch/blukacsy/granulomas_gat_v1.pt', 
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    
    # Check GPU availability
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU found. DDP requires at least one GPU.")
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Spawn processes
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
