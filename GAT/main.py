import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from gat_model import GATModel
from utils.load_data import load_train_val_test_data, create_dataloader
import argparse
import os
from torcheval.metrics import MulticlassAUPRC

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_graph', type=str, required=True, help='Path to training graph CSV')
    parser.add_argument('--train_features', type=str, required=True, help='Path to training features NPY')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels NPY')
    parser.add_argument('--val_graph', type=str, required=True, help='Path to validation graph CSV')
    parser.add_argument('--val_features', type=str, required=True, help='Path to validation features NPY')
    parser.add_argument('--val_labels', type=str, required=True, help='Path to validation labels NPY')
    parser.add_argument('--test_graph', type=str, required=True, help='Path to test graph CSV')
    parser.add_argument('--test_features', type=str, required=True, help='Path to test features NPY')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to test labels NPY')
    parser.add_argument('--in_channels', type=int, required=True, help='Input feature dimension')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--out_channels', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 for full graph)')
    parser.add_argument('--save_dir', type=str, default="saved_models/", help='Directory to save models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load datasets
    data_dict = load_train_val_test_data(
        args.train_graph, args.train_features, args.train_labels,
        args.val_graph, args.val_features, args.val_labels,
        args.test_graph, args.test_features, args.test_labels
    )

    # Create dataloaders (full-batch training)
    train_loader = create_dataloader(data_dict['train'], args.batch_size, shuffle=True)
    val_loader = create_dataloader(data_dict['val'], args.batch_size, shuffle=False)
    test_loader = create_dataloader(data_dict['test'], args.batch_size, shuffle=False) 

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATModel(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_auprc = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        val_auprc = evaluate_auprc(model, val_loader, device, args.out_channels)
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.4f}, Val AUPRC: {val_auprc:.4f}')

        # Save best model based on AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f'Best model saved at epoch {epoch+1} with val AUPRC {val_auprc:.4f}')

    # Test 

    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    test_auprc = evaluate_auprc(model, test_loader, device, args.out_channels)
    print(f'Test AUPRC: {test_auprc:.4f}')

def evaluate_auprc(model, loader, device, num_classes):
    """Evaluate model using Multiclass AUPRC metric"""
    metric = MulticlassAUPRC(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)            
            probabilities = F.softmax(out, dim=1)
            metric.update(probabilities.cpu(), batch.y.cpu())
    
    return metric.compute().item()

if __name__ == '__main__':
    main()