# Cell Type Classification with Graph Neural Networks

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================
# 1. CUSTOM MESSAGE PASSING LAYER
# =============================================

class CellGCNConv(MessagePassing):
    """
    Custom GCN layer for cell type classification with message passing
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(CellGCNConv, self).__init__(aggr='add')  # Use 'add' aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, x, edge_index):
        """
        Forward pass with message passing
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        """
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Step 2: Linear transformation of node features
        x = self.lin(x)
        
        # Step 3: Compute normalization coefficients
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4: Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)
        
        # Step 5: Apply bias
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def message(self, x_j, norm):
        """
        Construct messages from neighboring nodes
        Args:
            x_j: Neighboring node features [num_edges, out_channels]
            norm: Normalization coefficients [num_edges]
        """
        # Normalize features by graph structure
        return norm.view(-1, 1) * x_j

# =============================================
# 2. GRAPH NEURAL NETWORK MODEL
# =============================================

class CellTypeGNN(nn.Module):
    """
    Graph Neural Network for cell type classification
    """
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(CellTypeGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First GNN layer
        self.convs = nn.ModuleList()
        self.convs.append(CellGCNConv(num_features, hidden_dim))
        
        # Hidden GNN layers
        for _ in range(num_layers - 2):
            self.convs.append(CellGCNConv(hidden_dim, hidden_dim))
        
        # Final layer
        if num_layers > 1:
            self.convs.append(CellGCNConv(hidden_dim, num_classes))
        else:
            self.convs[0] = CellGCNConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GNN
        """
        # Apply GNN layers with ReLU and dropout
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

# =============================================
# 3. DATA GENERATION AND PREPROCESSING
# =============================================

def generate_synthetic_cell_data(n_cells=1000, n_genes=500, n_cell_types=5):
    """
    Generate synthetic single-cell RNA-seq data for testing
    """
    print("Generating synthetic cell data...")
    
    # Create cell type labels
    cell_types = np.random.choice(n_cell_types, n_cells)
    
    # Generate gene expression with cell-type-specific patterns
    X = np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(np.float32)
    
    # Add cell-type-specific marker genes
    for cell_type in range(n_cell_types):
        cell_mask = cell_types == cell_type
        marker_start = cell_type * (n_genes // n_cell_types)
        marker_end = marker_start + (n_genes // (n_cell_types * 2))
        X[cell_mask, marker_start:marker_end] *= 3  # Boost marker expression
    
    # Log-normalize
    X = np.log1p(X)
    
    # Create cell names and gene names
    cell_names = [f"Cell_{i}" for i in range(n_cells)]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    cell_type_names = [f"CellType_{i}" for i in cell_types]
    
    return X, cell_types, cell_names, gene_names, cell_type_names

def create_cell_graph(X, k=10, method='knn'):
    """
    Create a graph from cell expression data
    Args:
        X: Cell expression matrix [n_cells, n_genes]
        k: Number of nearest neighbors
        method: 'knn' for k-nearest neighbors
    """
    print(f"Creating cell graph using {method} with k={k}...")
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    
    if method == 'knn':
        # Use cosine similarity for finding neighbors
        similarities = cosine_similarity(X)
        
        # Create k-NN graph
        nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        # Build edge list (exclude self-connections)
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # Skip self (first element)
                edges.append([i, neighbor])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    return edge_index

def prepare_graph_data(X, cell_types, edge_index, train_ratio=0.6, val_ratio=0.2):
    """
    Prepare data for PyTorch Geometric
    """
    n_cells = X.shape[0]
    
    # Convert to tensors
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(cell_types, dtype=torch.long)
    
    # Create train/val/test masks
    indices = np.random.permutation(n_cells)
    train_size = int(train_ratio * n_cells)
    val_size = int(val_ratio * n_cells)
    
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    val_mask = torch.zeros(n_cells, dtype=torch.bool)
    test_mask = torch.zeros(n_cells, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    return data

# =============================================
# 4. TRAINING FUNCTIONS
# =============================================

def train_model(model, data, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss only on training nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_model(model, data, mask, device):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc

# =============================================
# 5. VISUALIZATION FUNCTIONS
# =============================================

def plot_training_history(train_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_graph_structure(data, max_nodes=100):
    """Visualize the cell graph structure"""
    import networkx as nx
    from sklearn.manifold import TSNE
    
    # Subsample for visualization if too many nodes
    if data.x.shape[0] > max_nodes:
        indices = np.random.choice(data.x.shape[0], max_nodes, replace=False)
        # This is simplified - in practice you'd need to filter edges too
        print(f"Subsampling {max_nodes} cells for visualization")
    
    # Use t-SNE to visualize cell similarities
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(data.x.numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                         c=data.y.numpy(), cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Cell Types in 2D t-SNE Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

# =============================================
# 6. MAIN EXECUTION
# =============================================

def main():
    """Main execution function"""
    print("=== Cell Type Classification with Graph Neural Networks ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Generate synthetic data
    X, cell_types, cell_names, gene_names, cell_type_names = generate_synthetic_cell_data(
        n_cells=800, n_genes=200, n_cell_types=4
    )
    
    print(f"Generated data: {X.shape[0]} cells, {X.shape[1]} genes, {len(np.unique(cell_types))} cell types")
    
    # 2. Create cell graph
    edge_index = create_cell_graph(X, k=15)
    print(f"Created graph with {edge_index.shape[1]} edges")
    
    # 3. Prepare data
    data = prepare_graph_data(X, cell_types, edge_index)
    data = data.to(device)
    
    print(f"Train cells: {data.train_mask.sum()}")
    print(f"Val cells: {data.val_mask.sum()}")
    print(f"Test cells: {data.test_mask.sum()}")
    
    # 4. Create model
    model = CellTypeGNN(
        num_features=data.x.shape[1],
        hidden_dim=64,
        num_classes=len(np.unique(cell_types)),
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 5. Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 6. Training loop
    train_losses = []
    train_accs = []
    val_accs = []
    
    print("\nStarting training...")
    for epoch in range(100):
        # Train
        loss = train_model(model, data, optimizer, device)
        
        # Evaluate
        train_acc = evaluate_model(model, data, data.train_mask, device)
        val_acc = evaluate_model(model, data, data.val_mask, device)
        
        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    # 7. Final evaluation
    test_acc = evaluate_model(model, data, data.test_mask, device)
    print(f'\nFinal Test Accuracy: {test_acc:.4f}')
    
    # 8. Visualizations
    plot_training_history(train_losses, train_accs, val_accs)
    visualize_graph_structure(data.cpu())
    
    # 9. Detailed evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[data.test_mask].max(1)[1].cpu().numpy()
        true = data.y[data.test_mask].cpu().numpy()
    
    print("\nClassification Report:")
    print(classification_report(true, pred))
    
    return model, data

# Run the main function
if __name__ == "__main__":
    model, data = main()