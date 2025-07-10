import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import LabelEncoder

def load_graph_features_labels(graph_csv_path, features_npy_path, labels_npy_path=None):
    """
    Load graph and features data for GAT model
    
    Args:
        graph_csv_path: Path to CSV file with columns (NodeA, NodeB, weight)
        features_npy_path: Path to NPY file with shape (n_cells, n_genes)
        labels_npy_path: Optional path to labels NPY file
    
    Returns:
        torch_geometric.data.Data object
    """
    
    # Load graph data
    print(f"Loading graph from: {graph_csv_path}")
    graph_df = pd.read_csv(graph_csv_path)
    

    source_nodes = graph_df.iloc[:, 0].values
    target_nodes = graph_df.iloc[:, 1].values
    edge_weights = graph_df.iloc[:, 2].values 

    # Create edge index in PyTorch Geometric format [2, num_edges]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # Create edge attributes from weights
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    # Load features
    print(f"Loading features from: {features_npy_path}")
    features = np.load(features_npy_path)
    print(f"Features shape: {features.shape}")
    
    # Convert to torch tensor
    x = torch.tensor(features, dtype=torch.float)
    
    # Load labels if provided

    print(f"Loading labels from: {labels_npy_path}")
    labels = np.load(labels_npy_path)
    y = torch.tensor(labels, dtype=torch.long)
    

    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,                    # Node features [num_nodes, num_features]
        edge_index=edge_index,  # Edge connectivity [2, num_edges]
        edge_attr=edge_attr,    # Edge weights [num_edges, 1] 
        y=y                     # Node labels [num_nodes] (optional)
    )
    
    print(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges")

    return data


def load_train_val_test_data(train_graph_csv, train_features_npy, train_labels_npy,
                            val_graph_csv, val_features_npy, val_labels_npy,
                            test_graph_csv, test_features_npy, test_labels_npy):
    """
    Load training, validation, and optionally test data
    
    Returns:
        Dictionary with train, val, and optionally test Data objects
    """
    
    # Load training data
    train_data = load_graph_features_labels(train_graph_csv, train_features_npy, train_labels_npy)
    val_data = load_graph_features_labels(val_graph_csv, val_features_npy, val_labels_npy)
    test_data = load_graph_features_labels(test_graph_csv, test_features_npy, test_labels_npy)

    
    data_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    
    return data_dict

def create_dataloader(data, batch_size, shuffle):
    """
    Create DataLoader 
    
    Args:
        data: PyTorch Geometric Data object
        batch_size: Batch size (usually 1 for single large graph)
        shuffle: Whether to shuffle data
    
    Returns:
        PyTorch Geometric DataLoader
    """
    # For single large graph, we typically use batch_size=1
    loader = PyGDataLoader([data], batch_size=batch_size, shuffle=shuffle)
    return loader




    

    
 
