import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from .base_dataloader import BaseDataLoader

class GATDataLoader(BaseDataLoader):
    """DataLoader for GAT model"""
    
    def load_data(self):
        """Load train, val, and test graph datasets"""
        self.train_data = self._load_graph_data('train')
        self.val_data = self._load_graph_data('val')
        self.test_data = self._load_graph_data('test')
        
        print(f"Loaded graphs - Train: {self.train_data.num_nodes} nodes, "
              f"Val: {self.val_data.num_nodes} nodes, Test: {self.test_data.num_nodes} nodes")
    
    def _load_graph_data(self, split):
        """Load graph data for a given split"""
        # Load features and labels
        features, labels = self._load_features_labels(split)
        
        # Load graph structure
        graph_path = self.config.graph_dir / f"{split}_graph.csv"
        print(f"Loading {split} graph from: {graph_path}")
        graph_df = pd.read_csv(graph_path)
        
        # Extract edge information
        source_nodes = graph_df.iloc[:, 0].values
        target_nodes = graph_df.iloc[:, 1].values
        edge_weights = graph_df.iloc[:, 2].values if graph_df.shape[1] > 2 else None
        
        # Create edge index
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Create edge attributes if weights exist
        edge_attr = None
        if edge_weights is not None:
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Convert to tensors
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        
        return data
    
    def get_dataloader(self, dataset, batch_size, shuffle=False):
        """Create a DataLoader from dataset"""
        # For single large graph, we typically use batch_size=1
        return PyGDataLoader(
            [dataset], 
            batch_size=batch_size, 
            shuffle=shuffle
        )
