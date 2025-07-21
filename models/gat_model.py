import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATModel(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input batch normalization - normalizes gene expression features
        self.bn_input = nn.BatchNorm1d(in_channels)
        
        # GAT layers and batch normalization layers
        self.convs = nn.ModuleList()  # Contains FEATURE TRANSFORMATION weights
        self.bns = nn.ModuleList()
        
        # FIRST GAT LAYER
        # Feature Transformation: in_channels (e.g., 2000 genes) → hidden_channels (e.g., 128)
        # Attention Computation: Learns how to compute attention between any two connected cells
        self.convs.append(
            GATv2Conv(
                in_channels,          # Input features (gene expression size)
                hidden_channels,      # Hidden features (learnable transformation)
                heads=num_heads,      # Multi-head attention (8 parallel attention mechanisms)
                dropout=dropout,
                add_self_loops=True,  # Each cell attends to itself
                edge_dim=None,        # No edge features, only node features
                fill_value='mean',    # How to handle self-loops
                bias=True,
                share_weights=False   # Different weights for source/target transformations
            )
        )
        # Batch norm for concatenated multi-head output (hidden_channels * num_heads)
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # HIDDEN GAT LAYERS
        # Each layer learns progressively more complex feature transformations
        # and attention patterns between cells
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_channels * num_heads,  # Input from previous layer
                    hidden_channels,              # Output size
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=None,
                    fill_value='mean',
                    bias=True,
                    share_weights=False
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        

        self.convs.append(
            GATv2Conv(
                hidden_channels * num_heads,  # Input from last hidden layer
                out_channels,                 # Number of cell types (e.g., 30)
                heads=1,                      # Single attention head
                concat=False,                 # Average instead of concatenate
                dropout=dropout,
                add_self_loops=True,
                edge_dim=None,
                fill_value='mean',
                bias=True,
                share_weights=False
            )
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),  # Transform GAT output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)   # Final cell type prediction
        )
        
    def forward(self, x, edge_index, batch=None):
    
        
        x = self.bn_input(x)  # Shape: [num_nodes, in_channels]
        
        for i in range(self.num_layers - 1):
            x_res = x  # Store for residual connection
            

            x = self.convs[i](x, edge_index)
            
            x = self.bns[i](x)
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
            
           
            if x_res.shape[1] == x.shape[1]:
                x = x + x_res  # Element-wise addition
        
    
        x = self.convs[-1](x, edge_index)  # Shape: [num_nodes, out_channels]
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # STEP 4: Additional classification head
        # Final refinement of features for cell type prediction
        x = self.classifier(x)  # Shape: [num_nodes, out_channels]
        
        # OUTPUT: Logits for each cell's predicted cell type
        # Use argmax during inference to get predicted class
        return x  # Shape: [num_nodes, num_cell_types]

 