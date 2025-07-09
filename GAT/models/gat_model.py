import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATModel(nn.Module):
    """
    Graph Attention Network for cell type annotation
    
    Key Concepts:
    - Feature Transformation Weights: Linear layers that transform gene expression 
      into meaningful hidden representations (e.g., 2000 genes → 128 features)
    - Attention Computation Weights: Learn how to calculate attention scores between 
      connected cells to determine influence strength
    - Size-Agnostic: Weights work on any graph size (20K nodes for training, 
      10K nodes for testing) because they operate on features, not specific nodes
    """
    
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
        
        # FINAL GAT LAYER
        # Feature Transformation: hidden features → output_channels (cell type logits)
        # Single head for final classification (no concatenation)
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
        
        # ADDITIONAL MLP CLASSIFICATION HEAD
        # Final feature transformation: refines GAT output for better classification
        # These are also FEATURE TRANSFORMATION WEIGHTS
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),  # Transform GAT output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)   # Final cell type prediction
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GAT
        
        Args:
            x: Node features [num_nodes, in_channels] - gene expression for each cell
               Can be ANY number of nodes (20K for training, 10K for testing)
            edge_index: Graph connectivity [2, num_edges] - which cells are connected
                       Can be ANY graph structure, learned weights adapt automatically
            batch: Not used in single graph case
            
        Returns:
            Cell type predictions [num_nodes, out_channels] for each input cell
        """
        
        # STEP 1: Normalize input gene expression features
        x = self.bn_input(x)  # Shape: [num_nodes, in_channels]
        
        # STEP 2: Apply GAT layers with residual connections
        for i in range(self.num_layers - 1):
            x_res = x  # Store for residual connection
            
            # CORE GAT OPERATION:
            # 1. FEATURE TRANSFORMATION: Apply learned weight matrix W to transform features
            # 2. ATTENTION COMPUTATION: For each edge (i,j), compute attention score 
            #    using learned attention weights: attention_ij = a([W*h_i || W*h_j])
            # 3. ATTENTION-WEIGHTED AGGREGATION: Each node aggregates neighbor features
            #    weighted by attention scores: h_i' = Σ(attention_ij * W*h_j)
            x = self.convs[i](x, edge_index)
            
            # Batch normalization and activation
            x = self.bns[i](x)
            x = F.elu(x)  # ELU activation function
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # RESIDUAL CONNECTION: Helps with gradient flow and learning
            # Only add if dimensions match (first layer might have different dims)
            if x_res.shape[1] == x.shape[1]:
                x = x + x_res  # Element-wise addition
        
        # STEP 3: Final GAT layer (output layer)
        # Final feature transformation to cell type space
        x = self.convs[-1](x, edge_index)  # Shape: [num_nodes, out_channels]
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # STEP 4: Additional classification head
        # Final refinement of features for cell type prediction
        x = self.classifier(x)  # Shape: [num_nodes, out_channels]
        
        # OUTPUT: Logits for each cell's predicted cell type
        # Use argmax during inference to get predicted class
        return x  # Shape: [num_nodes, num_cell_types]


"""
KEY TRANSFERABILITY CONCEPTS:

1. FEATURE TRANSFORMATION WEIGHTS:
   - Linear layers (nn.Linear) with weight matrices W
   - Transform input dimensions to output dimensions
   - Work on ANY number of cells because matrix multiplication is size-agnostic
   - Example: W[2000×128] can transform [20K×2000] or [10K×2000] gene expression

2. ATTENTION COMPUTATION WEIGHTS:
   - Learned in GATv2Conv as attention parameter 'a'
   - Compute attention scores between any two connected cells
   - Work on ANY graph structure because they operate per-edge
   - Example: Same attention function works on 20K-node graph or 10K-node graph

3. SIZE GENERALIZATION:
   - Training: Learn weights on 20K cell graph
   - Testing: Apply same weights to 10K cell graph
   - Works because weights encode general rules for:
     * How to transform gene expression features
     * How to compute cell-cell attention
     * How to aggregate neighbor information
   - No node-specific parameters, only feature-based transformations

4. GRAPH STRUCTURE INDEPENDENCE:
   - Model learns HOW to process graph-structured data
   - Doesn't memorize specific graph connections
   - Can handle different cell neighborhoods and connectivity patterns
   - Attention mechanism adapts to whatever graph structure is provided
"""