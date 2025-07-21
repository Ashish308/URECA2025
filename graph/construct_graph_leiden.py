#!/usr/bin/env python3
"""
Simplified script that constructs a kNN graph from expression matrix and outputs a single CSV file
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from existing scGNN if available
try:
    from scGNN.graph_function import kernelDistance
except:
    def kernelDistance(distance, delta=1.0):
        """Fallback kernel distance function"""
        return np.exp(-distance / (2 * delta**2))


class GraphConstructor:
    """
    Constructs kNN graph from expression matrix
    """
    
    def __init__(self, 
                 k: int = 15,
                 distance_metric: str = 'euclidean',
                 use_pca: bool = True,
                 n_pcs: int = 50,
                 scale_data: bool = True,
                 prune_graph: bool = True,
                 prune_std: float = 1.0,
                 seed: int = 42):
        """
        Initialize graph constructor
        
        Args:
            k: Number of nearest neighbors
            distance_metric: Distance metric ('euclidean', 'cosine', 'correlation')
            use_pca: Whether to use PCA
            n_pcs: Number of principal components
            scale_data: Whether to scale the data
            prune_graph: Whether to prune edges using statistical method
            prune_std: Standard deviations for pruning
            seed: Random seed
        """
        self.k = k
        self.distance_metric = distance_metric
        self.use_pca = use_pca
        self.n_pcs = n_pcs
        self.scale_data = scale_data
        self.prune_graph = prune_graph
        self.prune_std = prune_std
        self.seed = seed
        
        np.random.seed(seed)
        
    def preprocess_features(self, expression_matrix: np.ndarray) -> np.ndarray:
        """
        Preprocess expression matrix
        
        Args:
            expression_matrix: Gene expression matrix (cells x genes)
            
        Returns:
            Preprocessed matrix
        """
        # Log transform if not already done
        if expression_matrix.min() >= 0:
            expression_matrix = np.log1p(expression_matrix)
        
        # Scale data
        if self.scale_data:
            scaler = StandardScaler()
            expression_matrix = scaler.fit_transform(expression_matrix)
        
        # Apply PCA
        if self.use_pca:
            n_components = min(self.n_pcs, expression_matrix.shape[0] - 1, expression_matrix.shape[1])
            pca = PCA(n_components=n_components, random_state=self.seed)
            expression_matrix = pca.fit_transform(expression_matrix)
            
        return expression_matrix
    
    def construct_knn_graph(self, features: np.ndarray) -> pd.DataFrame:
        """
        Construct kNN graph and return as DataFrame
        
        Args:
            features: Feature matrix (cells x features)
            
        Returns:
            DataFrame with columns ['source', 'target', 'weight']
        """
        n_cells = features.shape[0]
        edge_list = []
        
        # Calculate distance matrix
        dist_matrix = distance.cdist(features, features, self.distance_metric)
        
        # For each cell, find k nearest neighbors
        for i in range(n_cells):
            # Get k+1 nearest neighbors (including self)
            distances = dist_matrix[i, :]
            nearest_indices = np.argsort(distances)[:self.k+1]
            nearest_distances = distances[nearest_indices]
            
            # Skip self (first index)
            neighbor_indices = nearest_indices[1:]
            neighbor_distances = nearest_distances[1:]
            
            if self.prune_graph:
                # Statistical pruning
                mean_dist = np.mean(neighbor_distances)
                std_dist = np.std(neighbor_distances)
                threshold = mean_dist + self.prune_std * std_dist
                
                for neighbor_idx, dist in zip(neighbor_indices, neighbor_distances):
                    if dist <= threshold:
                        # Convert distance to weight
                        weight = kernelDistance(dist)
                        edge_list.append({
                            'source': i,
                            'target': int(neighbor_idx),
                            'weight': float(weight)
                        })
            else:
                # Add all k neighbors
                for neighbor_idx, dist in zip(neighbor_indices, neighbor_distances):
                    weight = kernelDistance(dist)
                    edge_list.append({
                        'source': i,
                        'target': int(neighbor_idx),
                        'weight': float(weight)
                    })
        
        # Convert to DataFrame
        graph_df = pd.DataFrame(edge_list)
        
        return graph_df
    
    def construct_graph_from_file(self, input_file: str, output_file: str):
        """
        Main method to construct graph from file and save to CSV
        
        Args:
            input_file: Path to .npy file with expression matrix
            output_file: Path to output .csv file
        """
        # Load expression matrix
        logging.info(f"Loading expression matrix from {input_file}")
        expression_matrix = np.load(input_file)
        logging.info(f"Expression matrix shape: {expression_matrix.shape}")
        
        # Preprocess
        logging.info("Preprocessing features...")
        features = self.preprocess_features(expression_matrix)
        logging.info(f"Features shape after preprocessing: {features.shape}")
        
        # Construct graph
        logging.info(f"Constructing kNN graph with k={self.k}")
        graph_df = self.construct_knn_graph(features)
        logging.info(f"Graph constructed with {len(graph_df)} edges")
        
        # Save to CSV
        graph_df.to_csv(output_file, index=False)
        logging.info(f"Graph saved to {output_file}")
        
        return graph_df


def main():
    parser = argparse.ArgumentParser(description='Construct kNN graph from expression matrix')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to .npy file with expression matrix (cells x genes)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output .csv file')
    parser.add_argument('--k', type=int, default=15,
                       help='Number of nearest neighbors')
    parser.add_argument('--distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'correlation'],
                       help='Distance metric')
    parser.add_argument('--use_pca', action='store_true', default=False,
                       help='Use PCA before graph construction')
    parser.add_argument('--n_pcs', type=int, default=50,
                       help='Number of principal components')
    parser.add_argument('--no_scale', action='store_true',
                       help='Do not scale data')
    parser.add_argument('--no_prune', action='store_true',
                       help='Do not apply statistical pruning')
    parser.add_argument('--prune_std', type=float, default=1.0,
                       help='Standard deviations for pruning')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()

    
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create graph constructor
    constructor = GraphConstructor(
        k=args.k,
        distance_metric=args.distance_metric,
        use_pca=args.use_pca,
        n_pcs=args.n_pcs,
        scale_data=not args.no_scale,
        prune_graph=not args.no_prune,
        prune_std=args.prune_std,
        seed=args.seed
    )
    
    # Construct graph
    constructor.construct_graph_from_file(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
