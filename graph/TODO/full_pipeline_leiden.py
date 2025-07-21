#!/usr/bin/env python3
"""
Full pipeline including Leiden clustering, data splitting, and GAT preparation
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import igraph as ig
import leidenalg
import argparse
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from construct_graph_leiden import GraphConstructor


class FullPipeline:
    """
    Complete pipeline for graph construction, clustering, and data preparation
    """
    
    def __init__(self, 
                 k: int = 15,
                 distance_metric: str = 'euclidean',
                 use_pca: bool = True,
                 n_pcs: int = 50,
                 scale_data: bool = True,
                 resolution: float = 1.0,
                 prune_graph: bool = True,
                 prune_std: float = 1.0,
                 seed: int = 42):
        """
        Initialize pipeline
        """
        self.graph_constructor = GraphConstructor(
            k=k,
            distance_metric=distance_metric,
            use_pca=use_pca,
            n_pcs=n_pcs,
            scale_data=scale_data,
            prune_graph=prune_graph,
            prune_std=prune_std,
            seed=seed
        )
        self.resolution = resolution
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
    def construct_full_graph(self, expression_matrix: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Construct graph and return both edge list and adjacency matrix
        """
        # Preprocess features
        features = self.graph_constructor.preprocess_features(expression_matrix)
        
        # Construct graph
        graph_df = self.graph_constructor.construct_knn_graph(features)
        
        # Create adjacency matrix
        n_cells = expression_matrix.shape[0]
        row_indices = graph_df['source'].values
        col_indices = graph_df['target'].values
        weights = graph_df['weight'].values
        
        adj_matrix = sp.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_cells, n_cells)
        )
        
        # Make symmetric
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix = adj_matrix.multiply(0.5)
        
        return graph_df, adj_matrix
    
    def perform_leiden_clustering(self, adj_matrix: sp.sparse.csr_matrix) -> np.ndarray:
        """
        Perform Leiden clustering on the adjacency matrix
        """
        self.logger.info(f"Performing Leiden clustering with resolution={self.resolution}")
        
        # Convert to igraph
        sources, targets = adj_matrix.nonzero()
        weights = adj_matrix[sources, targets].A1
        edges = list(zip(sources, targets))
        
        # Create igraph
        g = ig.Graph(edges=edges, directed=False)
        g.es['weight'] = weights
        
        # Find partition using Leiden algorithm
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=self.resolution,
            seed=self.seed
        )
        
        clusters = np.array(partition.membership)
        n_clusters = len(set(clusters))
        
        self.logger.info(f"Found {n_clusters} clusters")
        
        return clusters
    
    def split_data(self, 
                   expression_matrix: np.ndarray,
                   graph_df: pd.DataFrame,
                   clusters: np.ndarray,
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2) -> Dict:
        """
        Split data into train/val/test sets
        """
        n_cells = expression_matrix.shape[0]
        indices = np.arange(n_cells)
        
        # Stratified split based on clusters
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1-train_ratio,
            stratify=clusters,
            random_state=self.seed
        )
        
        val_idx, test_idx = train_test_split(
            test_idx,
            test_size=(1-train_ratio-val_ratio)/(1-train_ratio),
            stratify=clusters[test_idx],
            random_state=self.seed
        )
        
        # Create subgraphs for each split
        split_data = {
            'train': self._create_split_data(expression_matrix, graph_df, clusters, train_idx),
            'val': self._create_split_data(expression_matrix, graph_df, clusters, val_idx),
            'test': self._create_split_data(expression_matrix, graph_df, clusters, test_idx)
        }
        
        return split_data
    
    def _create_split_data(self, expression_matrix, graph_df, clusters, indices):
        """
        Create data for a specific split
        """
        # Create index mapping
        index_map = {old: new for new, old in enumerate(indices)}
        
        # Filter edges
        edges_list = []
        for _, row in graph_df.iterrows():
            src, tgt = int(row['source']), int(row['target'])
            if src in indices and tgt in indices:
                edges_list.append({
                    'source': index_map[src],
                    'target': index_map[tgt],
                    'weight': row['weight']
                })
        
        return {
            'features': expression_matrix[indices],
            'labels': clusters[indices],
            'edges': pd.DataFrame(edges_list),
            'indices': indices
        }
    
    def save_for_gat(self, split_data: Dict, output_dir: Path):
        """
        Save data in format compatible with GAT training
        """
        # Create directory structure
        features_dir = output_dir / 'Features'
        labels_dir = output_dir / 'Labels'
        graphs_dir = output_dir / 'scGNN'
        
        for dir_path in [features_dir, labels_dir, graphs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save for each split
        for split_name, data in split_data.items():
            # Save features
            np.save(features_dir / f'{split_name}_features.npy', data['features'])
            
            # Save labels
            np.save(labels_dir / f'{split_name}_labels.npy', data['labels'])
            
            # Save graph
            data['edges'].to_csv(graphs_dir / f'{split_name}_graph.csv', index=False)
        
        self.logger.info(f"Data saved to {output_dir}")
    
    def run_full_pipeline(self, 
                         expression_matrix: np.ndarray,
                         output_dir: str,
                         train_ratio: float = 0.6,
                         val_ratio: float = 0.2):
        """
        Run the complete pipeline
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Construct graph
        self.logger.info("Constructing graph...")
        graph_df, adj_matrix = self.construct_full_graph(expression_matrix)
        
        # Save full graph
        graph_df.to_csv(output_dir / 'full_graph.csv', index=False)
        
        # Step 2: Perform clustering
        self.logger.info("Performing Leiden clustering...")
        clusters = self.perform_leiden_clustering(adj_matrix)
        
        # Save clusters
        cluster_df = pd.DataFrame({
            'cell_id': range(len(clusters)),
            'cluster': clusters
        })
        cluster_df.to_csv(output_dir / 'clusters.csv', index=False)
        
        # Step 3: Split data
        self.logger.info("Splitting data...")
        split_data = self.split_data(
            expression_matrix,
            graph_df,
            clusters,
            train_ratio,
            val_ratio
        )
        
        # Step 4: Save for GAT
        self.save_for_gat(split_data, output_dir)
        
        # Save metadata
        metadata = {
            'n_cells': expression_matrix.shape[0],
            'n_genes': expression_matrix.shape[1],
            'n_clusters': len(np.unique(clusters)),
            'graph_params': {
                'k': self.graph_constructor.k,
                'distance_metric': self.graph_constructor.distance_metric,
                'use_pca': self.graph_constructor.use_pca,
                'n_pcs': self.graph_constructor.n_pcs
            },
            'clustering_params': {
                'resolution': self.resolution
            },
            'splits': {
                name: {
                    'n_cells': len(data['indices']),
                    'n_edges': len(data['edges'])
                }
                for name, data in split_data.items()
            }
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info("Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Full pipeline with Leiden clustering')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to .npy file with expression matrix')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all results')
    
    # Graph construction parameters
    parser.add_argument('--k', type=int, default=15,
                       help='Number of nearest neighbors')
    parser.add_argument('--distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'correlation'],
                       help='Distance metric')
    parser.add_argument('--use_pca', action='store_true',
                       help='Use PCA before graph construction')
    parser.add_argument('--n_pcs', type=int, default=50,
                       help='Number of principal components')
    
    # Clustering parameters
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Leiden resolution parameter')
    
    # Data splitting parameters
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    expression_matrix = np.load(args.input_file)
    
    # Create and run pipeline
    pipeline = FullPipeline(
        k=args.k,
        distance_metric=args.distance_metric,
        use_pca=args.use_pca,
        n_pcs=args.n_pcs,
        resolution=args.resolution,
        seed=args.seed
    )
    
    pipeline.run_full_pipeline(
        expression_matrix,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == '__main__':
    main()
