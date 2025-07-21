#!/usr/bin/env python3
"""
Example script showing how to use the kNN-Leiden pipeline
"""

import numpy as np
from pathlib import Path
import sys

# Add pipeline directory to path
sys.path.append('pipeline')

from construct_graph_leiden import KNNLeidenPipeline
from integrate_with_gat import GATIntegrator
from utils import evaluate_clustering, plot_cluster_distribution, analyze_graph_properties


def run_example():
    """
    Example usage of the pipeline
    """
    # Configuration
    input_file = 'data/expression_matrix.npy'  # Your expression matrix
    output_dir = 'results/leiden_knn_output'
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found!")
        print("Please provide a gene expression matrix as a .npy file")
        print("Shape should be (n_cells, n_genes)")
        return
    
    # Load expression matrix
    print(f"Loading expression matrix from {input_file}")
    expression_matrix = np.load(input_file)
    print(f"Expression matrix shape: {expression_matrix.shape}")
    
    # Initialize pipeline with custom parameters
    pipeline = KNNLeidenPipeline(
        k=15,                    # Number of nearest neighbors
        distance_metric='euclidean',  # Distance metric
        use_pca=True,           # Use PCA
        n_pcs=50,               # Number of PCs
        scale_data=True,        # Scale data
        resolution=1.0,         # Leiden resolution
        prune_graph=True,       # Statistical pruning
        prune_std=1.0,          # Pruning threshold
        seed=42                 # Random seed
    )
    
    # Run the pipeline
    print("\nRunning pipeline...")
    results = pipeline.run_pipeline(
        expression_matrix,
        output_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        use_scanpy_graph=False  # Use scGNN-style graph construction
    )
    
    print("\nPipeline completed!")
    print(f"Results saved to: {output_dir}")
    
    # Verify and integrate with GAT
    print("\nVerifying GAT compatibility...")
    output_path = Path(output_dir)
    
    integrator = GATIntegrator()
    if integrator.verify_data_format(output_path):
        print("✓ Data is compatible with GAT training")
        
        # Create sample weights
        integrator.create_sample_weights(output_path)
        
        # Generate training script
        integrator.generate_training_script(
            output_path,
            output_path / 'train_models.sh'
        )
        
        print(f"\nYou can now train your models using:")
        print(f"  cd {output_path}")
        print(f"  bash train_models.sh")
    else:
        print("✗ Data format issues detected")
    
    # Additional analysis
    print("\nPerforming additional analysis...")
    
    # Load results
    clusters = np.load(output_path / 'Labels' / 'train_labels.npy')
    features = np.load(output_path / 'Features' / 'train_features.npy')
    
    # Evaluate clustering
    metrics = evaluate_clustering(features, clusters, output_path)
    print("\nClustering metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot cluster distribution
    plot_cluster_distribution(clusters, output_path / 'cluster_distribution.png')
    print(f"\nCluster distribution plot saved to: {output_path / 'cluster_distribution.png'}")
    
    # Analyze graph properties
    import pandas as pd
    graph_df = pd.read_csv(output_path / 'scGNN' / 'train_graph.csv')
    edge_list = [(row['source'], row['target'], row['weight']) 
                 for _, row in graph_df.iterrows()]
    
    graph_props = analyze_graph_properties(
        edge_list, 
        len(clusters),
        output_path
    )
    print("\nGraph properties:")
    for prop, value in graph_props.items():
        print(f"  {prop}: {value}")


def generate_test_data():
    """
    Generate synthetic test data for demonstration
    """
    print("Generating synthetic test data...")
    
    # Generate synthetic single-cell data
    n_cells = 5000
    n_genes = 2000
    n_cell_types = 5
    
    # Create synthetic expression matrix with distinct cell types
    expression_matrix = np.zeros((n_cells, n_genes))
    
    cells_per_type = n_cells // n_cell_types
    for i in range(n_cell_types):
        start_idx = i * cells_per_type
        end_idx = (i + 1) * cells_per_type if i < n_cell_types - 1 else n_cells
        
        # Each cell type has different expression patterns
        base_expression = np.random.gamma(2, 2, n_genes)
        for j in range(start_idx, end_idx):
            noise = np.random.normal(0, 0.5, n_genes)
            expression_matrix[j] = np.maximum(base_expression + noise, 0)
    
    # Add some dropout
    dropout_mask = np.random.random(expression_matrix.shape) < 0.1
    expression_matrix[dropout_mask] = 0
    
    # Save test data
    Path('data').mkdir(exist_ok=True)
    np.save('data/expression_matrix.npy', expression_matrix)
    
    print(f"Generated synthetic data:")
    print(f"  Shape: {expression_matrix.shape}")
    print(f"  Saved to: data/expression_matrix.npy")
    
    return expression_matrix


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run kNN-Leiden pipeline example')
    parser.add_argument('--generate_test_data', action='store_true',
                       help='Generate synthetic test data')
    parser.add_argument('--input_file', type=str, default='data/expression_matrix.npy',
                       help='Path to expression matrix')
    parser.add_argument('--output_dir', type=str, default='results/leiden_knn_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.generate_test_data:
        generate_test_data()
    
    # Update paths if provided
    if args.input_file != 'data/expression_matrix.npy':
        input_file = args.input_file
    if args.output_dir != 'results/leiden_knn_output':
        output_dir = args.output_dir
    
    run_example()
