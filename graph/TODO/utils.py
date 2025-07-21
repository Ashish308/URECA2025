"""
Utility functions for the pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path


def evaluate_clustering(features: np.ndarray, labels: np.ndarray, 
                       output_dir: Path = None) -> dict:
    """
    Evaluate clustering quality using multiple metrics
    """
    metrics = {
        'silhouette_score': silhouette_score(features, labels),
        'davies_bouldin_score': davies_bouldin_score(features, labels),
        'calinski_harabasz_score': calinski_harabasz_score(features, labels),
        'n_clusters': len(np.unique(labels))
    }
    
    if output_dir:
        with open(output_dir / 'clustering_metrics.txt', 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    
    return metrics


def plot_cluster_distribution(labels: np.ndarray, output_path: Path):
    """
    Plot distribution of cluster sizes
    """
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.bar(unique, counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Cells')
    plt.title('Distribution of Cluster Sizes')
    
    # Add count labels on bars
    for i, (cluster, count) in enumerate(zip(unique, counts)):
        plt.text(cluster, count + max(counts) * 0.01, str(count), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_graph_properties(edge_list: list, n_nodes: int, 
                           output_dir: Path = None) -> dict:
    """
    Analyze graph properties
    """
    import networkx as nx
    
    # Create networkx graph
    G = nx.Graph()
    for src, tgt, weight in edge_list:
        G.add_edge(src, tgt, weight=weight)
    
    # Calculate properties
    properties = {
        'n_nodes': n_nodes,
        'n_edges': G.number_of_edges(),
        'avg_degree': 2 * G.number_of_edges() / n_nodes,
        'density': nx.density(G),
        'n_connected_components': nx.number_connected_components(G)
    }
    
    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    properties['largest_component_size'] = len(largest_cc)
    properties['largest_component_ratio'] = len(largest_cc) / n_nodes
    
    if output_dir:
        with open(output_dir / 'graph_properties.txt', 'w') as f:
            for prop, value in properties.items():
                f.write(f"{prop}: {value}\n")
    
    return properties


def compare_with_scgnn(pipeline_output_dir: str, scgnn_graph_file: str):
    """
    Compare pipeline output with scGNN graph
    """
    # Load pipeline graph
    pipeline_graph = pd.read_csv(Path(pipeline_output_dir) / 'full_graph.csv')
    
    # Load scGNN graph
    scgnn_graph = pd.read_csv(scgnn_graph_file)
    
    # Compare properties
    print("Pipeline graph:")
    print(f"  Nodes: {pipeline_graph[['source', 'target']].max().max() + 1}")
    print(f"  Edges: {len(pipeline_graph)}")
    print(f"  Avg weight: {pipeline_graph['weight'].mean():.4f}")
    
    print("\nscGNN graph:")
    print(f"  Nodes: {scgnn_graph[['NodeA', 'NodeB']].max().max() + 1}")  
    print(f"  Edges: {len(scgnn_graph)}")
    if 'Weights' in scgnn_graph.columns:
        print(f"  Avg weight: {scgnn_graph['Weights'].mean():.4f}")
