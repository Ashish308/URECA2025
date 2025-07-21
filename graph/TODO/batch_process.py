"""
Batch processing script for multiple datasets or parameter configurations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict
import itertools
from construct_graph_leiden import KNNLeidenPipeline
import multiprocessing as mp
from functools import partial


class BatchProcessor:
    """
    Process multiple datasets or parameter configurations
    """
    
    def __init__(self, base_output_dir: str = 'batch_results'):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
    def process_parameter_grid(self, 
                             expression_matrix: np.ndarray,
                             param_grid: Dict[str, List],
                             n_jobs: int = -1) -> pd.DataFrame:
        """
        Process multiple parameter configurations
        
        Args:
            expression_matrix: Input expression matrix
            param_grid: Dictionary of parameters to test
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            DataFrame with results
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        # Prepare jobs
        jobs = []
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            output_dir = self.base_output_dir / f"config_{i:03d}"
            jobs.append((expression_matrix, param_dict, output_dir))
        
        # Run in parallel
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        with mp.Pool(n_jobs) as pool:
            results = pool.map(self._process_single_config, jobs)
        
        # Compile results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.base_output_dir / 'parameter_search_results.csv', index=False)
        
        # Find best configuration
        best_idx = results_df['silhouette_score'].idxmax()
        best_config = results_df.loc[best_idx]
        
        print(f"\nBest configuration (by silhouette score):")
        print(best_config)
        
        with open(self.base_output_dir / 'best_config.json', 'w') as f:
            json.dump(best_config.to_dict(), f, indent=2)
        
        return results_df
    
    def _process_single_config(self, args):
        """
        Process a single configuration
        """
        expression_matrix, params, output_dir = args
        
        try:
            # Create pipeline with parameters
            pipeline = KNNLeidenPipeline(**params)
            
            # Run pipeline
            results = pipeline.run_pipeline(
                expression_matrix,
                str(output_dir),
                train_ratio=0.6,
                val_ratio=0.2
            )
            
            # Load clustering results
            clusters = pd.read_csv(output_dir / 'clusters.csv')['cluster'].values
            
            # Calculate metrics
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA
            
            # Use PCA features for silhouette score
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(expression_matrix)
            
            sil_score = silhouette_score(features_pca, clusters)
            
            # Compile results
            result = params.copy()
            result.update({
                'output_dir': str(output_dir),
                'n_clusters': results['n_clusters'],
                'n_edges': results['n_edges'],
                'silhouette_score': sil_score,
                'status': 'success'
            })
            
        except Exception as e:
            result = params.copy()
            result.update({
                'output_dir': str(output_dir),
                'status': 'failed',
                'error': str(e)
            })
        
        return result
    
    def process_multiple_datasets(self,
                                dataset_paths: List[str],
                                pipeline_params: Dict,
                                n_jobs: int = -1) -> pd.DataFrame:
        """
        Process multiple datasets with same parameters
        """
        results = []
        
        for dataset_path in dataset_paths:
            dataset_name = Path(dataset_path).stem
            output_dir = self.base_output_dir / dataset_name
            
            try:
                # Load data
                expression_matrix = np.load(dataset_path)
                
                # Run pipeline
                pipeline = KNNLeidenPipeline(**pipeline_params)
                pipeline_results = pipeline.run_pipeline(
                    expression_matrix,
                    str(output_dir)
                )
                
                result = {
                    'dataset': dataset_name,
                    'dataset_path': dataset_path,
                    'n_cells': expression_matrix.shape[0],
                    'n_genes': expression_matrix.shape[1],
                    **pipeline_results,
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'dataset': dataset_name,
                    'dataset_path': dataset_path,
                    'status': 'failed',
                    'error': str(e)
                }
            
            results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.base_output_dir / 'dataset_results.csv', index=False)
        
        return results_df


# Example usage
if __name__ == '__main__':
    # Example: Parameter grid search
    expression_matrix = np.load('data/expression_matrix.npy')
    
    param_grid = {
        'k': [10, 15, 20, 30],
        'resolution': [0.5, 0.8, 1.0, 1.5],
        'distance_metric': ['euclidean', 'cosine'],
        'prune_graph': [True, False]
    }
    
    processor = BatchProcessor('batch_results')
    results = processor.process_parameter_grid(
        expression_matrix,
        param_grid,
        n_jobs=4
    )
    
    print("\nParameter search completed!")
    print(f"Results saved to: batch_results/parameter_search_results.csv")
