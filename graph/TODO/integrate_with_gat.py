"""
Integration script to ensure compatibility with existing GAT training code
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


class GATIntegrator:
    """
    Ensures output is compatible with existing GAT dataloader
    """
    
    @staticmethod
    def verify_data_format(data_dir: Path) -> bool:
        """
        Verify that data is in correct format for GAT
        """
        required_files = {
            'Features': ['train_features.npy', 'val_features.npy', 'test_features.npy'],
            'Labels': ['train_labels.npy', 'val_labels.npy', 'test_labels.npy'],
            'Graphs': ['train_graph.csv', 'val_graph.csv', 'test_graph.csv']
        }
        
        all_exist = True
        for subdir, files in required_files.items():
            for file in files:
                file_path = data_dir / subdir / file
                if not file_path.exists():
                    print(f"Missing: {file_path}")
                    all_exist = False
                else:
                    print(f"Found: {file_path}")
                    
        return all_exist
    
    @staticmethod
    def create_sample_weights(data_dir: Path):
        """
        Create sample weights based on cluster sizes
        """
        labels_dir = data_dir / 'Labels'
        
        for split in ['train', 'val', 'test']:
            labels = np.load(labels_dir / f'{split}_labels.npy')
            
            # Calculate inverse frequency weights
            unique, counts = np.unique(labels, return_counts=True)
            weights = {label: 1.0 / count for label, count in zip(unique, counts)}
            sample_weights = np.array([weights[label] for label in labels])
            
            # Normalize
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
            
            # Save
            np.save(labels_dir / f'{split}_weights.npy', sample_weights)
            
        print("Created sample weights for balanced training")
    
    @staticmethod
    def generate_training_script(data_dir: Path, output_path: Path):
        """
        Generate a training script for the GAT model
        """
        script_content = f"""#!/bin/bash
# Auto-generated training script for GAT

# Training GAT model
python train_gat.py \\
    --features_dir {data_dir}/Features \\
    --graph_dir {data_dir}/Graphs \\
    --labels_dir {data_dir}/Labels \\
    --in_channels 23693 \\
    --hidden_channels 128 \\
    --out_channels 30 \\
    --num_heads 8 \\
    --num_layers 3 \\
    --dropout 0.2 \\
    --lr 0.005 \\
    --epochs 100 \\
    --batch_size 1 \\
    --save_dir saved_models/

# Training FNN model for comparison
python train_fnn.py \\
    --features_dir {data_dir}/Features \\
    --labels_dir {data_dir}/Labels \\
    --lr 0.005 \\
    --epochs 100 \\
    --batch_size 1000 \\
    --save_dir saved_models/
"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        output_path.chmod(0o755)
        
        print(f"Generated training script: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Integrate pipeline output with GAT')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with pipeline output')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify data format')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    # Verify data format
    print("Verifying data format...")
    if GATIntegrator.verify_data_format(data_dir):
        print("✓ All required files are present")
    else:
        print("✗ Some files are missing")
        return
    
    if not args.verify_only:
        # Create sample weights
        print("\nCreating sample weights...")
        GATIntegrator.create_sample_weights(data_dir)
        
        # Generate training script
        print("\nGenerating training script...")
        GATIntegrator.generate_training_script(
            data_dir,
            data_dir / 'train_models.sh'
        )
        
        print("\nIntegration complete! You can now train your models using:")
        print(f"  bash {data_dir}/train_models.sh")


if __name__ == '__main__':
    main()
