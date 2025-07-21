import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import argparse

def load_data(data_path):
    """Load data as adata object and add manual annotations"""
    try:
        print(f"Loading data from: {data_path}")
        adata = sc.read_h5ad(data_path)
        adata.uns['log1p']["base"] = None  # bug fix
    except Exception as e:
        print(f"Error loading H5AD file: {str(e)}")
        exit(1)

    return adata

def create_labels(train_labels, val_labels, test_labels, labels_dir):
    """Create and save train/val/test labels, sample weights, and indices"""
    
    print("\nCreating labels...")
    
    # Ensure output directory exists
    os.makedirs(labels_dir, exist_ok=True)
    
    
    # Save all label-related files
    np.save(os.path.join(labels_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(labels_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(labels_dir, "test_labels.npy"), test_labels)

    
    print(f"Labels saved to: {labels_dir}\n")


#TODO: integrate scGNN pipeline for graph creation and new kNN pipeline

# def create_features(train_features, val_features, test_features, features_dir, scGNN_dir):
#     """Create and save train/val/test features as CSV files formatted for scGNN"""
#     print("\nCreating features...")
    
#     # Ensure output directory exists
#     os.makedirs(features_dir, exist_ok=True)
#     os.makedirs(scGNN_dir, exist_ok=True)

    
#     if scipy.sparse.issparse(train_features):
#         train_features = train_features.toarray()
#         val_features = val_features.toarray()
#         test_features = test_features.toarray()
    
#     #Save as .npy

#     np.save(os.path.join(features_dir, "train_features.npy"), train_features)
#     np.save(os.path.join(features_dir, "val_features.npy"), val_features)
#     np.save(os.path.join(features_dir, "test_features.npy"), test_features)
    
#     print(f"Features saved to: {features_dir}\n")


#     #Save as .csv

#     datasets = [
#         (train_features, "train_features.csv"),
#         (val_features, "val_features.csv"), 
#         (test_features, "test_features.csv")
#     ]

#     for features, filename in datasets:

#         features_df = pd.DataFrame(
#             data=features.T,  # Transpose for scGNN format
#             index=[f'gene_{i}' for i in range(features.shape[1])],  # genes as rows
#             columns=[f'cell_{i}' for i in range(features.shape[0])]  # cells as columns
#         )
        
#         csv_path = os.path.join(scGNN_dir, filename)
#         features_df.to_csv(csv_path, chunksize=1000)

#     print(f"Features saved to: {scGNN_dir}\n")

def create_features(train_features, val_features, test_features, features_dir):
    """Create and save train/val/test features as CSV files formatted for scGNN"""
    print("\nCreating features...")
    
    # Ensure output directory exists
    os.makedirs(features_dir, exist_ok=True)

    
    if scipy.sparse.issparse(train_features):
        train_features = train_features.toarray()
        val_features = val_features.toarray()
        test_features = test_features.toarray()
    
    #Save as .npy

    np.save(os.path.join(features_dir, "train_features.npy"), train_features)
    np.save(os.path.join(features_dir, "val_features.npy"), val_features)
    np.save(os.path.join(features_dir, "test_features.npy"), test_features)
    
    print(f"Features saved to: {features_dir}\n")


def main(name):
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths for input and output
    data_path = os.path.join(script_dir, "../data/manual_annotation/" + name + ".h5ad")
    labels_dir = os.path.join(script_dir, "../data/manual_annotation/" + name + "/labels")
    features_dir = os.path.join(script_dir, "../data/manual_annotation/" + name + "/features")
    
    # Load and preprocess data
    adata = load_data(data_path)
    
    print(f"\nTotal cells: {adata.shape[0]}")
    print(f"Total genes: {adata.shape[1]}\n")
    print(f"Cell type distribution:\n{adata.obs['cell_type_distribution'].value_counts()}")

    # Split data into train/val/test
    X = adata.X
    y = adata.obs['celltype'].values
    
    # First split: 80% train+val, 20% test
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=19, stratify=y
    )
    
    # Second split: 60% train, 20% val (from the 80% above)
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, test_size=0.25, random_state=30, stratify=train_val_labels
    )
    
    print(f"\nData split summary:")
    print(f"Train: {train_features.shape[0]} cells")
    print(f"Val: {val_features.shape[0]} cells") 
    print(f"Test: {test_features.shape[0]} cells")
    
    # Create labels and save them
    create_labels(train_labels, val_labels, test_labels, labels_dir)
    
    # Create feature files
    create_features(train_features, val_features, test_features, features_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="name before .h5ad")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the parsed argument
    main(args.name)
