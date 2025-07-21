import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

def load_and_preprocess_data(data_path):
    """Load data as adata object and add manual annotations"""
    try:
        print(f"Loading data from: {data_path}")
        adata = sc.read_h5ad(data_path)
        adata.uns['log1p']["base"] = None  # bug fix
    except Exception as e:
        print(f"Error loading H5AD file: {str(e)}")
        exit(1)

    # Cell type annotation
    cluster_type = 'my_clust_1'
    annotation_dict = {
        '9': 'CAP1', '24': 'CAP2', '9b': 'VEC', '27': 'LEC',
        '17': 'Ciliated', '15': 'Secretory', '22': 'AT1', '6': 'AT2',
        '12': 'AT2-t1', '19': 'AT2-t2', '14': 'AF', '25': 'Pericyte',
        '20': 'Mesothelial', '3': 'B1', '3b': 'B2', '0': 'Th1',
        '8': 'Tnaive', '11': 'Tex', '77': 'Treg', '11b': 'NK',
        '4a': 'AM', '4': 'M-t1', '10': 'M-lc', '7': 'M-t2',
        '7b': 'M-C1q', '7c': 'iMon', '23': 'pDC', '13': 'DC',
        '5b': 'N1', '5': 'N2'
    }

    # Add cell type annotations
    adata.obs['cell_type_distribution'] = [annotation_dict[clust] for clust in adata.obs[cluster_type]]

    # Numeric encoding for cell types
    replacement_dict = {
        'AT2': 0, 'B1': 1, 'M-t1': 2, 'DC': 3, 'Th1': 4,
        'M-t2': 5, 'Secretory': 6, 'AM': 7, 'N1': 8, 'M-C1q': 9,
        'AT2-t2': 10, 'AF': 11, 'VEC': 12, 'CAP1': 13, 'N2': 14,
        'AT2-t1': 15, 'Pericyte': 16, 'pDC': 17, 'Ciliated': 18,
        'NK': 19, 'AT1': 20, 'Tnaive': 21, 'Treg': 22, 'M-lc': 23,
        'Mesothelial': 24, 'Tex': 25, 'CAP2': 26, 'LEC': 27, 'iMon': 28,
        'B2': 29
    }

    adata.obs['celltype'] = (
        adata.obs['cell_type_distribution']
        .replace(replacement_dict)
        .astype(np.int64)  
    )

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


def create_features(train_features, val_features, test_features, features_dir, scGNN_dir):
    """Create and save train/val/test features as CSV files formatted for scGNN"""
    print("\nCreating features...")
    
    # Ensure output directory exists
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(scGNN_dir, exist_ok=True)

    
    if scipy.sparse.issparse(train_features):
        train_features = train_features.toarray()
        val_features = val_features.toarray()
        test_features = test_features.toarray()
    
    #Save as .npy

    np.save(os.path.join(features_dir, "train_features.npy"), train_features)
    np.save(os.path.join(features_dir, "val_features.npy"), val_features)
    np.save(os.path.join(features_dir, "test_features.npy"), test_features)
    
    print(f"Features saved to: {features_dir}\n")


    #Save as .csv

    datasets = [
        (train_features, "train_features.csv"),
        (val_features, "val_features.csv"), 
        (test_features, "test_features.csv")
    ]

    for features, filename in datasets:

        features_df = pd.DataFrame(
            data=features.T,  # Transpose for scGNN format
            index=[f'gene_{i}' for i in range(features.shape[1])],  # genes as rows
            columns=[f'cell_{i}' for i in range(features.shape[0])]  # cells as columns
        )
        
        csv_path = os.path.join(scGNN_dir, filename)
        features_df.to_csv(csv_path, chunksize=1000)

    print(f"Features saved to: {scGNN_dir}\n")



def create_manual_annotation(adata, manual_annotation_dir):
    """Create and save H5AD files with manual annotations"""
    print("\nCreating manual annotations...")

    # Broad cell type classification
    fourth_dict = {
        'Blood vessels': ['CAP1','CAP2','VEC','AEC'],
        'Lymphatic EC': ['LEC'],
        'Airway epithelium': ['Ciliated','Secretory'],
        'Alveolar epithelium': ['AT1','AT2','AT2-t1','AT2-t2'],
        'Fibroblast': ['AF','Pericyte'],
        'Smooth muscle': ['SMC'],
        'Mesothelial': ['Mesothelial'],
        'B lineage': ['B1','B2'],
        'T lineage': ['Th1','Tnaive','Treg','Tex'],
        'NK': ['NK'],
        'Macrophage': ['AM','M-t1','M-t2','M-C1q','M-lc'],        
        'mononuclear': ['iMon','DC','pDC'],
        'Neutrophil': ['N1','N2']
    }

    group_lookup = {ft: broad_cat for broad_cat, fine_types in fourth_dict.items() for ft in fine_types}
    adata.obs['broad_celltype'] = adata.obs['cell_type_distribution'].map(group_lookup)

    # Ensure output directory exists
    os.makedirs(manual_annotation_dir, exist_ok=True)
    
    # Save subset H5AD files by broad cell type
    cell_type_subsets = {
        cat: adata[adata.obs['broad_celltype'] == cat].copy()
        for cat in adata.obs['broad_celltype'].unique()
    }
    
    for cell_type, adata_sub in cell_type_subsets.items():
        output_path = os.path.join(manual_annotation_dir, f"{cell_type.lower().replace(' ', '_')}.h5ad")
        adata_sub.write(output_path)
        print(f"Saved {cell_type}: {adata_sub.shape[0]} cells")
    
    # Save full processed data
    annotated_path = os.path.join(manual_annotation_dir, "annotated_granulomas.h5ad")
    adata.write(annotated_path)
    print(f"Saved full annotated data: {adata.shape[0]} cells")
    
    print(f"Manual annotations saved to: {manual_annotation_dir}\n")

def main():
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths for input and output
    data_path = os.path.join(script_dir, "../data/granulomas_final.h5ad")
    labels_dir = os.path.join(script_dir, "../data/labels")
    features_dir = os.path.join(script_dir, "../data/features")
    scGNN_dir = os.path.join(script_dir, "../data/scGNN")
    manual_annotation_dir = os.path.join(script_dir, "../data/Manual_Annotation")
    
    # Load and preprocess data
    adata = load_and_preprocess_data(data_path)
    
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
    create_features(train_features, val_features, test_features, features_dir, scGNN_dir)
    
    # Create manual annotation files
    create_manual_annotation(adata, manual_annotation_dir)

if __name__ == "__main__":
    main()