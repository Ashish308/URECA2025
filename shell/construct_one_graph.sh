#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide a name argument."
    echo "Usage: ./construct_one_graph.sh <name>"
    exit 1
fi

name=$1 

mkdir /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/graphs

python construct_graph_leiden.py \
    --input_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/features/train_features.npy \
    --output_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/graphs/train_graph.csv \
    --k 15 \
    --use_pca \
    --n_pcs 50

python construct_graph_leiden.py \
    --input_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/features/val_features.npy \
    --output_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/graphs/val_graph.csv \
    --k 15 \
    --use_pca \
    --n_pcs 50

python construct_graph_leiden.py \
    --input_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/features/test_features.npy \
    --output_file /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name}/graphs/test_graph.csv \
    --k 15 \
    --use_pca \
    --n_pcs 50

