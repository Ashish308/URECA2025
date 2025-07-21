#!/bin/bash

#Example: Train fnn model
python scripts/train.py --model fnn \
    --experiment_name fnn_experiment \
    --data_dir /Users/Beta/Desktop/Research/URECA2025/organized_version/data \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --epochs 200

#Example: Train gat model
python scripts/train.py --model gat \
    --experiment_name gat_experiment \
    --data_dir /Users/Beta/Desktop/Research/URECA2025/organized_version/data \
    --hidden_channels 64 \
    --num_heads 8 \
    --num_layers 2 \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --epochs 200

# Example 1: Just construct a graph from expression matrix
# Input: test_features.npy -> Output: test_graph.csv
python pipeline/construct_graph_leiden.py \
    --input_file data/Features/val_features.npy \
    --output_file data/Graphs/val_graph.csv \
    --k 15 \
    --use_pca \
    --n_pcs 50

# Example 2: Run full pipeline with clustering and data preparation
python pipeline/full_pipeline_leiden.py \
    --input_file data/expression_matrix.npy \
    --output_dir results/full_pipeline_output \
    --k 15 \
    --resolution 1.0 \
    --use_pca \
    --n_pcs 50 \
    --train_ratio 0.6 \
    --val_ratio 0.2

# Example 3: Construct graph without PCA or scaling
python pipeline/construct_graph_leiden.py \
    --input_file data/raw_features.npy \
    --output_file data/raw_graph.csv \
    --k 20 \
    --no_scale \
    --no_prune

# Example 4: Use cosine distance metric
python pipeline/construct_graph_leiden.py \
    --input_file data/normalized_features.npy \
    --output_file data/cosine_graph.csv \
    --k 15 \
    --distance_metric cosine \
    --use_pca \
    --n_pcs 30
