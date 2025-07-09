#!/bin/bash

# Auto-detect hardware and install correct environment
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon (M1/M2) - MPS setup
    echo "Detected Apple Silicon (MPS)"
    ENV_FILE="environment_mps.yml"
    ENV_NAME="sc_gat_mps"
elif [[ $(nvidia-smi 2>/dev/null) ]] && [[ $(uname -m) == "x86_64" ]]; then
    # NVIDIA GPU - CUDA setup
    echo "Detected NVIDIA GPU (CUDA)"
    ENV_FILE="environment_cuda.yml"
    ENV_NAME="sc_gat_cuda"
else
    # CPU-only fallback
    echo "No compatible GPU detected - CPU-only mode"
    ENV_FILE="environment_cpu.yml"
    ENV_NAME="sc_gat_cpu"
fi

# Create Conda environment
echo "Creating '$ENV_NAME' environment..."
conda env create -f $ENV_FILE -n $ENV_NAME
