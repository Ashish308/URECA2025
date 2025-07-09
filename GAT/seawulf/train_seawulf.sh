#!/bin/bash
#SBATCH --job-name=gat_train
#SBATCH --output=gat_%j.out
#SBATCH --error=gat_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB

module load python/3.9
module load cuda/11.7

python train_ddp.py