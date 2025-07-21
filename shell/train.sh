if [ -z "$1" ]; then
    echo "Error: Please provide a name argument."
    echo "Usage: ./construct_one_graph.sh <name>"
    exit 1
fi

name=$1 

#Example: Train fnn model
python scripts/train.py --model fnn \
    --experiment_name ${name}_fnn_experiment \
    --data_dir /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name} \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --epochs 200

#Example: Train gat model
python scripts/train.py --model gat \
    --experiment_name ${name}_gat_experiment \
    --data_dir /Users/Beta/Desktop/Research/URECA2025/organized_version/data/manual_annotation/${name} \
    --hidden_channels 64 \
    --num_heads 8 \
    --num_layers 2 \
    --learning_rate 0.01 \
    --dropout 0.3 \
    --epochs 200