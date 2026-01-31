#!/bin/bash
# Run annotations for large models (70B+ params)
# Usage: nohup ./scripts/run_large_models.sh [device] > results/annotations_large.log 2>&1 &

DEVICE=${1:-cuda:0}
MODELS_DIR="/data/open-weight-llms/models"

run_model() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo "############################################################"
    echo "Starting: $MODEL | Time: $(date)"
    echo "############################################################"
    
    # CPS
    python3 src/run_cps.py --model $MODEL --n_samples 2500 --history_window $CPS_HISTORY \
        --data_path ./data/GoldenData/WTD/OOCPS_aied.csv --models_dir $MODELS_DIR \
        --results_dir ./results/CPS --device $DEVICE
    
    # DELI
    python3 src/run_deli.py --model $MODEL --n_samples 14000 \
        --data_path ./data/GoldenData/DeliData/delidata_train.csv --models_dir $MODELS_DIR \
        --results_dir ./results/DELI --device $DEVICE
    
    # TalkMoves
    python3 src/run_talkmoves.py --model $MODEL --data_type teacher --split train --n_samples 150000 \
        --data_dir ./data/GoldenData/TalkMoves/data --models_dir $MODELS_DIR \
        --results_dir ./results/TalkMoves --device $DEVICE
    
    echo "Completed $MODEL at $(date)"
}

# Large models - using h=5 as default (best from ablation of similar models)
run_model "llama-70b" 5
run_model "qwen-72b" 5
run_model "mixtral-8x7b" 5

echo "Large models complete at $(date)"
