#!/bin/bash
# Run script for CARNAP (2x RTX 6000, ~48GB each)
# Models that fit: mistral-nemo, deepseek-v2-lite, gemma-27b (if downloaded)
#
# Usage: ./scripts/run_carnap.sh [MODELS_DIR] [DEVICE]
# Example: ./scripts/run_carnap.sh /path/to/models cuda:0

MODELS_DIR=${1:-"/data/open-weight-llms/models"}
DEVICE=${2:-"cuda:0"}

echo "============================================================"
echo "CARNAP Annotation Script (RTX 6000)"
echo "Models dir: $MODELS_DIR"
echo "Device: $DEVICE"
echo "Started: $(date)"
echo "============================================================"

# Skip function
skip_if_done() {
    local FILE=$1
    local MIN_SAMPLES=$2
    if [ -f "$FILE" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$FILE'))))" 2>/dev/null || echo "0")
        if [ "$COUNT" -ge "$MIN_SAMPLES" ]; then
            echo "  SKIP: $FILE already has $COUNT samples"
            return 0
        fi
    fi
    return 1
}

run_model() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo ""
    echo "############################################################"
    echo "Starting: $MODEL | $(date)"
    echo "############################################################"
    
    # CPS
    if ! skip_if_done "results/CPS/cps_${MODEL}.json" 2400; then
        echo "  Running CPS..."
        python3 src/run_cps.py --model $MODEL --n_samples 2500 --history_window $CPS_HISTORY \
            --data_path ./data/GoldenData/WTD/OOCPS_aied.csv --models_dir $MODELS_DIR \
            --results_dir ./results/CPS --device $DEVICE
    fi
    
    # DELI  
    if ! skip_if_done "results/DELI/deli_${MODEL}.json" 13500; then
        echo "  Running DELI..."
        python3 src/run_deli.py --model $MODEL --n_samples 14000 \
            --data_path ./data/GoldenData/DeliData/delidata_train.csv --models_dir $MODELS_DIR \
            --results_dir ./results/DELI --device $DEVICE
    fi
    
    # TalkMoves
    if ! skip_if_done "results/TalkMoves/talkmoves_teacher_${MODEL}.json" 145000; then
        echo "  Running TalkMoves..."
        python3 src/run_talkmoves.py --model $MODEL --data_type teacher --split train --n_samples 150000 \
            --data_dir ./data/GoldenData/TalkMoves/data --models_dir $MODELS_DIR \
            --results_dir ./results/TalkMoves --device $DEVICE
    fi
    
    echo "Completed $MODEL at $(date)"
}

# Best configs from ablation
# Run models that fit on RTX 6000 (~48GB)
run_model "mistral-nemo" 5
run_model "deepseek-v2-lite" 5

# Uncomment if gemma-27b is downloaded and accessible
# run_model "gemma-27b" 5

echo ""
echo "============================================================"
echo "CARNAP Complete! $(date)"
echo "============================================================"
