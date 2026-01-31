#!/bin/bash
# Run script for SHANNON - LARGE MODELS ONLY (2x RTX PRO 6000, 98GB each)
# Keep Shannon for: llama-70b, qwen-72b, mixtral-8x7b, gpt-oss-120b, gpt-oss-20b
#
# Usage: ./scripts/run_shannon.sh [DEVICE]
# Example: ./scripts/run_shannon.sh cuda:0

MODELS_DIR="/data/open-weight-llms/models"
DEVICE=${1:-"cuda:0"}

echo "============================================================"
echo "SHANNON Large Model Script (RTX PRO 6000)"
echo "Device: $DEVICE"
echo "Started: $(date)"
echo "============================================================"

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

# LARGE MODELS - Shannon only (using h=5 as default)
run_model "llama-70b" 5
run_model "qwen-72b" 5
run_model "mixtral-8x7b" 5
run_model "gpt-oss-20b" 5
# run_model "gpt-oss-120b" 5  # Needs 2 GPUs, may need special handling

echo ""
echo "============================================================"
echo "SHANNON Large Models Complete! $(date)"
echo "============================================================"
