#!/bin/bash
# Run script for NAMI (2x RTX 3090, 24GB each)
# Models that fit: gemma-9b, mistral-7b, qwen-14b, qwen-7b, llama-8b
#
# Usage: ./scripts/run_nami.sh [MODELS_DIR] [DEVICE]
# Example: ./scripts/run_nami.sh /path/to/models cuda:0

MODELS_DIR=${1:-"/data/open-weight-llms/models"}
DEVICE=${2:-"cuda:0"}

echo "============================================================"
echo "NAMI Annotation Script (RTX 3090)"
echo "Models dir: $MODELS_DIR"
echo "Device: $DEVICE"
echo "Started: $(date)"
echo "============================================================"

# Skip function - checks if result already exists with sufficient samples
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

# Best configs from ablation (history window for CPS)
# Run models that fit on RTX 3090 (24GB)
run_model "gemma-9b" 5
run_model "mistral-7b" 5
run_model "qwen-14b" 5  # Only needs DELI + TalkMoves

echo ""
echo "============================================================"
echo "NAMI Complete! $(date)"
echo "============================================================"
