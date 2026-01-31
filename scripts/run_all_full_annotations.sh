#!/bin/bash
# Run full annotations for all completed models using their best ablation configs
# This script reads ablation summaries and uses the best history_window for each model/dataset
# Usage: ./scripts/run_all_full_annotations.sh [device]

DEVICE=${1:-cuda:1}
MODELS_DIR="/data/open-weight-llms/models"

echo "============================================================"
echo "Running Full Annotations for All Completed Models"
echo "Device: $DEVICE"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/CPS results/DELI results/TalkMoves

# Models with completed ablation (add more as they complete)
# Best configs extracted from ablation summaries:
# CPS: h5 generally best
# DELI: h10 with CoT generally best  
# TalkMoves: h1 with CoT generally best

run_model_annotations() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo ""
    echo "############################################################"
    echo "Running annotations for: $MODEL"
    echo "Time: $(date)"
    echo "############################################################"
    
    # CPS (use model-specific best history)
    echo "[1/3] CPS Annotation..."
    python3 src/run_cps.py \
        --model $MODEL \
        --n_samples 2500 \
        --history_window $CPS_HISTORY \
        --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
        --models_dir $MODELS_DIR \
        --results_dir ./results/CPS \
        --device $DEVICE
    
    # DELI (h10 is consistently best)
    echo "[2/3] DELI Annotation..."
    python3 src/run_deli.py \
        --model $MODEL \
        --n_samples 14000 \
        --data_path ./data/GoldenData/DeliData/delidata_train.csv \
        --models_dir $MODELS_DIR \
        --results_dir ./results/DELI \
        --device $DEVICE
    
    # TalkMoves (h1 is consistently best for teacher)
    echo "[3/3] TalkMoves Annotation..."
    python3 src/run_talkmoves.py \
        --model $MODEL \
        --data_type teacher \
        --split train \
        --n_samples 150000 \
        --data_dir ./data/TalkMoves/data \
        --models_dir $MODELS_DIR \
        --results_dir ./results/TalkMoves \
        --device $DEVICE
    
    echo "Completed: $MODEL at $(date)"
}

# Run for each model with their best CPS history config
# llama-8b: already running separately, skip or wait
# qwen-7b: CPS best at h3
# qwen-14b: CPS best at h5

# Check if llama-8b annotation is still running
if pgrep -f "run_cps.py.*llama-8b" > /dev/null || pgrep -f "run_deli.py.*llama-8b" > /dev/null || pgrep -f "run_talkmoves.py.*llama-8b" > /dev/null; then
    echo "llama-8b annotations still in progress, skipping..."
else
    echo "llama-8b annotations complete or not running"
fi

# Run qwen-7b (best CPS: h3)
run_model_annotations "qwen-7b" 3

# Run qwen-14b (best CPS: h5)
run_model_annotations "qwen-14b" 5

echo ""
echo "============================================================"
echo "All Model Annotations Completed"
echo "Finished: $(date)"
echo "============================================================"
