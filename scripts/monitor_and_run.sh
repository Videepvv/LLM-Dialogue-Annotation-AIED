#!/bin/bash
# Monitor script that waits for llama-8b to finish, then runs other models
# Checks every 20 minutes
# Usage: nohup ./scripts/monitor_and_run.sh > results/monitor.log 2>&1 &

DEVICE="cuda:1"
MODELS_DIR="/data/open-weight-llms/models"
CHECK_INTERVAL=1200  # 20 minutes in seconds

echo "============================================================"
echo "Monitor Script Started"
echo "Checking every 20 minutes for llama-8b completion"
echo "Started: $(date)"
echo "============================================================"

# Function to check if llama-8b annotation is running
is_llama_running() {
    pgrep -f "run_cps.py.*llama-8b" > /dev/null 2>&1 || \
    pgrep -f "run_deli.py.*llama-8b" > /dev/null 2>&1 || \
    pgrep -f "run_talkmoves.py.*llama-8b" > /dev/null 2>&1
}

# Function to run annotations for a model
run_model_annotations() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo ""
    echo "############################################################"
    echo "Starting annotations for: $MODEL"
    echo "CPS history_window: $CPS_HISTORY"
    echo "Time: $(date)"
    echo "############################################################"
    
    # CPS
    echo "[1/3] CPS Annotation for $MODEL..."
    python3 src/run_cps.py \
        --model $MODEL \
        --n_samples 2500 \
        --history_window $CPS_HISTORY \
        --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
        --models_dir $MODELS_DIR \
        --results_dir ./results/CPS \
        --device $DEVICE
    
    # DELI
    echo "[2/3] DELI Annotation for $MODEL..."
    python3 src/run_deli.py \
        --model $MODEL \
        --n_samples 14000 \
        --data_path ./data/GoldenData/DeliData/delidata_train.csv \
        --models_dir $MODELS_DIR \
        --results_dir ./results/DELI \
        --device $DEVICE
    
    # TalkMoves
    echo "[3/3] TalkMoves Annotation for $MODEL..."
    python3 src/run_talkmoves.py \
        --model $MODEL \
        --data_type teacher \
        --split train \
        --n_samples 150000 \
        --data_dir ./data/TalkMoves/data \
        --models_dir $MODELS_DIR \
        --results_dir ./results/TalkMoves \
        --device $DEVICE
    
    echo "Completed $MODEL at $(date)"
}

# Wait loop - check every 20 minutes
while is_llama_running; do
    echo "[$(date)] llama-8b still running. Checking again in 20 minutes..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "============================================================"
echo "llama-8b FINISHED at $(date)"
echo "Starting other model annotations..."
echo "============================================================"

# Run qwen-7b (best CPS config: h=3)
run_model_annotations "qwen-7b" 3

# Run qwen-14b (best CPS config: h=5)  
run_model_annotations "qwen-14b" 5

echo ""
echo "============================================================"
echo "All Annotations Complete!"
echo "Finished: $(date)"
echo "============================================================"
