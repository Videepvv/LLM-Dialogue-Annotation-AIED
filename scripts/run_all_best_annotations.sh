#!/bin/bash
# Run full annotations for all models with completed ablations using their best configs
# Usage: nohup ./scripts/run_all_best_annotations.sh [device] > results/annotations.log 2>&1 &

DEVICE=${1:-cuda:1}
MODELS_DIR="/data/open-weight-llms/models"

echo "============================================================"
echo "Running Full Annotations for All Models with Completed Ablations"
echo "Device: $DEVICE"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/CPS results/DELI results/TalkMoves

# Function to run annotations for a model with its best config
run_model_annotations() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo ""
    echo "############################################################"
    echo "Starting annotations for: $MODEL"
    echo "CPS history_window: $CPS_HISTORY"
    echo "Time: $(date)"
    echo "############################################################"
    
    # Skip if CPS result already exists and is complete
    if [ -f "results/CPS/cps_${MODEL}.json" ]; then
        CPS_COUNT=$(python3 -c "import json; print(len(json.load(open('results/CPS/cps_${MODEL}.json'))))" 2>/dev/null || echo "0")
        if [ "$CPS_COUNT" -ge 2400 ]; then
            echo "CPS already complete for $MODEL ($CPS_COUNT samples), skipping..."
        else
            echo "[1/3] CPS Annotation for $MODEL..."
            python3 src/run_cps.py \
                --model $MODEL \
                --n_samples 2500 \
                --history_window $CPS_HISTORY \
                --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
                --models_dir $MODELS_DIR \
                --results_dir ./results/CPS \
                --device $DEVICE
        fi
    else
        echo "[1/3] CPS Annotation for $MODEL..."
        python3 src/run_cps.py \
            --model $MODEL \
            --n_samples 2500 \
            --history_window $CPS_HISTORY \
            --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
            --models_dir $MODELS_DIR \
            --results_dir ./results/CPS \
            --device $DEVICE
    fi
    
    # Skip if DELI result already exists and is complete
    if [ -f "results/DELI/deli_${MODEL}.json" ]; then
        DELI_COUNT=$(python3 -c "import json; print(len(json.load(open('results/DELI/deli_${MODEL}.json'))))" 2>/dev/null || echo "0")
        if [ "$DELI_COUNT" -ge 13000 ]; then
            echo "DELI already complete for $MODEL ($DELI_COUNT samples), skipping..."
        else
            echo "[2/3] DELI Annotation for $MODEL..."
            python3 src/run_deli.py \
                --model $MODEL \
                --n_samples 14000 \
                --data_path ./data/GoldenData/DeliData/delidata_train.csv \
                --models_dir $MODELS_DIR \
                --results_dir ./results/DELI \
                --device $DEVICE
        fi
    else
        echo "[2/3] DELI Annotation for $MODEL..."
        python3 src/run_deli.py \
            --model $MODEL \
            --n_samples 14000 \
            --data_path ./data/GoldenData/DeliData/delidata_train.csv \
            --models_dir $MODELS_DIR \
            --results_dir ./results/DELI \
            --device $DEVICE
    fi
    
    # TalkMoves - FIXED PATH to GoldenData
    echo "[3/3] TalkMoves Annotation for $MODEL..."
    python3 src/run_talkmoves.py \
        --model $MODEL \
        --data_type teacher \
        --split train \
        --n_samples 150000 \
        --data_dir ./data/GoldenData/TalkMoves/data \
        --models_dir $MODELS_DIR \
        --results_dir ./results/TalkMoves \
        --device $DEVICE
    
    echo "Completed $MODEL at $(date)"
}

# Best configs from ablation results (all 8 models completed):
# Model            | CPS best h | DELI best | TalkMoves best
# llama-8b         | 5          | h10 coton | h1 coton
# qwen-7b          | 3          | h10 cotoff| h1 cotoff
# qwen-14b         | 5          | h10 coton | h1 coton
# qwen-32b         | 5          | h10 coton | h1 coton
# gemma-9b         | 5          | h10 coton | h1 coton
# mistral-7b       | 5          | h10 coton | h1 coton
# mistral-nemo     | 5          | h10 coton | h1 coton
# deepseek-v2-lite | 5          | h10 coton | h1 coton

# Run all models with their best CPS history config
run_model_annotations "llama-8b" 5
run_model_annotations "qwen-7b" 3
run_model_annotations "qwen-14b" 5
run_model_annotations "qwen-32b" 5
run_model_annotations "gemma-9b" 5
run_model_annotations "mistral-7b" 5
run_model_annotations "mistral-nemo" 5
run_model_annotations "deepseek-v2-lite" 5

echo ""
echo "============================================================"
echo "All Model Annotations Complete!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Running final kappa analysis..."
python3 src/calculate_kappa.py --all --results_dir results
