#!/bin/bash
# Run annotations for models starting from qwen-7b (for parallel execution)
# Usage: nohup ./scripts/run_annotations_gpu1.sh cuda:1 > results/annotations_gpu1.log 2>&1 &

DEVICE=${1:-cuda:1}
MODELS_DIR="/data/open-weight-llms/models"

run_model_annotations() {
    local MODEL=$1
    local CPS_HISTORY=$2
    
    echo "############################################################"
    echo "Starting: $MODEL | Time: $(date)"
    echo "############################################################"
    
    # CPS
    if [ -f "results/CPS/cps_${MODEL}.json" ]; then
        CPS_COUNT=$(python3 -c "import json; print(len(json.load(open('results/CPS/cps_${MODEL}.json'))))" 2>/dev/null || echo "0")
        if [ "$CPS_COUNT" -ge 2400 ]; then
            echo "CPS already complete ($CPS_COUNT samples), skipping..."
        else
            python3 src/run_cps.py --model $MODEL --n_samples 2500 --history_window $CPS_HISTORY \
                --data_path ./data/GoldenData/WTD/OOCPS_aied.csv --models_dir $MODELS_DIR \
                --results_dir ./results/CPS --device $DEVICE
        fi
    else
        python3 src/run_cps.py --model $MODEL --n_samples 2500 --history_window $CPS_HISTORY \
            --data_path ./data/GoldenData/WTD/OOCPS_aied.csv --models_dir $MODELS_DIR \
            --results_dir ./results/CPS --device $DEVICE
    fi
    
    # DELI
    if [ -f "results/DELI/deli_${MODEL}.json" ]; then
        DELI_COUNT=$(python3 -c "import json; print(len(json.load(open('results/DELI/deli_${MODEL}.json'))))" 2>/dev/null || echo "0")
        if [ "$DELI_COUNT" -ge 13000 ]; then
            echo "DELI already complete ($DELI_COUNT samples), skipping..."
        else
            python3 src/run_deli.py --model $MODEL --n_samples 14000 \
                --data_path ./data/GoldenData/DeliData/delidata_train.csv --models_dir $MODELS_DIR \
                --results_dir ./results/DELI --device $DEVICE
        fi
    else
        python3 src/run_deli.py --model $MODEL --n_samples 14000 \
            --data_path ./data/GoldenData/DeliData/delidata_train.csv --models_dir $MODELS_DIR \
            --results_dir ./results/DELI --device $DEVICE
    fi
    
    # TalkMoves
    python3 src/run_talkmoves.py --model $MODEL --data_type teacher --split train --n_samples 150000 \
        --data_dir ./data/GoldenData/TalkMoves/data --models_dir $MODELS_DIR \
        --results_dir ./results/TalkMoves --device $DEVICE
    
    echo "Completed $MODEL at $(date)"
}

# Run second half of models
run_model_annotations "qwen-32b" 5
run_model_annotations "gemma-9b" 5
run_model_annotations "mistral-7b" 5
run_model_annotations "mistral-nemo" 5
run_model_annotations "deepseek-v2-lite" 5

echo "GPU1 batch complete at $(date)"
