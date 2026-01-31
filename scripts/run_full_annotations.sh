#!/bin/bash
# Run full annotations using best configs from ablation results
# Usage: ./scripts/run_full_annotations.sh <model> [device]
# Example: ./scripts/run_full_annotations.sh llama-8b cuda:1

MODEL=$1
DEVICE=${2:-auto}
MODELS_DIR="/data/open-weight-llms/models"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name> [device]"
    echo "Available models: llama-8b, qwen-7b, etc."
    exit 1
fi

echo "============================================================"
echo "Running Full Annotations for: $MODEL"
echo "Device: $DEVICE"
echo "Using best configs from ablation results"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/CPS
mkdir -p results/DELI
mkdir -p results/TalkMoves

# Best configs from ablation (based on completed results):
# - CPS: h5_cotoff gave best kappa for llama-8b (0.065)
# - DELI: h10_coton gave best kappa for llama-8b (0.618)  
# - TalkMoves: h1_coton gave best kappa for llama-8b (0.211)

# Note: The annotation scripts use CoT by default (reasoning in output)
# History window is the main tunable parameter

# 1. CPS Annotation (full dataset)
# Best config: history_window=5
echo ""
echo "############################################################"
echo "[1/3] Running CPS Full Annotation for $MODEL"
echo "Config: history_window=5"
echo "Time: $(date)"
echo "############################################################"

python3 src/run_cps.py \
    --model $MODEL \
    --n_samples 2500 \
    --history_window 5 \
    --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
    --models_dir $MODELS_DIR \
    --results_dir ./results/CPS \
    --device $DEVICE

# 2. DELI Annotation (full dataset)
# Best config: history context ~10 turns (hardcoded in script, but we can modify)
echo ""
echo "############################################################"
echo "[2/3] Running DELI Full Annotation for $MODEL"
echo "Time: $(date)"
echo "############################################################"

python3 src/run_deli.py \
    --model $MODEL \
    --n_samples 14000 \
    --data_path ./data/GoldenData/DeliData/delidata_train.csv \
    --models_dir $MODELS_DIR \
    --results_dir ./results/DELI \
    --device $DEVICE

# 3. TalkMoves Annotation (teacher, full dataset)
echo ""
echo "############################################################"
echo "[3/3] Running TalkMoves Teacher Full Annotation for $MODEL"
echo "Time: $(date)"
echo "############################################################"

python3 src/run_talkmoves.py \
    --model $MODEL \
    --data_type teacher \
    --split train \
    --n_samples 150000 \
    --data_dir ./data/TalkMoves/data \
    --models_dir $MODELS_DIR \
    --results_dir ./results/TalkMoves \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Full Annotations Completed for $MODEL"
echo "Finished: $(date)"
echo "============================================================"
echo "Results saved in:"
echo "  - results/CPS/cps_${MODEL}.json"
echo "  - results/DELI/deli_${MODEL}.json"
echo "  - results/TalkMoves/talkmoves_teacher_${MODEL}.json"
