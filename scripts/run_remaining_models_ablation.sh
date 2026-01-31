#!/bin/bash
# Run full ablation suite for remaining models (excluding llama-8b which is already running)
# Usage: ./scripts/run_remaining_models_ablation.sh [device]

DEVICE=${1:-auto}

# All models except llama-8b (already running) and llama-3.2-11b (incomplete)
MODELS=(
    "qwen-7b"
    "mistral-7b"
    "gemma-9b"
    "deepseek-v2-lite"
    "mistral-nemo"
    "qwen-14b"
    "mixtral-8x7b"
    "qwen-32b"
    "llama-70b"
    "qwen-72b"
)

echo "============================================================"
echo "Running Full Ablation for Remaining Models"
echo "Device: $DEVICE"
echo "Models: ${MODELS[@]}"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/ablation

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "############################################################"
    echo "Starting ablation for: $MODEL"
    echo "Time: $(date)"
    echo "############################################################"
    
    # 1. CPS Ablation
    echo -e "\n[1/3] Running CPS Ablation for $MODEL..."
    python3 src/run_ablation_cps.py --model $MODEL --config all --device $DEVICE
    
    # 2. DELI Ablation
    echo -e "\n[2/3] Running DELI Ablation for $MODEL..."
    python3 src/run_ablation_deli.py --model $MODEL --config all --device $DEVICE
    
    # 3. TalkMoves Ablation
    echo -e "\n[3/3] Running TalkMoves (Teacher) Ablation for $MODEL..."
    python3 src/run_ablation_talkmoves.py --model $MODEL --data_type teacher --config all --device $DEVICE
    
    echo ""
    echo "Completed ablation for: $MODEL"
    echo "Time: $(date)"
done

echo ""
echo "============================================================"
echo "ALL ABLATIONS COMPLETED!"
echo "Finished: $(date)"
echo "Results saved in: results/ablation/"
echo "============================================================"
