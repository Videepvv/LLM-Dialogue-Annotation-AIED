#!/bin/bash
# Run full ablation suite for ALL models on ALL datasets
# Usage: ./scripts/run_all_models_ablation.sh [device]
# Example: ./scripts/run_all_models_ablation.sh cuda:0

DEVICE=${1:-auto}

# All available models (excluding llama-3.2-11b which is incomplete)
MODELS=(
    "llama-8b"
    "llama-70b"
    "qwen-7b"
    "qwen-14b"
    "qwen-32b"
    "qwen-72b"
    "gemma-9b"
    "mistral-7b"
    "mistral-nemo"
    "mixtral-8x7b"
    "deepseek-v2-lite"
)

echo "============================================================"
echo "Running Full Ablation for ALL Models"
echo "Device: $DEVICE"
echo "Models: ${MODELS[@]}"
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
echo "Results saved in: results/ablation/"
echo "============================================================"
