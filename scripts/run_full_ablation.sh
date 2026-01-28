#!/bin/bash
# Run full ablation suite for a specific model
# Usage: ./scripts/run_full_ablation.sh <model_name> [device]
# Example: ./scripts/run_full_ablation.sh llama-8b cuda:0

MODEL=$1
DEVICE=${2:-auto}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name> [device]"
    echo "Available models: llama-3b, llama-8b, qwen-7b, mistral-7b, etc."
    exit 1
fi

echo "============================================================"
echo "Starting Full Ablation Suite for Model: $MODEL"
echo "Device: $DEVICE"
echo "============================================================"

# Ensure output directory exists
mkdir -p results/ablation

# 1. CPS Ablation (8 configs)
echo -e "\n[1/3] Running CPS Ablation..."
python3 src/run_ablation_cps.py --model $MODEL --config all --device $DEVICE

# 2. DELI Ablation (8 configs)
echo -e "\n[2/3] Running DELI Ablation..."
python3 src/run_ablation_deli.py --model $MODEL --config all --device $DEVICE

# 3. TalkMoves Ablation (4 configs)
echo -e "\n[3/3] Running TalkMoves (Teacher) Ablation..."
python3 src/run_ablation_talkmoves.py --model $MODEL --data_type teacher --config all --device $DEVICE

echo "============================================================"
echo "Ablation Suite Completed for $MODEL"
echo "Results saved in: results/ablation/"
echo "============================================================"
