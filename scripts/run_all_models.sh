#!/bin/bash
# Run inference for all models on all datasets
# Usage: ./run_all_models.sh

set -e

MODELS=("llama-8b" "qwen-7b" "mistral-7b" "gemma-9b" "phi-3.5")
N_SAMPLES=1000  # Adjust as needed

echo "=========================================="
echo "Running inference for all models"
echo "=========================================="

# TalkMoves - Teacher
echo ""
echo ">>> TALKMOVES - TEACHER"
for model in "${MODELS[@]}"; do
    echo "  Running $model..."
    python src/run_talkmoves.py --model $model --data_type teacher --n_samples $N_SAMPLES
done

# TalkMoves - Student
echo ""
echo ">>> TALKMOVES - STUDENT"
for model in "${MODELS[@]}"; do
    echo "  Running $model..."
    python src/run_talkmoves.py --model $model --data_type student --n_samples $N_SAMPLES
done

# DELI
echo ""
echo ">>> DELI"
for model in "${MODELS[@]}"; do
    echo "  Running $model..."
    python src/run_deli.py --model $model --n_samples $N_SAMPLES
done

# CPS
echo ""
echo ">>> CPS"
for model in "${MODELS[@]}"; do
    echo "  Running $model..."
    python src/run_cps.py --model $model --n_samples $N_SAMPLES
done

echo ""
echo "=========================================="
echo "Computing metrics"
echo "=========================================="
python src/calculate_kappa.py --all

echo ""
echo "Done!"
