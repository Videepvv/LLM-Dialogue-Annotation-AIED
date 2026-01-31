#!/bin/bash
# Gemini Ablation Tests - All Datasets
# Tests history windows: 0, 3, 5, 10
#
# Usage: ./scripts/run_ablation_gemini.sh YOUR_API_KEY

API_KEY=${1:-$GOOGLE_API_KEY}
MODEL="gemini-2.0-flash"
N_SAMPLES=200

if [ -z "$API_KEY" ]; then
    echo "ERROR: No API key provided!"
    exit 1
fi

echo "============================================================"
echo "Gemini 2.0 Flash Ablation Tests"
echo "Model: $MODEL"
echo "Samples per config: $N_SAMPLES"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/ablation

HISTORY_WINDOWS=(0 3 5 10)

echo ""
echo "############################################################"
echo "[1/3] CPS Ablation"
echo "############################################################"

for H in "${HISTORY_WINDOWS[@]}"; do
    echo ""
    echo "--- CPS | history=$H | $(date) ---"
    python3 src/run_cps_api.py \
        --model $MODEL \
        --api_key "$API_KEY" \
        --n_samples $N_SAMPLES \
        --history_window $H \
        --data_path ./data/ablation/cps_ablation_198.csv \
        --results_dir ./results/ablation \
        --delay 0.1
    
    mv results/ablation/cps_${MODEL}.json results/ablation/cps_${MODEL}_h${H}.json 2>/dev/null
done

echo ""
echo "############################################################"
echo "[2/3] DELI Ablation"
echo "############################################################"

for H in "${HISTORY_WINDOWS[@]}"; do
    echo ""
    echo "--- DELI | history=$H | $(date) ---"
    python3 src/run_deli_api.py \
        --model $MODEL \
        --api_key "$API_KEY" \
        --n_samples $N_SAMPLES \
        --data_path ./data/ablation/deli_ablation_200.csv \
        --results_dir ./results/ablation \
        --delay 0.1
    
    mv results/ablation/deli_${MODEL}.json results/ablation/deli_${MODEL}_h${H}.json 2>/dev/null
done

echo ""
echo "############################################################"
echo "[3/3] TalkMoves Ablation"
echo "############################################################"

echo ""
echo "--- TalkMoves | $(date) ---"
python3 src/run_talkmoves_api.py \
    --model $MODEL \
    --api_key "$API_KEY" \
    --data_type teacher \
    --n_samples $N_SAMPLES \
    --data_dir ./data/GoldenData/TalkMoves/data \
    --results_dir ./results/ablation \
    --delay 0.1

echo ""
echo "============================================================"
echo "Gemini Ablation Complete!"
echo "Finished: $(date)"
echo "============================================================"
