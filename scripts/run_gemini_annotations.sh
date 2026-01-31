#!/bin/bash
# Gemini 2.0 Flash Full Annotations with Best Configs
# Best from ablation: CPS h=5, DELI h=10
#
# Usage: ./scripts/run_gemini_annotations.sh YOUR_API_KEY

API_KEY=${1:-$GOOGLE_API_KEY}
MODEL="gemini-2.0-flash"

if [ -z "$API_KEY" ]; then
    echo "ERROR: No API key provided!"
    exit 1
fi

echo "============================================================"
echo "Gemini 2.0 Flash Full Annotations"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/CPS results/DELI results/TalkMoves

# CPS - Full dataset with best config (h=5)
echo ""
echo "[1/3] CPS Annotation (n=2500, history=5)"
python3 src/run_cps_api.py \
    --model $MODEL \
    --api_key "$API_KEY" \
    --n_samples 2500 \
    --history_window 5 \
    --data_path ./data/GoldenData/WTD/OOCPS_aied.csv \
    --results_dir ./results/CPS \
    --delay 0.05

# DELI - Full dataset with best config
echo ""
echo "[2/3] DELI Annotation (n=14000)"
python3 src/run_deli_api.py \
    --model $MODEL \
    --api_key "$API_KEY" \
    --n_samples 14000 \
    --data_path ./data/GoldenData/DeliData/delidata_train.csv \
    --results_dir ./results/DELI \
    --delay 0.05

# TalkMoves - Full dataset
echo ""
echo "[3/3] TalkMoves Annotation (teacher, n=150000)"
python3 src/run_talkmoves_api.py \
    --model $MODEL \
    --api_key "$API_KEY" \
    --data_type teacher \
    --split train \
    --n_samples 150000 \
    --data_dir ./data/GoldenData/TalkMoves/data \
    --results_dir ./results/TalkMoves \
    --delay 0.05

echo ""
echo "============================================================"
echo "Gemini Annotations Complete!"
echo "Finished: $(date)"
echo "============================================================"
