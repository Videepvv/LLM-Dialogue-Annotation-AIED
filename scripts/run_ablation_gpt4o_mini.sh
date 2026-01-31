#!/bin/bash
# GPT-4o-mini Ablation Tests - All Datasets
# Tests history windows: 0, 3, 5, 10, -1 (all)
# Tests CoT: on (with reasoning), off (just answer)
#
# Usage: 
#   export OPENAI_API_KEY="your-key-here"
#   ./scripts/run_ablation_gpt4o_mini.sh
#
# Or pass key as argument:
#   ./scripts/run_ablation_gpt4o_mini.sh sk-your-key-here

API_KEY=${1:-$OPENAI_API_KEY}
MODEL="gpt-4o-mini"
N_SAMPLES=200  # Use 200 samples for ablation (balance cost vs. accuracy)

if [ -z "$API_KEY" ]; then
    echo "ERROR: No API key provided!"
    echo "Usage: export OPENAI_API_KEY='your-key' && ./scripts/run_ablation_gpt4o_mini.sh"
    echo "   or: ./scripts/run_ablation_gpt4o_mini.sh sk-your-key-here"
    exit 1
fi

echo "============================================================"
echo "GPT-4o-mini Ablation Tests"
echo "Samples per config: $N_SAMPLES"
echo "Started: $(date)"
echo "============================================================"

mkdir -p results/ablation

# History windows to test
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
    
    # Rename output to include config
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

# TalkMoves only tests h=0 and h=1 (limited context in data)
for H in 0 1; do
    echo ""
    echo "--- TalkMoves | history=$H | $(date) ---"
    python3 src/run_talkmoves_api.py \
        --model $MODEL \
        --api_key "$API_KEY" \
        --data_type teacher \
        --n_samples $N_SAMPLES \
        --data_dir ./data/ablation \
        --results_dir ./results/ablation \
        --delay 0.1
    
    mv results/ablation/talkmoves_teacher_${MODEL}.json results/ablation/talkmoves_teacher_${MODEL}_h${H}.json 2>/dev/null
done

echo ""
echo "============================================================"
echo "GPT-4o-mini Ablation Complete!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results saved in: results/ablation/"
echo "Files: cps_${MODEL}_h*.json, deli_${MODEL}_h*.json, talkmoves_teacher_${MODEL}_h*.json"
echo ""
echo "Run kappa analysis with:"
echo "  python3 src/calculate_kappa.py --all --results_dir results/ablation"
