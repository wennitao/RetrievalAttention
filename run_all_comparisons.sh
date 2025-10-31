#!/bin/bash
# 对比三种query平滑方法：
# 1. Baseline (无校正)
# 2. RoPE correction only
# 3. RoPE correction + AR prediction

set -e

echo "=========================================="
echo "Query Smoothing Methods Comparison"
echo "=========================================="

cd /mnt/data/RetrievalAttention

conda activate retroinfer

# Method 1: Baseline

echo ""
echo "[1/3] Running BASELINE (no correction)..."
export QUERY_SIM_LOG=1
export QUERY_SIM_SUMMARY_PATH=logs/query_sim_baseline.csv
export ROPE_CORRECTION=0
export ENABLE_AR_PREDICTION=0

python -u throughput_eval/test.py \
    --attn_type RetroInfer \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --task_name NIAH \
    --context_len 60000 \
    --batch_size 1 \
    --dtype fp16 \
    --device cuda:0

echo "✓ Baseline saved to logs/query_sim_baseline.csv"

# Method 2: RoPE correction only
echo ""
echo "[2/3] Running RoPE CORRECTION..."
export QUERY_SIM_SUMMARY_PATH=logs/query_sim_rope_corrected.csv
export ROPE_CORRECTION=1
export ENABLE_AR_PREDICTION=0

python -u throughput_eval/test.py \
    --attn_type RetroInfer \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --task_name NIAH \
    --context_len 60000 \
    --batch_size 1 \
    --dtype fp16 \
    --device cuda:0

echo "✓ RoPE correction saved to logs/query_sim_rope_corrected.csv"

# Method 3: RoPE + AR prediction
echo ""
echo "[3/3] Running RoPE + AR PREDICTION..."
export QUERY_SIM_SUMMARY_PATH=logs/query_sim_ar_predicted.csv
export ROPE_CORRECTION=1
export ENABLE_AR_PREDICTION=1
export AR_ALPHA=1.0
export AR_BETA=0.1

python -u throughput_eval/test.py \
    --attn_type RetroInfer \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --task_name NIAH \
    --context_len 60000 \
    --batch_size 1 \
    --dtype fp16 \
    --device cuda:0

echo "✓ AR prediction saved to logs/query_sim_ar_predicted.csv"

# Compare all three methods
echo ""
echo "=========================================="
echo "Comparison Results"
echo "=========================================="

echo ""
echo "Baseline vs RoPE Correction:"
python compare_query_sim.py --original logs/query_sim_baseline.csv --corrected logs/query_sim_rope_corrected.csv

echo ""
echo "RoPE Correction vs RoPE + AR:"
python compare_query_sim.py --original logs/query_sim_rope_corrected.csv --corrected logs/query_sim_ar_predicted.csv

echo ""
echo "Baseline vs RoPE + AR (overall):"
python compare_query_sim.py --original logs/query_sim_baseline.csv --corrected logs/query_sim_ar_predicted.csv

echo ""
echo "=========================================="
echo "✓ All comparisons complete!"
echo "=========================================="
