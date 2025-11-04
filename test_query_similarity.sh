#!/bin/bash

# 测试新的 query similarity logging 功能
# 这个脚本会启用日志记录并运行简单测试

# 设置环境变量
export LOG_QUERY_SIMILARITY=1
export QUERY_SIMILARITY_OUTPUT="logs/query_similarity_new.csv"
export QUERY_SIMILARITY_MAX_PAIRS=50  # 只记录前50对，避免文件过大

# 创建日志目录
mkdir -p logs

echo "============================================"
echo "Testing Query Similarity Logging"
echo "Output file: $QUERY_SIMILARITY_OUTPUT"
echo "============================================"

# 运行测试
python simple_test.py \
    --batch_size 1 \
    --gen_len 50 \
    --device cuda:0 \
    --dtype fp16 \
    --attn_type RetroInfer \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k

echo ""
echo "============================================"
echo "Test completed!"
echo "Check the output CSV file at: $QUERY_SIMILARITY_OUTPUT"
echo "============================================"
echo ""
echo "Sample of the CSV file:"
head -n 10 "$QUERY_SIMILARITY_OUTPUT"
