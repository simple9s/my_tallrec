#!/bin/bash
# Usage: bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS TEST_DATA
# Example: bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/model_20_seed42 ./data/test_20.json

GPU_ID=$1
BASE_MODEL=$2
LORA_WEIGHTS=$3
TEST_DATA=$4

RESULT_FILE="${LORA_WEIGHTS}/results.json"

echo "Evaluating:"
echo "  GPU: $GPU_ID"
echo "  Base Model: $BASE_MODEL"
echo "  LoRA Weights: $LORA_WEIGHTS"
echo "  Test Data: $TEST_DATA"

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_candidates.py \
    --base_model $BASE_MODEL \
    --lora_weights $LORA_WEIGHTS \
    --test_data_path $TEST_DATA \
    --result_json_data $RESULT_FILE \
    --batch_size 8

echo "Results saved to $RESULT_FILE"