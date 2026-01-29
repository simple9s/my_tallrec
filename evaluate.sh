#!/bin/bash
# 评估脚本
# Usage: bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS DATASET [CANDIDATES]
# Example: bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 20

set -e

GPU_ID=$1
BASE_MODEL=$2
LORA_WEIGHTS=$3
DATASET=${4:-luxury_beauty}
CANDIDATES=${5:-20}

if [ -z "$GPU_ID" ] || [ -z "$BASE_MODEL" ] || [ -z "$LORA_WEIGHTS" ]; then
    echo "Usage: bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS [DATASET] [CANDIDATES]"
    echo "Example: bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/luxury_beauty_seed42 luxury_beauty 20"
    echo "DATASET: luxury_beauty, software, video_games"
    echo "CANDIDATES: 20 or 100"
    exit 1
fi

TEST_DATA="./data/test_${DATASET}_${CANDIDATES}.json"
RESULT_FILE="${LORA_WEIGHTS}/results_${CANDIDATES}.json"

echo "Evaluating: Dataset=$DATASET, Candidates=$CANDIDATES"

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: $TEST_DATA not found"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python -u evaluate.py \
    --base_model $BASE_MODEL \
    --lora_weights $LORA_WEIGHTS \
    --test_data_path $TEST_DATA \
    --result_json_path $RESULT_FILE

echo "Results: $RESULT_FILE"