#!/bin/bash
# Usage: bash evaluate.sh GPU_ID BASE_MODEL LORA_WEIGHTS CANDIDATES [DATASET]
# Example: bash evaluate.sh 0 meta-llama/Llama-3.2-3B-Instruct ./output/movie_20_seed42 20 movie

GPU_ID=$1
BASE_MODEL=$2
LORA_WEIGHTS=$3
CANDIDATES=$4
DATASET=${5:-movie}  # movie, book, luxury_beauty, software

if [ "$DATASET" = "movie" ]; then
    TEST_DATA="./data/test_${CANDIDATES}.json"
elif [ "$DATASET" = "book" ]; then
    TEST_DATA="./data/test_book_${CANDIDATES}.json"
elif [ "$DATASET" = "luxury_beauty" ]; then
    TEST_DATA="./data/test_luxury_beauty_${CANDIDATES}.json"
elif [ "$DATASET" = "software" ]; then
    TEST_DATA="./data/test_software_${CANDIDATES}.json"
fi

RESULT_FILE="${LORA_WEIGHTS}/results.json"

echo "Evaluating:"
echo "  GPU: $GPU_ID"
echo "  Base Model: $BASE_MODEL"
echo "  LoRA Weights: $LORA_WEIGHTS"
echo "  Dataset: $DATASET"
echo "  Test Data: $TEST_DATA"

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_candidates.py \
    --base_model $BASE_MODEL \
    --lora_weights $LORA_WEIGHTS \
    --test_data_path $TEST_DATA \
    --result_json_data $RESULT_FILE \
    --batch_size 8

echo "Results saved to $RESULT_FILE"