#!/bin/bash
# Usage: bash train.sh GPU_ID SEED BASE_MODEL CANDIDATES
# Example: bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct 20

GPU_ID=$1
SEED=$2
BASE_MODEL=$3  # e.g., "meta-llama/Llama-3.2-3B-Instruct" or "facebook/opt-6.7b"
CANDIDATES=$4  # 20 or 100

OUTPUT_DIR="./output/model_${CANDIDATES}_seed${SEED}"
TRAIN_DATA="./data/train_${CANDIDATES}.json"
VAL_DATA="./data/valid_${CANDIDATES}.json"
INSTRUCTION_MODEL=""  # Set to your instruction-tuned checkpoint if available

mkdir -p $OUTPUT_DIR

echo "Training with:"
echo "  GPU: $GPU_ID"
echo "  Seed: $SEED"
echo "  Base Model: $BASE_MODEL"
echo "  Candidates: $CANDIDATES"
echo "  Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID python -u finetune_candidates.py \
    --base_model $BASE_MODEL \
    --train_data_path $TRAIN_DATA \
    --val_data_path $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --sample 256 \
    --seed $SEED

echo "Training completed. Model saved to $OUTPUT_DIR"