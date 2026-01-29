#!/bin/bash
# 训练脚本
# Usage: bash train.sh GPU_ID SEED BASE_MODEL DATASET
# Example: bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct luxury_beauty

set -e

GPU_ID=$1
SEED=$2
BASE_MODEL=$3
DATASET=${4:-luxury_beauty}

if [ -z "$GPU_ID" ] || [ -z "$SEED" ] || [ -z "$BASE_MODEL" ]; then
    echo "Usage: bash train.sh GPU_ID SEED BASE_MODEL [DATASET]"
    echo "Example: bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct luxury_beauty"
    echo "DATASET options: luxury_beauty, software, video_games"
    exit 1
fi

TRAIN_DATA="./data/train_${DATASET}.json"
VAL_DATA="./data/valid_${DATASET}.json"
OUTPUT_DIR="./output/${DATASET}_seed${SEED}"

echo "Training: GPU=$GPU_ID, Seed=$SEED, Dataset=$DATASET"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found"
    exit 1
fi

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$GPU_ID python -u finetune.py \
    --base_model $BASE_MODEL \
    --train_data_path $TRAIN_DATA \
    --val_data_path $VAL_DATA \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 50 \
    --learning_rate 3e-5 \
    --cutoff_len 1024 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]'

echo "Training completed: $OUTPUT_DIR"