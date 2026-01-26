#!/bin/bash
# Usage: bash train.sh GPU_ID SEED BASE_MODEL CANDIDATES [DATASET]
# Example: bash train.sh 0 42 meta-llama/Llama-3.2-3B-Instruct 20 movie

GPU_ID=$1
SEED=$2
BASE_MODEL=$3
CANDIDATES=$4
DATASET=${5:-movie}  # movie, book, luxury_beauty, software

# Set checkpoint path here for resuming training
# Example: INSTRUCTION_MODEL="./output/movie_20_seed42/checkpoint-1000"
INSTRUCTION_MODEL=""

if [ "$DATASET" = "movie" ]; then
    TRAIN_DATA="./data/train_${CANDIDATES}.json"
    VAL_DATA="./data/valid_${CANDIDATES}.json"
elif [ "$DATASET" = "book" ]; then
    TRAIN_DATA="./data/train_book_${CANDIDATES}.json"
    VAL_DATA="./data/valid_book_${CANDIDATES}.json"
elif [ "$DATASET" = "luxury_beauty" ]; then
    TRAIN_DATA="./data/train_luxury_beauty_${CANDIDATES}.json"
    VAL_DATA="./data/valid_luxury_beauty_${CANDIDATES}.json"
elif [ "$DATASET" = "software" ]; then
    TRAIN_DATA="./data/train_software_${CANDIDATES}.json"
    VAL_DATA="./data/valid_software_${CANDIDATES}.json"
fi

OUTPUT_DIR="./output/${DATASET}_${CANDIDATES}_seed${SEED}"

mkdir -p $OUTPUT_DIR

echo "Training with:"
echo "  GPU: $GPU_ID"
echo "  Seed: $SEED"
echo "  Base Model: $BASE_MODEL"
echo "  Dataset: $DATASET"
echo "  Candidates: $CANDIDATES"
echo "  Train Data: $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"

if [ -z "$INSTRUCTION_MODEL" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u finetune_candidates.py \
        --base_model $BASE_MODEL \
        --train_data_path $TRAIN_DATA \
        --val_data_path $VAL_DATA \
        --output_dir $OUTPUT_DIR \
        --batch_size 128 \
        --micro_batch_size 8 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,v_proj]' \
        --train_on_inputs \
        --group_by_length \
        --sample -1 \
        --seed $SEED
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u finetune_candidates.py \
        --base_model $BASE_MODEL \
        --train_data_path $TRAIN_DATA \
        --val_data_path $VAL_DATA \
        --output_dir $OUTPUT_DIR \
        --batch_size 128 \
        --micro_batch_size 8 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,v_proj]' \
        --train_on_inputs \
        --group_by_length \
        --sample -1 \
        --seed $SEED \
        --resume_from_checkpoint $INSTRUCTION_MODEL
fi

echo "Training completed. Model saved to $OUTPUT_DIR"