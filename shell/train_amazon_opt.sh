#!/bin/bash

echo "GPU: $1, Seed: $2"
seed=$2

# 修改这些路径
output_dir=out
base_model=facebook/opt-125m
train_data=amazon_beauty_train.json
val_data=data/amazon_beauty_valid.json

# 可选：如果有预训练的instruction模型
# instruction_model=/path/to/alpaca-lora-7B

for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 64
        do
            mkdir -p $output_dir
            echo "lr: $lr, dropout: $dropout, seed: $seed, sample: $sample"
            CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec_opt.py \
                --base_model $base_model \
                --model_type opt \
                --train_data_path $train_data \
                --val_data_path $val_data \
                --output_dir ${output_dir}_${seed}_${sample} \
                --batch_size 128 \
                --micro_batch_size 32 \
                --num_epochs 200 \
                --learning_rate $lr \
                --cutoff_len 512 \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout $dropout \
                --lora_target_modules '[q_proj,v_proj]' \
                --train_on_inputs \
                --group_by_length \
                --sample $sample \
                --seed $seed
                # --resume_from_checkpoint $instruction_model \
        done
    done
done