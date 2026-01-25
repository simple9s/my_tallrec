#!/bin/bash

CUDA_ID=$1
output_dir=$2

# 修改这些路径
base_model=/path/to/opt-6.7b  # 改成你的OPT模型路径
test_data=/path/to/data/amazon_books_test.json  # 改成你的测试数据路径
model_type=opt  # opt或llama

# 获取所有训练好的模型路径
model_paths=$(ls -d ${output_dir}*)

for path in $model_paths
do
    echo "Evaluating model: $path"
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate_opt.py \
        --base_model $base_model \
        --model_type $model_type \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data ${output_dir}_results.json \
        --batch_size 32
done

echo "All evaluations completed!"
echo "Results saved to ${output_dir}_results.json"