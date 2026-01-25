import sys
import fire
import torch

torch.set_num_threads(1)
import transformers
import json
import os
import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def compute_hit_at_k(pred_scores, labels, k=1):
    """
    计算Hit@K指标
    pred_scores: 预测分数数组
    labels: 标签数组 (1=正样本, 0=负样本)
    k: top-k
    """
    group_size = 20  # 1正样本 + 19负样本
    num_groups = len(labels) // group_size

    if num_groups == 0:
        return 0.0

    hits = 0
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size

        group_scores = pred_scores[start_idx:end_idx]
        group_labels = labels[start_idx:end_idx]

        # 按分数降序排序
        sorted_indices = np.argsort(group_scores)[::-1]

        # 获取top-k的标签
        top_k_indices = sorted_indices[:k]
        top_k_labels = [group_labels[idx] for idx in top_k_indices]

        # 如果top-k中有正样本，则命中
        if 1 in top_k_labels:
            hits += 1

    hit_rate = hits / num_groups
    return hit_rate


def main(
        load_8bit: bool = False,
        base_model: str = "",
        model_type: str = "opt",
        lora_weights: str = "",
        test_data_path: str = "",
        result_json_data: str = "result.json",
        batch_size: int = 32,
):
    assert base_model, "Please specify a --base_model"
    assert lora_weights, "Please specify a --lora_weights"

    model_name = lora_weights.split('/')[-1]

    if test_data_path.find('book') > -1 or test_data_path.find('Book') > -1:
        test_sce = 'book'
    elif test_data_path.find('movie') > -1 or test_data_path.find('Movie') > -1:
        test_sce = 'movie'
    else:
        test_sce = 'amazon'

    # 加载已有结果
    if os.path.exists(result_json_data):
        with open(result_json_data, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if not data.get(test_sce):
        data[test_sce] = {}
    if not data[test_sce].get(model_name):
        data[test_sce][model_name] = {}

    # 如果已经评估过，跳过
    if 'hit@1' in data[test_sce][model_name]:
        print(f"Already evaluated: {model_name} on {test_sce}")
        exit(0)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    tokenizer.padding_side = "left"

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instructions,
            inputs=None,
            temperature=0,
            top_p=1.0,
            top_k=40,
            num_beams=1,
            max_new_tokens=128,
            **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs_tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = transformers.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs_tokenized,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        scores = generation_output.scores[0].softmax(dim=-1)

        # 根据模型类型获取Yes/No token ID
        if model_type.lower() == "opt":
            yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
        else:  # llama
            yes_token_id = 8241
            no_token_id = 3782

        logits = torch.tensor(scores[:, [no_token_id, yes_token_id]], dtype=torch.float32).softmax(dim=-1)

        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] if 'Response:\n' in _ else _.split('### Response:')[-1] for _ in output]

        return output, logits[:, 1].tolist()  # 返回Yes的概率

    # 加载数据
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    instructions = [item['instruction'] for item in test_data]
    inputs = [item['input'] for item in test_data]
    gold = [item['label'] for item in test_data]

    outputs = []
    logits = []

    print("Evaluating...")

    def batch_data(data_list, batch_size=32):
        chunk_size = (len(data_list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield data_list[batch_size * i: batch_size * (i + 1)]

    for batch_instructions, batch_inputs in tqdm(
            zip(batch_data(instructions, batch_size), batch_data(inputs, batch_size)),
            total=len(instructions) // batch_size):
        output, logit = evaluate(batch_instructions, batch_inputs)
        outputs.extend(output)
        logits.extend(logit)

    # 计算Hit@1
    hit1 = compute_hit_at_k(np.array(logits), np.array(gold), k=1)

    print(f"\nResults for {model_name} on {test_sce}:")
    print(f"Hit@1: {hit1:.4f}")

    # 保存结果
    data[test_sce][model_name]['hit@1'] = hit1
    data[test_sce][model_name]['predictions'] = logits

    with open(result_json_data, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {result_json_data}")


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)