import sys
import fire
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def calculate_metrics(ranked_positions, k_list=[1, 5, 10, 20]):
    """计算Hit@K和NDCG@K指标"""
    metrics = {}

    for k in k_list:
        hits = []
        ndcgs = []

        for pos in ranked_positions:
            hit = 1.0 if pos < k else 0.0
            hits.append(hit)

            if pos < k:
                ndcg = 1.0 / np.log2(pos + 2)
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)

        metrics[f'Hit@{k}'] = np.mean(hits) if hits else 0.0
        metrics[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0

    return metrics


def main(
        base_model: str = "",
        lora_weights: str = "",
        test_data_path: str = "",
        result_json_path: str = "",
):
    assert base_model, "Please specify --base_model"
    assert lora_weights, "Please specify --lora_weights"
    assert test_data_path, "Please specify --test_data_path"

    if not result_json_path:
        result_json_path = os.path.join(lora_weights, "results.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # 添加特殊tokens
    special_tokens = {
        'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 加载LoRA
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def score_candidate(history, candidate):
        """计算P(Yes) / (P(Yes) + P(No))"""
        prompt = f"""Below is a recommendation task. Determine if the user will purchase the candidate item.

### User Purchase History:
{history}
[HistoryEmb]

### Candidate Item:
{candidate}
[CandidateEmb]

### Question:
Based on the user representation [UserRep], will the user purchase this candidate item?

### Answer:
"""

        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)['input_ids'].to(device)
        prompt_len = prompt_ids.shape[1]

        # 计算P(Yes|prompt)
        yes_full = tokenizer(prompt + "Yes", return_tensors="pt", truncation=True, max_length=1024)['input_ids'].to(
            device)
        yes_answer_ids = yes_full[0, prompt_len:]  # "Yes"对应的token(s)

        with torch.no_grad():
            yes_outputs = model(input_ids=yes_full)
            yes_logits = yes_outputs.logits[0]  # [seq_len, vocab]

        yes_log_prob = 0.0
        for i, token_id in enumerate(yes_answer_ids):
            pos = prompt_len - 1 + i  # logits[pos]预测的是input_ids[pos+1]
            if pos < yes_logits.shape[0]:
                log_probs = torch.log_softmax(yes_logits[pos], dim=-1)
                yes_log_prob += log_probs[token_id].item()

        # 计算P(No|prompt)
        no_full = tokenizer(prompt + "No", return_tensors="pt", truncation=True, max_length=1024)['input_ids'].to(
            device)
        no_answer_ids = no_full[0, prompt_len:]

        with torch.no_grad():
            no_outputs = model(input_ids=no_full)
            no_logits = no_outputs.logits[0]

        no_log_prob = 0.0
        for i, token_id in enumerate(no_answer_ids):
            pos = prompt_len - 1 + i
            if pos < no_logits.shape[0]:
                log_probs = torch.log_softmax(no_logits[pos], dim=-1)
                no_log_prob += log_probs[token_id].item()

        # 归一化: P(Yes) / (P(Yes) + P(No))
        yes_prob = np.exp(yes_log_prob)
        no_prob = np.exp(no_log_prob)
        score = yes_prob / (yes_prob + no_prob + 1e-10)

        return score

    # 加载测试数据
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"Evaluating {len(test_data)} samples...")

    ranked_positions = []
    debug_results = []

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        history = item['history']
        candidates = item['candidates']
        target_index = item['target_index']

        # 对所有候选打分
        scores = [score_candidate(history, cand) for cand in candidates]

        # 排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        target_position = ranked_indices.index(target_index)
        ranked_positions.append(target_position)

        debug_results.append({
            'sample_id': idx,
            'target': candidates[target_index],
            'target_score': scores[target_index],
            'target_rank': target_position + 1,
            'top_5': [
                {'rank': i + 1, 'candidate': candidates[ranked_indices[i]], 'score': scores[ranked_indices[i]]}
                for i in range(min(5, len(candidates)))
            ]
        })

    # 计算指标
    metrics = calculate_metrics(ranked_positions)

    # 保存结果
    model_name = os.path.basename(lora_weights)
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[model_name] = metrics

    with open(result_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    debug_path = result_json_path.replace('.json', '_debug.json')
    with open(debug_path, 'w') as f:
        json.dump(debug_results, f, indent=2)

    print("\n" + "=" * 50)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    fire.Fire(main)