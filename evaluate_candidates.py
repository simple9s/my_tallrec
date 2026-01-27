import sys
import fire
import torch
import json
import os
import numpy as np
from peft import PeftModel
from model_utils import load_model_and_tokenizer
from tqdm import tqdm

torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(ranked_positions, k_list=[1, 5, 10, 20]):
    metrics = {}
    for k in k_list:
        hits, ndcgs = [], []
        for pos in ranked_positions:
            hits.append(1.0 if pos < k else 0.0)
            ndcgs.append(1.0 / np.log2(pos + 2) if pos < k else 0.0)
        metrics[f"Hit@{k}"] = float(np.mean(hits))
        metrics[f"NDCG@{k}"] = float(np.mean(ndcgs))
    return metrics


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


def main(
    base_model: str,
    lora_weights: str,
    test_data_path: str,
    result_json_data: str = "results.json",
    batch_size: int = 8,
    load_8bit: bool = False,
):
    assert base_model and lora_weights

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        base_model,
        load_in_8bit=load_8bit,
        device_map="auto"
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']
    })
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    if not load_8bit:
        model.half()

    # Load test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    ranked_positions = []

    print(f"Evaluating {len(test_data)} samples...")

    for item in tqdm(test_data, desc="Evaluating"):
        prompt = generate_prompt(item["instruction"], item["input"])
        num_candidates = len(item["candidates"])
        target_index = item["target_index"]

        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # last token logits

        # Get logits for "1" ~ "K"
        scores = []
        for i in range(1, num_candidates + 1):
            token_id = tokenizer(str(i), add_special_tokens=False)["input_ids"]
            if len(token_id) != 1:
                scores.append(-1e9)
            else:
                scores.append(logits[token_id[0]].item())

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda x: scores[x],
            reverse=True
        )

        pos = ranked_indices.index(target_index)
        ranked_positions.append(pos)

    metrics = calculate_metrics(ranked_positions)

    # Save results
    model_name = os.path.basename(lora_weights.rstrip("/"))
    if os.path.exists(result_json_data):
        with open(result_json_data, "r") as f:
            result_data = json.load(f)
    else:
        result_data = {}

    result_data[model_name] = metrics

    with open(result_json_data, "w") as f:
        json.dump(result_data, f, indent=4)

    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("===============================")


if __name__ == "__main__":
    fire.Fire(main)
