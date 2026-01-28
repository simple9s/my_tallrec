import sys
import fire
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.set_num_threads(1)
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from model_utils import load_model_and_tokenizer
from tqdm import tqdm
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def calculate_metrics(ranked_positions, k_list=[1, 5, 10, 20]):
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
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
        test_data_path: str = "data/test_20.json",
        result_json_data: str = "results.json",
        batch_size: int = 8,
):
    assert base_model, "Please specify a --base_model"

    model_type = lora_weights.split('/')[-1]

    model, tokenizer = load_model_and_tokenizer(base_model, load_in_8bit=load_8bit, device_map="auto")

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[CandidateEmb]']})
    model.resize_token_embeddings(len(tokenizer))

    if device == "cuda":
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def score_candidate(prompt, candidate):
        """Score a single candidate"""
        full_text = prompt + candidate

        # Tokenize consistently
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")['input_ids'][0]
        full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")[
            'input_ids'][0]

        prompt_len = len(prompt_ids)

        if len(full_ids) <= prompt_len:
            return -1e10

        # Get model logits
        with torch.no_grad():
            outputs = model(input_ids=full_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Calculate log probabilities for candidate tokens
        log_probs = []
        for i in range(prompt_len - 1, min(len(full_ids) - 1, logits.shape[0] - 1)):
            next_token_id = full_ids[i + 1].item()
            log_prob = torch.log_softmax(logits[i], dim=-1)[next_token_id].item()
            log_probs.append(log_prob)

        return np.mean(log_probs) if log_probs else -1e10

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"Evaluating {len(test_data)} samples...")
    ranked_positions = []
    debug_results = []
    debug_file = result_json_data.replace('.json', '_debug.json')

    with open(debug_file, 'w') as f:
        f.write('[\n')

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        instruction = item['instruction']
        input_text = item['input']
        candidates = item['candidates']
        target_index = item['target_index']

        prompt = generate_prompt(instruction, input_text)

        # Score all candidates
        scores = []
        for cand in candidates:
            score = score_candidate(prompt, cand)
            scores.append(score)

        # Rank by score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        target_position = ranked_indices.index(target_index)
        ranked_positions.append(target_position)

        debug_results.append({
            'sample_id': idx,
            'target': candidates[target_index],
            'target_score': scores[target_index],
            'target_rank': target_position + 1,
            'top_5_predictions': [
                {
                    'rank': i + 1,
                    'candidate': candidates[ranked_indices[i]],
                    'score': scores[ranked_indices[i]]
                } for i in range(min(5, len(ranked_indices)))
            ]
        })

        with open(debug_file, 'a') as f:
            json.dump(debug_results[-1], f, indent=2)
            f.write(',\n' if idx < len(test_data) - 1 else '\n]')

        if idx < 3:
            print(f"\n{'=' * 60}")
            print(f"Sample {idx + 1}:")
            print(f"Prompt length: {len(tokenizer(prompt)['input_ids'])} tokens")
            print(f"Target: {candidates[target_index][:60]}...")
            print(f"Target score: {scores[target_index]:.4f}")
            print(f"Target rank: {target_position + 1}/{len(candidates)}")
            print(f"Top 3:")
            for i in range(min(3, len(ranked_indices))):
                ridx = ranked_indices[i]
                print(f"  {i + 1}. score={scores[ridx]:.4f}: {candidates[ridx][:50]}...")

    print(f"\nDebug results saved to: {debug_file}")

    metrics = calculate_metrics(ranked_positions)

    if os.path.exists(result_json_data):
        with open(result_json_data, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[model_type] = metrics

    with open(result_json_data, 'w') as f:
        json.dump(data, f, indent=4)

    print("\n" + "=" * 50)
    print(f"Results for {model_type}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("=" * 50)


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