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

    # Add special tokens
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

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def score_candidates_batch(prompts, candidates_list):
        """Batch scoring of candidates for multiple samples"""
        all_scores = []

        for prompt, candidates in zip(prompts, candidates_list):
            scores = []
            # Process candidates in batches
            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates[i:i + batch_size]
                batch_texts = [prompt + cand for cand in batch_candidates]

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                    device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Calculate average log probability for each candidate
                    for j, text in enumerate(batch_texts):
                        text_ids = tokenizer.encode(text, add_special_tokens=True)
                        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                        candidate_len = len(text_ids) - len(prompt_ids)

                        if candidate_len <= 0:
                            scores.append(-1e10)
                            continue

                        # Get log probs for candidate tokens
                        log_probs = []
                        start_idx = len(prompt_ids) - 1
                        for k in range(start_idx, min(start_idx + candidate_len, logits.shape[1] - 1)):
                            if k - start_idx + len(prompt_ids) < len(text_ids):
                                token_logits = logits[j, k]
                                next_token_id = text_ids[k - start_idx + len(prompt_ids)]
                                log_prob = torch.log_softmax(token_logits, dim=-1)[next_token_id].item()
                                log_probs.append(log_prob)

                        scores.append(np.mean(log_probs) if log_probs else -1e10)

            all_scores.append(scores)

        return all_scores

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    print(f"Evaluating {len(test_data)} samples with batch_size={batch_size}...")
    ranked_positions = []
    debug_results = []
    debug_file = result_json_data.replace('.json', '_debug.json')

    # Process in batches
    def batch_data(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    with open(debug_file, 'w') as f:
        f.write('[\n')

    sample_idx = 0
    for batch_items in tqdm(batch_data(test_data, batch_size), desc="Evaluating",
                            total=(len(test_data) + batch_size - 1) // batch_size):
        prompts = []
        candidates_list = []
        target_indices = []

        for item in batch_items:
            instruction = item['instruction']
            input_text = item['input']
            candidates = item['candidates']
            target_index = item['target_index']

            prompt = generate_prompt(instruction, input_text)
            prompts.append(prompt)
            candidates_list.append(candidates)
            target_indices.append(target_index)

        # Score all candidates in batch
        batch_scores = score_candidates_batch(prompts, candidates_list)

        # Process results
        for i, (scores, target_index, candidates) in enumerate(zip(batch_scores, target_indices, candidates_list)):
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            target_position = ranked_indices.index(target_index)
            ranked_positions.append(target_position)

            debug_results.append({
                'sample_id': sample_idx,
                'target': candidates[target_index],
                'target_score': scores[target_index],
                'target_rank': target_position + 1,
                'top_5_predictions': [
                    {
                        'rank': j + 1,
                        'candidate': candidates[ranked_indices[j]],
                        'score': scores[ranked_indices[j]]
                    } for j in range(min(5, len(ranked_indices)))
                ]
            })

            with open(debug_file, 'a') as f:
                json.dump(debug_results[-1], f, indent=2)
                f.write(',\n' if sample_idx < len(test_data) - 1 else '\n]')

            if sample_idx < 3:
                print(f"\n{'=' * 60}")
                print(f"Sample {sample_idx + 1}:")
                print(f"Target: {candidates[target_index][:60]}...")
                print(f"Target score: {scores[target_index]:.4f}")
                print(f"Target rank: {target_position + 1}/{len(candidates)}")
                print(f"Top 3:")
                for j in range(min(3, len(ranked_indices))):
                    ridx = ranked_indices[j]
                    print(f"  {j + 1}. score={scores[ridx]:.4f}: {candidates[ridx][:50]}...")

            sample_idx += 1

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