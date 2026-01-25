import sys
import fire
import torch

torch.set_num_threads(1)
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig
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


def calculate_metrics(predictions, targets, candidates_list, k_list=[1, 5, 10, 20]):
    """
    Calculate Hit@k and NDCG@k
    predictions: list of predicted item indices
    targets: list of ground truth indices
    candidates_list: list of candidate lists
    """
    metrics = {}
    for k in k_list:
        hits = []
        ndcgs = []
        for pred_idx, target_idx, candidates in zip(predictions, targets, candidates_list):
            if len(candidates) < k:
                continue
            # Hit@k
            hit = 1.0 if target_idx < k else 0.0
            hits.append(hit)

            # NDCG@k
            if target_idx < k:
                ndcg = 1.0 / np.log2(target_idx + 2)
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

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model, load_in_8bit=load_8bit, device_map="auto")

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
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instructions,
            inputs=None,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            num_beams=1,
            max_new_tokens=64,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, inp) for instruction, inp in zip(instructions, inputs)]
        inputs_tensor = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs_tensor,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1].strip() for _ in output]
        return output

    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    instructions = [_['instruction'] for _ in test_data]
    inputs = [_['input'] for _ in test_data]
    candidates_list = [_['candidates'] for _ in test_data]
    target_indices = [_['target_index'] for _ in test_data]

    # Batch evaluation
    predictions = []

    def batch_data(lst, batch_size=8):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    print("Evaluating...")
    for inst_batch, inp_batch, cand_batch in tqdm(zip(
            batch_data(instructions, batch_size),
            batch_data(inputs, batch_size),
            batch_data(candidates_list, batch_size)
    ), total=(len(instructions) + batch_size - 1) // batch_size):

        outputs = evaluate(inst_batch, inp_batch)

        # Find predicted indices
        for output, candidates in zip(outputs, cand_batch):
            output_clean = output.strip().strip('"').strip("'")
            try:
                pred_idx = next((i for i, c in enumerate(candidates)
                                 if c.lower() in output_clean.lower() or
                                 output_clean.lower() in c.lower()), len(candidates))
            except:
                pred_idx = len(candidates)
            predictions.append(pred_idx)

    # Calculate metrics
    metrics = calculate_metrics(predictions, target_indices, candidates_list)

    # Save results
    if os.path.exists(result_json_data):
        with open(result_json_data, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if model_type not in data:
        data[model_type] = {}

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