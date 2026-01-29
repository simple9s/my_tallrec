import os
os.environ['WANDB_DISABLED'] = 'true'

import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def train(
    base_model: str = "",
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "./output",
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 3e-5,
    cutoff_len: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
):
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   Training Configuration                     ║
╠══════════════════════════════════════════════════════════════╣
║ Base Model: {base_model:<47}║
║ Train Data: {train_data_path:<47}║
║ Val Data:   {val_data_path:<47}║
║ Output Dir: {output_dir:<47}║
║ Seed:       {seed:<47}║
╚══════════════════════════════════════════════════════════════╝
    """)

    assert base_model, "Please specify --base_model"

    # 计算梯度累积步数
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size

    # 加载模型和tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # 设置pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = prepare_model_for_int8_training(model)

    # Tokenization函数
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """生成训练prompt并tokenize"""
        # 构建prompt - 使用特殊token作为表示学习的锚点
        full_prompt = f"""Below is a recommendation task. Determine if the user will purchase the candidate item.

### User Purchase History:
{data_point["history"]}
[HistoryEmb]

### Candidate Item:
{data_point["candidate"]}
[CandidateEmb]

### Question:
Based on the user representation [UserRep], will the user purchase this candidate item?

### Answer:
{"Yes" if data_point["label"] == 1 else "No"}"""

        tokenized_full = tokenize(full_prompt)

        # 只在Answer部分计算loss
        user_prompt = f"""Below is a recommendation task. Determine if the user will purchase the candidate item.

### User Purchase History:
{data_point["history"]}
[HistoryEmb]

### Candidate Item:
{data_point["candidate"]}
[CandidateEmb]

### Question:
Based on the user representation [UserRep], will the user purchase this candidate item?

### Answer:
"""
        tokenized_user = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user["input_ids"])

        # Mask掉prompt部分
        tokenized_full["labels"] = (
            [-100] * user_prompt_len +
            tokenized_full["labels"][user_prompt_len:]
        )

        return tokenized_full

    # 配置LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 加载数据集
    print("Loading datasets...")
    train_data = load_dataset("json", data_files=train_data_path)
    val_data = load_dataset("json", data_files=val_data_path)

    # 处理数据
    train_dataset = train_data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
    val_dataset = val_data["train"].map(generate_and_tokenize_prompt)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # 配置Trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # 开始训练
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存trainer state
    trainer.save_state()

    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train)