from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, LlamaForCausalLM


def load_model_and_tokenizer(base_model, load_in_8bit=True, device_map="auto"):
    """
    Load model and tokenizer for both OPT and LLaMA
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Determine model type
    if "opt" in base_model.lower():
        model_class = OPTForCausalLM
    elif "llama" in base_model.lower():
        model_class = LlamaForCausalLM
    else:
        model_class = AutoModelForCausalLM

    import torch
    model = model_class.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # Setup tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    return model, tokenizer