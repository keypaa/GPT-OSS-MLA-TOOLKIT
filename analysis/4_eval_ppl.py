import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os

def evaluate_perplexity(model_path):
    print(f"--- PPL EVALUATION: {model_path} ---")
    
    # 1. Load Model (4-bit for speed/memory)
    # If you have plenty of VRAM (L40S), you can remove load_in_4bit for BF16 precision
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        print("Loading in 4-bit NF4...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except ImportError:
        print("BitsAndBytes not found, loading BF16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Load Standard Benchmark (WikiText2)
    print("Loading WikiText2 test set...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    # 3. Sliding Window Evaluation
    ## max_lenght = 2048 in case of very low VRAM
    ## stride = 256 in case of very low VRAM
    # Use a smaller window if OOM occurs, but 2048/4096 is standard
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"Evaluating over {seq_len} tokens...")
    pbar = tqdm(range(0, seq_len, stride))

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # How many new tokens to predict
        
        # Prepare inputs
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        
        # We don't want to predict the history (masked out)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # Loss is calculated on the target_ids
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 4. Compute PPL
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\n======================================")
    print(f"FINAL PERPLEXITY: {ppl.item():.2f}")
    print(f"======================================")
    
    return ppl.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model folder")
    args = parser.parse_args()
    evaluate_perplexity(args.model)