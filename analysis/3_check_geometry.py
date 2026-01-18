import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

def check_structure(model_path="openai/gpt-oss-20b"):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Loading model: {model_path}...")
    
    max_memory_mapping = {0: "35GiB", "cpu": "128GiB"}
    if not os.path.exists("offload_folder"): os.makedirs("offload_folder")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory_mapping, offload_folder="offload_folder", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    qk_stats = {}

    def get_hook(layer_idx, label):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            with torch.no_grad():
                avg_norm = torch.norm(output.float(), p=2, dim=-1).mean().item()
                if layer_idx not in qk_stats: qk_stats[layer_idx] = {}
                qk_stats[layer_idx][label] = avg_norm
        return hook

    target_layers = [0, 10, 20]
    print(f"Hooking layers {target_layers}...")
    for name, module in model.named_modules():
        parts = name.split('.')
        for p in parts:
            if p.isdigit() and int(p) in target_layers:
                if "q_proj" in parts[-1]: module.register_forward_hook(get_hook(int(p), "Q"))
                elif "k_proj" in parts[-1]: module.register_forward_hook(get_hook(int(p), "K"))

    print("Running inference...")
    text = "Testing geometry."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**inputs)

    print("\n--- Geometric Analysis ---")
    for layer in sorted(qk_stats.keys()):
        q = qk_stats[layer].get("Q", 0); k = qk_stats[layer].get("K", 0)
        ratio = q/k if k > 0 else 0
        print(f"Layer {layer}: Q={q:.1f} | K={k:.1f} | Ratio={ratio:.2f}")
        if ratio > 5.0 or ratio < 0.2: print("  >>> WARNING: Scale Mismatch")

if __name__ == "__main__":
    check_structure()