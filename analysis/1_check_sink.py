import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

def analyze_sinks(model_path="openai/gpt-oss-20b"):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Loading model: {model_path}...")
    
    # Safe Memory Config (35GB GPU limit + CPU Offload)
    max_memory_mapping = {0: "35GiB", "cpu": "128GiB"} 
    
    if not os.path.exists("offload_folder"): os.makedirs("offload_folder")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,  
            device_map="auto",           
            max_memory=max_memory_mapping,
            offload_folder="offload_folder",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading: {e}"); return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    key_norms = {}

    def get_key_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple): hidden = output[0]
            else: hidden = output
            with torch.no_grad():
                # L2 Norm per token
                norms = torch.norm(hidden, p=2, dim=-1).float().cpu()
            if layer_idx not in key_norms: key_norms[layer_idx] = []
            key_norms[layer_idx].append(norms)
        return hook

    print("Registering hooks on sample layers...")
    target_layers = [0, 1, 10, 20, 23] # Check start, middle, end
    
    for name, module in model.named_modules():
        parts = name.split('.')
        # Robust layer finding
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if idx in target_layers and ("k_proj" in name or "key_layer" in name):
                    module.register_forward_hook(get_key_hook(idx))
                    break

    print("Running inference...")
    text = "The quick brown fox jumps over the lazy dog. " * 3 
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**inputs)

    print("\n--- Sink Analysis Results ---")
    for layer, norm_batches in key_norms.items():
        if not norm_batches: continue
        stacked = torch.cat(norm_batches, dim=0)
        # Average over batch/heads to get (SeqLen,)
        if stacked.dim() == 3: avg = stacked.mean(dim=0).mean(dim=0)
        elif stacked.dim() == 2: avg = stacked.mean(dim=0)
        else: avg = stacked

        sink_norm = avg[:4].mean().item()
        context_norm = avg[4:].mean().item()
        ratio = sink_norm / (context_norm + 1e-6)

        print(f"Layer {layer}: Sink Norm: {sink_norm:.2f} | Context: {context_norm:.2f} | Ratio: {ratio:.2f}x")
        if ratio > 10.0: print(f"  >>> WARNING: Sink detected in Layer {layer}")

if __name__ == "__main__":
    analyze_sinks()