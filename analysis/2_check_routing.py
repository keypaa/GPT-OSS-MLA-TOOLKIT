import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc
import numpy as np

def check_routing(model_path="openai/gpt-oss-20b"):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Loading model: {model_path}...")
    
    max_memory_mapping = {0: "35GiB", "cpu": "128GiB"} 
    if not os.path.exists("offload_folder"): os.makedirs("offload_folder")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_memory=max_memory_mapping,
        offload_folder="offload_folder",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    routing_data = {}

    def get_router_hook(layer_idx):
        def hook(module, input, output):
            # Try to find integer indices in output
            indices = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        if item.dtype in [torch.int64, torch.int32]:
                            indices = item; break
                        # Fallback: Argmax logits if float
                        if item.dtype in [torch.bfloat16, torch.float32] and item.shape[-1] < 512:
                            indices = torch.argmax(item, dim=-1)
            elif isinstance(output, torch.Tensor):
                 indices = output if output.dtype in [torch.int64, torch.int32] else torch.argmax(output, dim=-1)

            if indices is not None:
                flat = indices.flatten().cpu().numpy()
                if layer_idx not in routing_data: routing_data[layer_idx] = []
                routing_data[layer_idx].extend(flat)
        return hook

    print("Hooking 'mlp.router' layers...")
    count = 0
    for name, module in model.named_modules():
        if "router" in name and "mlp" in name: 
            module.register_forward_hook(get_router_hook(count))
            count += 1

    print("Running inference...")
    text = "Explain the theory of relativity in simple terms." * 2
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**inputs)

    print("\n--- Routing Analysis ---")
    for layer, experts in routing_data.items():
        if len(experts) == 0: continue
        counts = {}
        for e in experts: counts[e] = counts.get(e, 0) + 1
        
        top_expert = max(counts, key=counts.get)
        share = counts[top_expert] / sum(counts.values())
        
        print(f"Layer {layer}: Used {len(counts)} experts. Top Load: {share:.1%} (Exp {top_expert})")
        if share > 0.80: print(f"  >>> CRITICAL: Collapse detected!")

if __name__ == "__main__":
    check_routing()