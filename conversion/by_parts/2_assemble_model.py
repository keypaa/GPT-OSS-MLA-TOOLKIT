import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import os
import gc
import shutil

def assemble_model(original_path, parts_dir, output_path):
    print("--- ASSEMBLING FINAL MODEL ---")
    
    # 1. Load the Base Model Skeleton (Low CPU usage)
    # We load it in "sharded" mode to keep RAM manageable
    print(f"Loading Base Model from: {original_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            original_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu", 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return
    
    # 2. Inject Converted Layers
    files = sorted([f for f in os.listdir(parts_dir) if f.endswith(".safetensors")])
    if len(files) == 0:
        print("ERROR: No layer files found in conversion_parts/")
        return
        
    print(f"Found {len(files)} converted layers. Injecting...")
    
    for f in files:
        path = os.path.join(parts_dir, f)
        
        # Extract layer index from filename "layer_05.safetensors"
        try:
            layer_idx = int(f.split('_')[1].split('.')[0])
        except:
            continue
            
        print(f"   Overwriting Layer {layer_idx}...", end="\r")
        
        # Load the small layer file
        state_dict = load_file(path)
        
        # Assign to main model memory
        # This replaces the original GQA weights with your new Low-Rank weights
        model.model.layers[layer_idx].self_attn.k_proj.weight.data = state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
        model.model.layers[layer_idx].self_attn.v_proj.weight.data = state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
        
        del state_dict
        gc.collect()

    print(f"\n\n3. Saving Full Model to {output_path}...")
    # This saves the model in chunks (safetensors)
    model.save_pretrained(output_path, max_shard_size="4GB")
    
    print("Saving Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("DONE. Model ready for PPL check.")

if __name__ == "__main__":
    model_id = "openai/gpt-oss-20b"
    
    print("Locating base model...")
    local_path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
        ignore_patterns=["*.msgpack", "*.h5"] 
    )
    
    # Run assembly
    assemble_model(local_path, "conversion_parts", "gpt-oss-20b-mla-init")