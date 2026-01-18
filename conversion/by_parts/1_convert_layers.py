import torch
import os
import gc
import argparse
import json
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
HEAD_DIM = 128
ROPE_DIM = 64
LATENT_DIM = 512

def compress_content_svd(w_k_content, w_v, latent_dim):
    # Combine [K_content; V]
    matrix_to_compress = torch.cat([w_k_content, w_v], dim=0).float()
    
    # Perform SVD
    try:
        U, S, Vh = torch.linalg.svd(matrix_to_compress, full_matrices=False)
    except RuntimeError:
        print("   (Switching to CPU SVD for stability)")
        U, S, Vh = torch.linalg.svd(matrix_to_compress.cpu(), full_matrices=False)
    
    # Truncate
    U_r = U[:, :latent_dim]
    S_r = S[:latent_dim]
    Vh_r = Vh[:latent_dim, :]
    
    # Reconstruct: U_r @ diag(S) @ Vh_r
    w_approx = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))
    return w_approx.to(torch.bfloat16)

def get_weight_from_shard(shard_path, weight_name):
    # Load the specific shard file (RAM heavy for a moment, but safe for 1 file)
    with torch.device("cpu"):
        state_dict = load_file(shard_path)
    if weight_name in state_dict:
        return state_dict[weight_name]
    return None

def convert_single_layer_surgical(model_path, layer_idx, output_dir):
    print(f"\n--- Processing Layer {layer_idx} ---")
    
    # 1. Read the Index Map to find where the weights live
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("Error: Could not find model.safetensors.index.json")
        return

    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    
    # 2. Identify the target parameter names
    # Note: Adjust naming if your model uses specific prefixes (e.g. "model.layers")
    k_name = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_name = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
    
    # 3. Find which file (shard) they are in
    if k_name not in weight_map:
        print(f"Error: Could not find {k_name} in weight map.")
        return
        
    shard_filename = weight_map[k_name]
    shard_path = os.path.join(model_path, shard_filename)
    
    print(f"   Target weights found in: {shard_filename}")
    
    # 4. Load the Shard directly
    try:
        shard_weights = load_file(shard_path)
    except Exception as e:
        print(f"   Error loading shard: {e}")
        return

    w_k = shard_weights[k_name]
    w_v = shard_weights[v_name]
    
    print(f"   Loaded K shape: {w_k.shape}")
    
    # 5. Get Config Info (Hardcoded for GPT-OSS-20B based on prev analysis)
    # We infer dims from the weights themselves to be safe
    # w_k shape: [Out_Features, In_Features] -> [Heads * HeadDim, Hidden]
    out_features, hidden_size = w_k.shape
    # Assuming HEAD_DIM=128 from config
    n_heads = out_features // HEAD_DIM
    
    # 6. TransMLA Math
    k_reshaped = w_k.view(n_heads, HEAD_DIM, hidden_size)
    k_rope = k_reshaped[:, :ROPE_DIM, :]     # Keep Exact
    k_content = k_reshaped[:, ROPE_DIM:, :]  # Compress
    k_content_flat = k_content.reshape(-1, hidden_size)
    
    # Compress
    combined_approx = compress_content_svd(k_content_flat, w_v, LATENT_DIM)
    
    # Reconstruct
    split_point = k_content_flat.shape[0]
    new_k_content_flat = combined_approx[:split_point, :]
    new_v = combined_approx[split_point:, :]
    
    new_k_content = new_k_content_flat.view(n_heads, HEAD_DIM - ROPE_DIM, hidden_size)
    new_k_reshaped = torch.cat([k_rope, new_k_content], dim=1)
    new_k = new_k_reshaped.reshape(-1, hidden_size)
    
    # 7. Save
    out_state_dict = {
        k_name: new_k,
        v_name: new_v
    }
    
    save_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.safetensors")
    save_file(out_state_dict, save_path)
    print(f"   SUCCESS: Saved to {save_path}")
    
    # Cleanup
    del shard_weights, w_k, w_v, new_k, new_v
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=24)
    args = parser.parse_args()
    
    model_id = "openai/gpt-oss-20b"
    
    print(f"--- PREPARING MODEL PATH ---")
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"] # Prefer safetensors
        )
        print(f"Model found at: {local_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        exit()

    temp_dir = "conversion_parts"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    for i in range(args.start, args.end):
        convert_single_layer_surgical(local_path, i, temp_dir)