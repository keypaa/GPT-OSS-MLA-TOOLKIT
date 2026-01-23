import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import os
import gc
import argparse
import numpy as np

# --- CONFIGURATION FOR GPT-OSS-20B ---
# Based on standard architecture (check config.json to be sure)
HEAD_DIM = 128       # Standard for large models
ROPE_DIM = 64        # Usually the first half of the head is rotary
LATENT_DIM = 512     # Target compression rank (The "V_latent" size)

def compress_content_svd(w_k_content, w_v, latent_dim):
    """
    Compresses ONLY the content part of K and all of V.
    Math: SVD([K_content; V]) -> Down @ Up
    """
    # 1. Stack (Concatenate along output features dim)
    # w_k_content shape: [N_Heads * (Head_Dim - Rope_Dim), Hidden]
    # w_v shape:         [N_Heads * Head_Dim, Hidden]
    
    # We transpose to [Hidden, Out] for easier SVD logic, then transpose back
    matrix_to_compress = torch.cat([w_k_content, w_v], dim=0).float()
    
    # 2. SVD
    # U: [Out, Out], S: [Out], Vh: [Out, Hidden]
    # We want to find the best 'latent_dim' vectors in Hidden space (Vh)
    try:
        U, S, Vh = torch.linalg.svd(matrix_to_compress, full_matrices=False)
    except RuntimeError:
        # Fallback for massive matrices (CPU SVD is more stable)
        print("   (Using CPU SVD for stability)...")
        U, S, Vh = torch.linalg.svd(matrix_to_compress.cpu(), full_matrices=False)
        U, S, Vh = U.to(w_k_content.device), S.to(w_k_content.device), Vh.to(w_k_content.device)

    # 3. Truncate
    U_r = U[:, :latent_dim]
    S_r = S[:latent_dim]
    Vh_r = Vh[:latent_dim, :]
    
    # 4. Create Low-Rank Approximation
    # W_approx = U_r @ diag(S_r) @ Vh_r
    # This reconstructs the weights mathematically
    w_approx = torch.matmul(U_r, torch.matmul(torch.diag(S_r), Vh_r))
    
    return w_approx.to(w_k_content.dtype)

def convert_model_v2(source_path, output_path):
    print(f"--- TRANS-MLA V2 CONVERSION (Exact RoPE) ---")
    print(f"Source: {source_path}")
    print(f"Config: Head={HEAD_DIM}, RoPE={ROPE_DIM}, Latent={LATENT_DIM}")
    
    print("1. Loading Teacher Model (CPU)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        source_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    config = teacher.config
    n_heads = config.num_key_value_heads # GQA heads (likely 8)
    hidden_size = config.hidden_size
    
    print(f"   Model has {n_heads} KV heads.")
    
    layers = teacher.model.layers
    total_layers = len(layers)
    
    print(f"2. Processing {total_layers} Layers...")
    
    for i, layer in enumerate(layers):
        print(f"   Layer {i}/{total_layers}...", end=" ")
        
        # Original Weights [Out_Features, In_Features]
        # Out_Features = n_heads * head_dim
        w_k = layer.self_attn.k_proj.weight.data
        w_v = layer.self_attn.v_proj.weight.data
        
        # --- STEP A: DE-INTERLEAVE / RESHAPE ---
        # We need to access the specific dimensions of each head.
        # Reshape to: [N_Heads, Head_Dim, Hidden]
        k_reshaped = w_k.view(n_heads, HEAD_DIM, hidden_size)
        
        # --- STEP B: SPLIT ROPE vs CONTENT ---
        # Slicing along dim 1 (Head_Dim)
        k_rope = k_reshaped[:, :ROPE_DIM, :]     # Keep EXACT
        k_content = k_reshaped[:, ROPE_DIM:, :]  # To Compress
        
        # Reshape Content back to 2D for SVD: [N_Heads * Content_Dim, Hidden]
        k_content_flat = k_content.reshape(-1, hidden_size)
        
        # --- STEP C: COMPRESS (Content + Values) ---
        # We assume Values are fully content (no RoPE on V)
        # This returns the Reconstructed (Lossy) weights
        combined_approx = compress_content_svd(k_content_flat, w_v, LATENT_DIM)
        
        # --- STEP D: UNPACK ---
        # The first rows are K_content, the rest are V
        split_point = k_content_flat.shape[0]
        
        new_k_content_flat = combined_approx[:split_point, :]
        new_v = combined_approx[split_point:, :]
        
        # --- STEP E: RE-STITCH KEYS ---
        # Reshape new content to [N_Heads, Content_Dim, Hidden]
        new_k_content = new_k_content_flat.view(n_heads, HEAD_DIM - ROPE_DIM, hidden_size)
        
        # Concatenate: [RoPE (Exact) ; Content (Approx)]
        # Result: [N_Heads, Head_Dim, Hidden]
        new_k_reshaped = torch.cat([k_rope, new_k_content], dim=1)
        
        # Flatten back to [Out_Features, Hidden]
        new_k = new_k_reshaped.reshape(-1, hidden_size)
        
        # --- STEP F: APPLY ---
        layer.self_attn.k_proj.weight.data = new_k
        layer.self_attn.v_proj.weight.data = new_v
        
        print("Done.")
        
        # Cleanup
        del w_k, w_v, k_reshaped, k_rope, k_content, combined_approx
        if i % 5 == 0: gc.collect()

    print(f"\n3. Saving TransMLA Model to {output_path}...")
    teacher.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Conversion Complete.")

if __name__ == "__main__":
    convert_model_v2("openai/gpt-oss-20b", "gpt-oss-20b-mla-init")