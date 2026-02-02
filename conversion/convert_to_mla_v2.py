"""
GPT-OSS to MLA Conversion Script (v2 - True Architecture)

Converts a GPT-OSS GQA model to true MLA architecture with:
- New kv_a_proj (compression) and kv_b_proj (decompression) layers
- SVD initialization for optimal low-rank approximation
- Preserved MoE structure and all other components
- Compressed KV cache storage

Usage:
    python conversion/convert_to_mla_v2.py --source openai/gpt-oss-20b --output gpt-oss-20b-mla-v2 --latent-dim 384
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import os
import gc
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_gpt_oss_mla import (
    GptOssMlaConfig,
    GptOssMlaForCausalLM,
)


def svd_initialize_projections(k_weight, v_weight, latent_dim, device='cpu'):
    """
    Initialize MLA projection weights using SVD.
    
    Args:
        k_weight: [num_kv_heads * head_dim, hidden_size] - Original K projection
        v_weight: [num_kv_heads * head_dim, hidden_size] - Original V projection
        latent_dim: Target compression dimension
        device: Device for computation
        
    Returns:
        kv_a_weight: [latent_dim, hidden_size] - Down projection (compression)
        kv_b_weight: [num_kv_heads * head_dim * 2, latent_dim] - Up projection (decompression)
    """
    print(f"      Running SVD (target rank: {latent_dim})...", end=" ")
    
    # Concatenate K and V weights: [2 * num_kv_heads * head_dim, hidden_size]
    kv_concat = torch.cat([k_weight, v_weight], dim=0).float().to(device)
    
    # SVD: KV = U @ diag(S) @ Vh
    try:
        U, S, Vh = torch.linalg.svd(kv_concat, full_matrices=False)
    except RuntimeError:
        print("(using CPU for stability)...", end=" ")
        kv_concat_cpu = kv_concat.cpu()
        U, S, Vh = torch.linalg.svd(kv_concat_cpu, full_matrices=False)
        U, S, Vh = U.to(device), S.to(device), Vh.to(device)
    
    # Truncate to latent_dim
    U_r = U[:, :latent_dim]  # [2 * num_kv_heads * head_dim, latent_dim]
    S_r = S[:latent_dim]      # [latent_dim]
    Vh_r = Vh[:latent_dim, :] # [latent_dim, hidden_size]
    
    # Split singular values evenly between projections
    sqrt_S = torch.sqrt(torch.diag(S_r))
    
    # Down projection (compression): hidden -> latent
    # Shape: [latent_dim, hidden_size]
    kv_a_weight = (sqrt_S @ Vh_r)
    
    # Up projection (decompression): latent -> kv
    # Shape: [2 * num_kv_heads * head_dim, latent_dim]
    kv_b_weight = U_r @ sqrt_S
    
    # Compute reconstruction error
    reconstructed = kv_b_weight @ kv_a_weight
    error = torch.norm(kv_concat - reconstructed) / torch.norm(kv_concat)
    print(f"Done (error: {error.item():.4f})")
    
    # Return weights in PyTorch Linear format: [out_features, in_features]
    # kv_a_proj: [latent_dim, hidden_size] (hidden -> latent)
    # kv_b_proj: [2*num_kv_heads*head_dim, latent_dim] (latent -> kv)
    return kv_a_weight, kv_b_weight


def copy_moe_weights(source_layer, target_layer):
    """Copy MoE expert weights from source to target."""
    # Router
    if hasattr(source_layer, 'mlp') and hasattr(source_layer.mlp, 'router'):
        target_layer.mlp.router.weight.data.copy_(source_layer.mlp.router.weight.data)
        print("        ✓ Router")
        
        # Experts - handle GptOssExperts object
        if hasattr(source_layer.mlp, 'experts'):
            try:
                # Get the actual expert modules
                src_experts_module = source_layer.mlp.experts
                tgt_experts_module = target_layer.mlp.experts
                
                # Try different ways to access experts
                if hasattr(src_experts_module, 'experts'):
                    # ModuleList stored as .experts attribute
                    src_experts_list = src_experts_module.experts
                    tgt_experts_list = tgt_experts_module.experts
                elif hasattr(src_experts_module, 'children'):
                    # Get via children()
                    src_experts_list = list(src_experts_module.children())
                    tgt_experts_list = list(tgt_experts_module.children())
                else:
                    # Try direct indexing
                    src_experts_list = [src_experts_module[i] for i in range(getattr(src_experts_module, 'num_experts', 32))]
                    tgt_experts_list = [tgt_experts_module[i] for i in range(getattr(tgt_experts_module, 'num_experts', 32))]
                
                # Determine number of experts to copy
                num_experts = min(len(src_experts_list), len(tgt_experts_list))
                
                if num_experts == 0:
                    print(f"        ⚠ No experts found in source or target")
                    return
                
                # Copy each expert
                copied = 0
                for i in range(num_experts):
                    try:
                        src_expert = src_experts_list[i]
                        tgt_expert = tgt_experts_list[i]
                        
                        # Copy gate_up_proj if it exists (fused version)
                        if hasattr(src_expert, 'gate_up_proj') and hasattr(tgt_expert, 'gate_up_proj'):
                            tgt_expert.gate_up_proj.weight.data.copy_(src_expert.gate_up_proj.weight.data)
                            copied += 1
                        # Otherwise copy separate projections
                        else:
                            if hasattr(src_expert, 'gate_proj') and hasattr(tgt_expert, 'gate_proj'):
                                tgt_expert.gate_proj.weight.data.copy_(src_expert.gate_proj.weight.data)
                            if hasattr(src_expert, 'up_proj') and hasattr(tgt_expert, 'up_proj'):
                                tgt_expert.up_proj.weight.data.copy_(src_expert.up_proj.weight.data)
                            if hasattr(src_expert, 'down_proj') and hasattr(tgt_expert, 'down_proj'):
                                tgt_expert.down_proj.weight.data.copy_(src_expert.down_proj.weight.data)
                            copied += 1
                    except Exception as e:
                        print(f"        ⚠ Error copying expert {i}: {e}")
                        continue
                
                print(f"        ✓ {copied}/{num_experts} Experts copied")
            except Exception as e:
                print(f"        ⚠ Error accessing experts: {e}")
    else:
        print("        ⚠ MoE structure not found, skipping")


def convert_model_to_mla(
    source_path,
    output_path,
    latent_dim=384,
    device='cpu',
    load_in_8bit=False,
    gpu_mem_limit=None,
    cpu_mem_limit=None,
    offload_folder=None,
):
    """
    Convert GPT-OSS GQA model to MLA architecture.
    
    Args:
        source_path: Path or HF model ID of source GPT-OSS model
        output_path: Path to save converted MLA model
        latent_dim: Compression dimension (384 = 25% reduction from 512)
        device: Device for computation ('cpu' or 'cuda')
        load_in_8bit: Load source in 8-bit quantization (saves VRAM)
    """
    print("="*80)
    print("GPT-OSS → MLA CONVERSION (True Architecture)")
    print("="*80)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Device: {device}")
    
    # Memory check and warning
    if device == "cpu":
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            print(f"Available RAM: {available_ram_gb:.1f} GB")
            if available_ram_gb < 80:
                print("⚠️  WARNING: Low RAM detected. Conversion may fail or be very slow.")
                print(f"   Recommended: 80+ GB free RAM, you have {available_ram_gb:.1f} GB")
        except ImportError:
            pass
    print()
    
    # Load source model
    print("1. Loading source GPT-OSS model...")
    print("   (This may take a few minutes)")
    
    max_memory = None
    if device == "cuda":
        # Cap GPU/CPU usage to avoid OOM; use integer GPU index for accelerate
        gpu_cap = gpu_mem_limit or "20GiB"
        cpu_cap = cpu_mem_limit or "28GiB"
        max_memory = {0: gpu_cap, "cpu": cpu_cap}

    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else device,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if max_memory:
        load_kwargs["max_memory"] = max_memory
    
    # MoE models need offload_folder when using device_map="auto"
    if device == "cuda":
        if offload_folder is None:
            offload_folder = os.path.join(output_path, "offload_temp")
        os.makedirs(offload_folder, exist_ok=True)
        load_kwargs["offload_folder"] = offload_folder
        print(f"   Using offload folder: {offload_folder}")
    
    if load_in_8bit and device == "cuda":
        print("   Attempting 8-bit mode...")
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"   ⚠ 8-bit loading failed: {e}")
            print("   Loading without additional quantization...")
            load_kwargs.pop("quantization_config", None)
    
    source_model = AutoModelForCausalLM.from_pretrained(source_path, **load_kwargs)
    source_config = source_model.config
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
    print("   ✓ Source model loaded")
    print()
    
    # Create MLA config
    print("2. Creating MLA configuration...")
    mla_config = GptOssMlaConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        intermediate_size=source_config.intermediate_size,
        num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads,
        head_dim=getattr(source_config, 'head_dim', 64),
        kv_lora_rank=latent_dim,  # MLA compression dimension
        rope_theta=getattr(source_config, 'rope_theta', 10000.0),
        rope_scaling=getattr(source_config, 'rope_scaling', None),
        max_position_embeddings=source_config.max_position_embeddings,
        rms_norm_eps=getattr(source_config, 'rms_norm_eps', 1e-5),
        num_local_experts=getattr(source_config, 'num_local_experts', 32),
        num_experts_per_tok=getattr(source_config, 'num_experts_per_tok', 4),
        router_aux_loss_coef=getattr(source_config, 'router_aux_loss_coef', 0.9),
        layer_types=getattr(source_config, 'layer_types', None),
        sliding_window=getattr(source_config, 'sliding_window', 128),
        attention_bias=getattr(source_config, 'attention_bias', True),
        attention_dropout=getattr(source_config, 'attention_dropout', 0.0),
        pad_token_id=getattr(source_config, 'pad_token_id', None),
        eos_token_id=getattr(source_config, 'eos_token_id', None),
    )
    
    # Calculate cache reduction
    original_kv_dims = source_config.num_key_value_heads * mla_config.head_dim * 2
    reduction_pct = (1 - latent_dim / original_kv_dims) * 100
    print(f"   Original KV dims per token: {original_kv_dims}")
    print(f"   MLA compressed dims per token: {latent_dim}")
    print(f"   Cache reduction: {reduction_pct:.1f}%")
    print("   ✓ MLA config created")
    print()
    
    # Initialize MLA model (keep on CPU to save VRAM)
    print("3. Initializing MLA model structure...")
    print("   (Keeping on CPU to avoid OOM)")
    mla_model = GptOssMlaForCausalLM(mla_config)
    # Don't move to device yet - we'll copy weights layer-by-layer
    print("   ✓ MLA model initialized (on CPU)")
    print()
    
    # Copy embeddings (move to source device temporarily)
    print("4. Copying embeddings...")
    source_device = source_model.model.embed_tokens.weight.device
    mla_model.model.embed_tokens.to(source_device)
    mla_model.model.embed_tokens.weight.data.copy_(source_model.model.embed_tokens.weight.data)
    mla_model.model.embed_tokens.to('cpu')  # Move back to CPU
    print("   ✓ Embeddings copied")
    
    # Free source embeddings to save memory
    del source_model.model.embed_tokens
    gc.collect()
    print()
    
    # Copy LM head (move to source device temporarily)
    print("5. Copying LM head...")
    source_device = source_model.lm_head.weight.device
    mla_model.lm_head.to(source_device)
    mla_model.lm_head.weight.data.copy_(source_model.lm_head.weight.data)
    mla_model.lm_head.to('cpu')  # Move back to CPU
    print("   ✓ LM head copied")
    
    # Free source LM head
    del source_model.lm_head
    gc.collect()
    print()
    
    # Copy final norm (move to source device temporarily)
    print("6. Copying final layer norm...")
    source_device = source_model.model.norm.weight.device
    mla_model.model.norm.to(source_device)
    mla_model.model.norm.weight.data.copy_(source_model.model.norm.weight.data)
    mla_model.model.norm.to('cpu')  # Move back to CPU
    print("   ✓ Final norm copied")
    
    # Free source norm
    del source_model.model.norm
    gc.collect()
    print()
    
    # Convert layers
    print(f"7. Converting {source_config.num_hidden_layers} decoder layers...")
    print("   (Processing one layer at a time, freeing source memory as we go)")
    print()
    
    for layer_idx in tqdm(range(source_config.num_hidden_layers), desc="   Converting layers"):
        src_layer = source_model.model.layers[layer_idx]
        tgt_layer = mla_model.model.layers[layer_idx]
        
        # Move target layer to same device as source layer temporarily
        source_device = src_layer.input_layernorm.weight.device
        tgt_layer.to(source_device)
        
        print(f"   Layer {layer_idx} (on {source_device}):")
        
        # Copy layer norms
        tgt_layer.input_layernorm.weight.data.copy_(src_layer.input_layernorm.weight.data)
        tgt_layer.post_attention_layernorm.weight.data.copy_(src_layer.post_attention_layernorm.weight.data)
        print("        ✓ Layer norms")
        
        # Copy Q projection (unchanged)
        tgt_layer.self_attn.q_proj.weight.data.copy_(src_layer.self_attn.q_proj.weight.data)
        if hasattr(src_layer.self_attn.q_proj, 'bias') and src_layer.self_attn.q_proj.bias is not None:
            tgt_layer.self_attn.q_proj.bias.data.copy_(src_layer.self_attn.q_proj.bias.data)
        print("        ✓ Q projection")
        
        # Copy O projection (unchanged)
        tgt_layer.self_attn.o_proj.weight.data.copy_(src_layer.self_attn.o_proj.weight.data)
        if hasattr(src_layer.self_attn.o_proj, 'bias') and src_layer.self_attn.o_proj.bias is not None:
            tgt_layer.self_attn.o_proj.bias.data.copy_(src_layer.self_attn.o_proj.bias.data)
        print("        ✓ O projection")
        
        # Initialize MLA projections using SVD
        print("        → Initializing MLA projections:")
        k_weight = src_layer.self_attn.k_proj.weight.data
        v_weight = src_layer.self_attn.v_proj.weight.data
        
        kv_a_weight, kv_b_weight = svd_initialize_projections(
            k_weight, v_weight, latent_dim, device=source_device
        )
        
        tgt_layer.self_attn.kv_a_proj_with_mqa.weight.data.copy_(kv_a_weight)
        tgt_layer.self_attn.kv_b_proj.weight.data.copy_(kv_b_weight)
        print("        ✓ MLA projections initialized")
        
        # Copy MoE weights
        print("        → Copying MoE:")
        copy_moe_weights(src_layer, tgt_layer)
        
        # CRITICAL: Delete source layer to free memory immediately
        del src_layer
        source_model.model.layers[layer_idx] = None  # Clear reference
        
        # Move target layer back to CPU to free GPU memory
        tgt_layer.to('cpu')
        
        # Force garbage collection after each layer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print()
    
    print("   ✓ All layers converted")
    print()
    
    # CRITICAL: Delete source model completely before saving to free maximum memory
    print("   Freeing source model memory...")
    del source_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print("   ✓ Source model freed")
    print()
    
    # Save model
    print(f"8. Saving MLA model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    mla_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("   ✓ Model saved")
    print()
    
    # Cleanup offload folder if it was auto-created
    if device == "cuda" and offload_folder and os.path.exists(offload_folder):
        try:
            import shutil
            shutil.rmtree(offload_folder)
            print(f"   ✓ Cleaned up offload folder: {offload_folder}")
            print()
        except Exception as e:
            print(f"   ⚠ Could not remove offload folder: {e}")
            print()
    
    # Summary
    print("="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print(f"  1. Evaluate baseline PPL: python analysis/4_eval_ppl.py --model {output_path}")
    print(f"  2. Run distillation: python training/train_distill.py --student {output_path}")
    print()
    print("Notes:")
    print("  - The converted model uses compressed KV cache (reduced memory)")
    print("  - Initial quality will be degraded (expect higher PPL)")
    print("  - Distillation training will recover quality")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-OSS to MLA architecture")
    parser.add_argument(
        "--source",
        type=str,
        default="openai/gpt-oss-20b",
        help="Source model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpt-oss-20b-mla-v2",
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=384,
        help="MLA latent dimension (default: 384 for 25%% cache reduction)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation (cpu recommended for stability)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load source model in 8-bit (saves VRAM, use with --device cuda)"
    )
    parser.add_argument(
        "--gpu-mem",
        type=str,
        default=None,
        help="Max GPU memory budget (e.g., 20GiB); defaults to 20GiB when using cuda"
    )
    parser.add_argument(
        "--cpu-mem",
        type=str,
        default=None,
        help="Max CPU memory budget (e.g., 28GiB); defaults to 28GiB when using cuda"
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Folder for offloading weights during loading (auto-created if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate latent_dim
    if args.latent_dim <= 0 or args.latent_dim >= 512:
        raise ValueError(f"Invalid latent_dim: {args.latent_dim}. Must be between 1 and 512.")
    
    # Run conversion
    convert_model_to_mla(
        source_path=args.source,
        output_path=args.output,
        latent_dim=args.latent_dim,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        gpu_mem_limit=args.gpu_mem,
        cpu_mem_limit=args.cpu_mem,
        offload_folder=args.offload_folder,
    )


if __name__ == "__main__":
    main()
