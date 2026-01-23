import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def measure_kv_cache(model_path, baseline_path=None):
    """
    Measures KV cache size during inference.
    
    Args:
        model_path: Path to model to evaluate
        baseline_path: Path to baseline model (for comparison)
    """
    print(f"\n{'='*60}")
    print(f"KV CACHE MEASUREMENT: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n1. Loading model...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get dimensions
    n_layers = config.num_hidden_layers
    n_heads = config.num_key_value_heads  # GQA heads
    head_dim = config.head_dim
    hidden_size = config.hidden_size
    
    print(f"\n   Model dimensions:")
    print(f"   - Layers: {n_layers}")
    print(f"   - KV Heads: {n_heads}")
    print(f"   - Head Dim: {head_dim}")
    print(f"   - Hidden Size: {hidden_size}")
    
    # Calculate per-token KV cache size (theoretical)
    # K cache: [batch, n_heads, seq_len, head_dim]
    # V cache: [batch, n_heads, seq_len, head_dim]
    # Per token per layer: 2 * n_heads * head_dim * bytes_per_param
    
    bytes_per_param = 2 if "bfloat16" in str(model.dtype) else 4
    kv_per_token_per_layer = 2 * n_heads * head_dim * bytes_per_param  # K and V
    kv_per_token_total = kv_per_token_per_layer * n_layers
    
    print(f"\n2. Theoretical KV Cache Size (per token):")
    print(f"   - Per layer: {kv_per_token_per_layer / 1024:.2f} KB")
    print(f"   - Total ({n_layers} layers): {kv_per_token_total / 1024 / 1024:.2f} MB")
    
    # Run a forward pass to measure actual KV cache
    print(f"\n3. Running inference to measure actual KV cache...")
    
    # Create dummy input
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    
    # Run with past_key_values return
    with torch.no_grad():
        outputs = model(
            dummy_input_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=True
        )
    
    # Measure actual KV cache if available
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        past_kv = outputs.past_key_values
        total_kv_bytes = 0
        
        print(f"\n   Actual KV cache shapes:")
        for i, (k, v) in enumerate(past_kv):
            k_bytes = k.numel() * k.element_size()
            v_bytes = v.numel() * v.element_size()
            layer_bytes = k_bytes + v_bytes
            total_kv_bytes += layer_bytes
            if i < 3:  # Print first 3 layers
                print(f"   - Layer {i}: K{tuple(k.shape)} + V{tuple(v.shape)} = {layer_bytes / 1024:.2f} KB")
        
        print(f"\n   Total KV cache (5 tokens): {total_kv_bytes / 1024 / 1024:.2f} MB")
        print(f"   Per token average: {total_kv_bytes / 5 / 1024 / 1024:.4f} MB")
    
    # Comparison with baseline
    if baseline_path and baseline_path != model_path:
        print(f"\n{'='*60}")
        print(f"BASELINE COMPARISON: {baseline_path}")
        print(f"{'='*60}")
        
        baseline_kv_per_token = kv_per_token_total
        print(f"\nBaseline theoretical KV cache per token: {baseline_kv_per_token / 1024 / 1024:.2f} MB")
        print(f"Current model theoretical KV cache per token: {kv_per_token_total / 1024 / 1024:.2f} MB")
        
        if baseline_kv_per_token > 0:
            reduction_ratio = (1 - kv_per_token_total / baseline_kv_per_token) * 100
            print(f"\nReduction: {reduction_ratio:.1f}%")
            if reduction_ratio < 0:
                print("⚠️ WARNING: Current model uses MORE cache (likely not true MLA)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model to measure")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline model for comparison")
    args = parser.parse_args()
    
    measure_kv_cache(args.model, args.baseline)
