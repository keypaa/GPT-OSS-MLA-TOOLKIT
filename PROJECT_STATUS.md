# GPT-OSS MLA Conversion - Project Status

**Last Updated:** 2026-01-23  
**Goal:** Implement TRUE MLA architecture for GPT-OSS models with KV cache reduction  
**Current Phase:** Architecture Design & Implementation Planning

---

## Executive Summary

**What We're Building:**
A Multi-Head Latent Attention (MLA) version of GPT-OSS-20B that achieves:
- >50% KV cache reduction during inference
- Preserves model quality (PPL < 3.0) via distillation
- Full architectural modification (not just weight compression)

**Current Status:**
- ✅ Phase 1 complete (diagnostics, data generation)
- ❌ Phase 2 blocked - current conversion approach is fundamentally wrong
- 🔄 **CRITICAL DECISION MADE:** Must implement true MLA architecture (Option A)

**Immediate Blocker:**
Need to design and implement custom `GptOssMLA` model architecture from scratch.

---

## What Has Been Done

### ✅ Phase 1: Diagnostics & Data (COMPLETE)

1. **Analysis Scripts Created:**
   - `analysis/1_check_sink.py` - Attention sink diagnostics ✅
   - `analysis/2_check_routing.py` - MoE routing health ✅
   - `analysis/3_check_geometry.py` - Q/K scaling checks ✅
   - `analysis/4_eval_ppl.py` - WikiText-2 perplexity eval ✅
   - `analysis/5_measure_kv_cache.py` - KV cache measurement ✅

2. **Calibration Data Generated:**
   - `calib_swa_golden.jsonl` - Short/dense samples for sliding window layers ✅
   - `calib_mixed_golden.jsonl` - Mixed baseline dataset ✅
   - Located in project root
   - Uses GTE-Large-v1.5 with hard negative mining

3. **Data Pipeline:**
   - `data_pipeline/generate_rfc_data.py` - Smart embedding-based generation ✅
   - Implements clustering + outlier detection
   - Optimized for L40S/A100 (48GB VRAM)

4. **Diagnostics Results:**
   - GPT-OSS-20B has healthy attention (no sink collapse)
   - MoE routing balanced (~12-20% per expert)
   - Model is suitable for MLA conversion

### ❌ Phase 2: Conversion (BLOCKED - WRONG APPROACH)

**What Exists:**
- `conversion/convert_to_mla.py` - BROKEN, uses wrong approach
- `gpt-oss-20b-mla-init/` - Model created with broken script (validity unknown)
- Model uploaded to HuggingFace: `keypa/gpt-oss-20b-mla-init`

**Critical Issues Found:**

1. **Wrong Constants:**
   ```python
   # Current (WRONG):
   HEAD_DIM = 128
   ROPE_DIM = 64
   LATENT_DIM = 512
   
   # Actual GPT-OSS-20B config:
   HEAD_DIM = 64
   ROPE_DIM = 64  # Full head uses RoPE
   LATENT_DIM = ??? (needs design decision)
   ```

2. **Fatal Logic Bug:**
   ```python
   k_rope = k_reshaped[:, :ROPE_DIM, :]     # [:, :64, :]
   k_content = k_reshaped[:, ROPE_DIM:, :]  # [:, 64:, :] → EMPTY!
   ```
   If `HEAD_DIM == ROPE_DIM`, the content slice is empty (64:64).

3. **Not True MLA:**
   - Only modifies weights in-place
   - No new projection layers created
   - No architecture changes
   - **KV cache is NOT reduced** (still stores full dimensions)
   - This is just "lossy weight compression," not MLA

### ✅ Phase 3: Distillation Script (EXISTS, NEEDS REVIEW)

**What Exists:**
- `training/train_distill.py` - KL divergence distillation ✅

**Status:**
- Code looks reasonable for healing weight-compressed model
- Will need modification for true MLA architecture
- Uses AdamW on projection layers only
- Temperature = 2.0 for KL divergence

---

## What TRUE MLA Requires

### Architecture Components Needed

Based on DeepSeek-V2/V3 papers and MLA theory:

#### 1. New Projection Layers (Per Layer)
```python
# Compression path
kv_a_proj_with_mqa = nn.Linear(hidden_size, latent_dim + q_head_dim)
# Decompression path  
kv_b_proj = nn.Linear(latent_dim, n_kv_heads * head_dim)
```

**Dimensions for GPT-OSS-20B:**
- `hidden_size`: 2880
- `n_kv_heads`: 8 (GQA heads)
- `head_dim`: 64
- `latent_dim`: **TBD** (512-1024 range, compression trade-off)

#### 2. Modified Attention Flow
```python
# COMPRESSION (run once per token)
compressed_kv = kv_a_proj_with_mqa(hidden_states)  # → [batch, seq, latent_dim]

# STORE IN KV CACHE: compressed_kv (NOT full K/V)

# DECOMPRESSION (every attention computation)
kv = kv_b_proj(compressed_kv)  # → [batch, seq, n_heads * head_dim]
k, v = split_into_heads(kv)

# ROPE APPLICATION (after decompression)
k = apply_rotary_pos_emb(k, position_ids)

# ATTENTION (standard)
attn_output = scaled_dot_product_attention(q, k, v)
```

#### 3. Key Differences from Current Approach

| Aspect | Current (Wrong) | True MLA (Needed) |
|--------|----------------|-------------------|
| **Projections** | Reuse existing k_proj/v_proj | New down/up projection layers |
| **KV Cache** | Store full [n_heads, seq, head_dim] | Store compressed [seq, latent_dim] |
| **RoPE** | Apply before compression | Apply AFTER decompression |
| **Architecture** | Same as GQA | New attention mechanism |
| **Cache Reduction** | 0% | ~85-90% |

### Weight Initialization Strategy

Use SVD to initialize the new MLA projections:

```python
# Concatenate original K and V weights
kv_concat = torch.cat([k_proj.weight, v_proj.weight], dim=0)
# kv_concat shape: [2 * n_kv_heads * head_dim, hidden_size]

# Perform SVD
U, S, Vh = torch.linalg.svd(kv_concat, full_matrices=False)

# Truncate to latent_dim
U_r = U[:, :latent_dim]
S_r = S[:latent_dim]
Vh_r = Vh[:latent_dim, :]

# Initialize projections
kv_a_proj.weight = (torch.sqrt(torch.diag(S_r)) @ Vh_r).T  # Down
kv_b_proj.weight = (U_r @ torch.sqrt(torch.diag(S_r)))     # Up
```

---

## Technical Decisions Required

### Decision 1: Latent Dimension
**Options:**
- 512: Higher compression (~90% cache reduction), more quality loss
- 768: Balanced
- 1024: Lower compression (~80% cache reduction), less quality loss

**Recommendation:** Start with 512, can increase if PPL is too degraded.

**Formula:**
```
Cache reduction = 1 - (latent_dim / (n_kv_heads * head_dim))
                = 1 - (512 / (8 * 64))
                = 1 - (512 / 512)
                = 0% ← WAIT, this doesn't work!
```

**CRITICAL REALIZATION:**
With `n_kv_heads=8` and `head_dim=64`, we have `8*64=512` total KV dimensions.
- If `latent_dim=512`, there's **no compression**!
- Need `latent_dim < 512` for actual reduction
- Try `latent_dim=256` (50% reduction) or `latent_dim=384` (25% reduction)

### Decision 2: RoPE Handling
**Question:** Does GPT-OSS apply RoPE to full 64 dims or partial?

**Investigation needed:**
- Check GPT-OSS model source code (if available)
- Or inspect actual attention implementation in existing model
- This affects how we apply RoPE after decompression

**Current assumption:** Full head (64 dims) uses RoPE

### Decision 3: Model Class Implementation
**Options:**
1. **Modify existing transformers code** - Copy `modeling_gpt_oss.py` and modify
2. **Create standalone class** - Write from scratch, less coupling
3. **Use transformers custom model** - Register new architecture type

**Recommendation:** Option 1 - easier to maintain compatibility

---

## Current GPT-OSS-20B Architecture

**From config.json:**
```json
{
  "model_type": "gpt_oss",
  "num_hidden_layers": 24,
  "hidden_size": 2880,
  "num_attention_heads": 64,  // Query heads
  "num_key_value_heads": 8,   // GQA - fewer KV heads
  "head_dim": 64,
  "intermediate_size": 2880,  // FFN
  "num_local_experts": 32,    // MoE
  "num_experts_per_tok": 4,
  "layer_types": [
    "sliding_attention", "full_attention", ...  // Alternating
  ],
  "sliding_window": 128,
  "max_position_embeddings": 131072,
  "rope_theta": 150000,
  "rope_scaling": { "rope_type": "yarn", ... }
}
```

**Key Constraints:**
- Must preserve MoE structure (32 experts, top-4 routing)
- Must preserve layer type alternation (sliding/full attention)
- Must preserve YARN RoPE scaling for long context

---

## Implementation Plan

### Step 1: Design MLA Attention Module ⏳

**File to create:** `modeling_gpt_oss_mla.py`

**Components:**
```python
class GptOssMlaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Query projection (unchanged)
        self.q_proj = nn.Linear(hidden_size, n_q_heads * head_dim)
        
        # MLA compression/decompression
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, latent_dim)
        self.kv_b_proj = nn.Linear(latent_dim, n_kv_heads * head_dim * 2)
        
        # RoPE (apply after decompression)
        self.rotary_emb = GptOssRotaryEmbedding(...)
        
    def forward(self, hidden_states, position_ids, past_key_value=None, ...):
        # Compute queries
        q = self.q_proj(hidden_states)
        q = apply_rope(q, position_ids)
        
        # Compress KV
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        
        # Cache management
        if past_key_value is not None:
            compressed_kv = torch.cat([past_key_value, compressed_kv], dim=1)
        
        # Decompress
        kv = self.kv_b_proj(compressed_kv)
        k, v = kv.chunk(2, dim=-1)
        k = apply_rope(k, position_ids)
        
        # Attention (standard GQA)
        attn_output = flash_attention(q, k, v, ...)
        
        return attn_output, compressed_kv  # Return compressed for caching
```

### Step 2: Create Full Model Class ⏳

**File to create:** `modeling_gpt_oss_mla.py` (continued)

**Components:**
- `GptOssMlaDecoderLayer` - Wraps attention + FFN + MoE
- `GptOssMlaModel` - Stack of decoder layers
- `GptOssMlaForCausalLM` - Add LM head

**Must preserve:**
- MoE routing logic
- Layer type handling (sliding vs full attention)
- All normalization layers
- Gradient checkpointing support

### Step 3: Weight Conversion Script (Rewrite) ⏳

**File to rewrite:** `conversion/convert_to_mla.py`

**New logic:**
```python
def convert_gqa_to_mla(source_model, latent_dim):
    """Convert GQA model to MLA by initializing projections via SVD."""
    
    # Create new MLA model structure
    mla_config = source_model.config.copy()
    mla_config.model_type = "gpt_oss_mla"
    mla_config.kv_latent_dim = latent_dim
    
    mla_model = GptOssMlaForCausalLM(mla_config)
    
    # Copy all non-attention weights
    copy_embedding_layers(source_model, mla_model)
    copy_moe_layers(source_model, mla_model)
    copy_norm_layers(source_model, mla_model)
    
    # Initialize MLA projections via SVD
    for i, (src_layer, mla_layer) in enumerate(zip(source_model.layers, mla_model.layers)):
        # Extract original K/V weights
        k_weight = src_layer.self_attn.k_proj.weight
        v_weight = src_layer.self_attn.v_proj.weight
        
        # SVD initialization
        kv_concat = torch.cat([k_weight, v_weight], dim=0)
        U, S, Vh = torch.linalg.svd(kv_concat, full_matrices=False)
        
        # Assign to new projections
        U_r = U[:, :latent_dim]
        S_r = S[:latent_dim]
        Vh_r = Vh[:latent_dim, :]
        
        sqrt_S = torch.sqrt(torch.diag(S_r))
        mla_layer.self_attn.kv_a_proj_with_mqa.weight.data = (sqrt_S @ Vh_r).T
        mla_layer.self_attn.kv_b_proj.weight.data = U_r @ sqrt_S
        
        # Copy Q projection unchanged
        mla_layer.self_attn.q_proj.weight.data = src_layer.self_attn.q_proj.weight.data.clone()
    
    return mla_model
```

### Step 4: Update Distillation Script ⏳

**File to modify:** `training/train_distill.py`

**Changes needed:**
- Load new MLA model class
- Ensure proper parameter selection (MLA projections + norms)
- Handle compressed KV cache during training
- May need smaller batch size (BF16 student is larger)

### Step 5: Evaluation & Validation ⏳

**Tests needed:**
1. **Correctness:** MLA model can load and run inference
2. **Cache size:** Measure actual KV cache reduction
3. **Baseline PPL:** Check quality loss from SVD initialization
4. **Post-distillation PPL:** Verify recovery to <3.0

---

## Known Issues & Risks

### Issue 1: No Reference Implementation
**Problem:** DeepSeek code exists but for their architecture, not GPT-OSS.
**Mitigation:** Study DeepSeek's MLA implementation, adapt carefully.

### Issue 2: RoPE Application Uncertainty
**Problem:** Don't know exact RoPE dimensions for GPT-OSS.
**Mitigation:** Inspect model code or run experiments.

### Issue 3: Distillation May Not Fully Recover
**Problem:** SVD + distillation might not reach target PPL.
**Mitigation:** Start with conservative latent_dim (384 vs 256).

### Issue 4: Inference Engine Support
**Problem:** SGLang/vLLM don't natively support GPT-OSS-MLA.
**Mitigation:** Phase 4 work, may need custom kernels.

### Issue 5: VRAM Requirements
**Problem:** 16GB may not be enough for training/eval.
**Mitigation:** 
- Use 4-bit quantization where possible
- Reduce batch size
- Use gradient checkpointing
- May need cloud GPU (L40S/A100)

---

## Files Status

### ✅ Ready to Use
- `analysis/1_check_sink.py`
- `analysis/2_check_routing.py`
- `analysis/3_check_geometry.py`
- `analysis/4_eval_ppl.py`
- `data_pipeline/generate_rfc_data.py`
- `calib_swa_golden.jsonl`
- `calib_mixed_golden.jsonl`

### ❌ Broken / Invalid
- `conversion/convert_to_mla.py` - Wrong constants, wrong approach
- `gpt-oss-20b-mla-init/` - Created with broken script, validity unknown
- `analysis/5_measure_kv_cache.py` - Works but not useful for current model

### ⚠️ Needs Review
- `training/train_distill.py` - May need updates for MLA architecture

### 📝 Need to Create
- `modeling_gpt_oss_mla.py` - **CRITICAL** - New model architecture
- `configuration_gpt_oss_mla.py` - Config class for MLA
- `conversion/convert_to_mla_v2.py` - Proper conversion script
- `tests/test_mla_attention.py` - Unit tests

---

## Next Session Checklist

**If you're reading this in a new session, here's what to do:**

### Immediate Actions:
1. ✅ Read this document completely
2. ✅ Read [AUDIT.md](AUDIT.md) for technical findings
3. ✅ Read [ROADMAP.md](ROADMAP.md) for original goals
4. ⬜ Confirm `latent_dim` decision (recommend 384)
5. ⬜ Start implementing `modeling_gpt_oss_mla.py`

### Questions to Ask User:
1. What latent dimension should we target? (256/384/512)
2. Do we have access to GPT-OSS source code to check RoPE implementation?
3. What GPU resources are available for training? (VRAM, count)
4. Is there a deadline or priority order for implementation?

### Before Running Anything:
- ❌ Do NOT run conversion script until rewritten
- ❌ Do NOT trust existing `gpt-oss-20b-mla-init` model
- ❌ Do NOT run PPL eval on broken model (waste of time/VRAM)
- ✅ DO implement architecture first
- ✅ DO write tests as you go
- ✅ DO update this document with progress

---

## Reference Materials

### Papers to Review:
1. **DeepSeek-V2 Paper:** Multi-Head Latent Attention architecture
2. **DeepSeek-V3 Paper:** Improvements and optimizations
3. **GQA Paper:** Grouped Query Attention (GPT-OSS uses this)

### Code References:
1. **DeepSeek transformers PR:** Look for MLA implementation
2. **GPT-OSS repo:** Check if source exists for RoPE details
3. **Flash Attention 2:** For optimized attention kernels

### Key Formulas:

**KV Cache Reduction:**
```
reduction_ratio = 1 - (latent_dim / (n_kv_heads * head_dim))
```

**SVD Reconstruction Error:**
```
error = ||KV - U_r @ S_r @ V_r||_F / ||KV||_F
```

**Distillation Loss:**
```
L_KL = KL(softmax(logits_teacher / T) || softmax(logits_student / T)) * T²
```

---

## Glossary

**MLA (Multi-Head Latent Attention):** Attention mechanism that compresses KV cache via learned projections.

**GQA (Grouped Query Attention):** Multiple query heads share fewer KV heads. GPT-OSS has 64 Q heads, 8 KV heads.

**SVD (Singular Value Decomposition):** Matrix factorization used to initialize low-rank projections.

**KV Cache:** Stored key/value tensors from previous tokens for autoregressive generation.

**Latent Dimension:** Size of compressed KV representation. Lower = more compression = more quality loss.

**RoPE (Rotary Positional Embedding):** Positional encoding method that rotates query/key vectors.

**Distillation:** Training student model to match teacher's output distribution (soft targets).

---

**Document Version:** 1.0  
**Status:** Architecture design phase  
**Next Milestone:** Complete `modeling_gpt_oss_mla.py` implementation
