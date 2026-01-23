# Project Audit - GPT-OSS MLA Toolkit

**Date:** 2026-01-23  
**Status:** Pre-Experiment Review

---

## Critical Issues Found

### 1. ❌ Conversion Script is Broken
**File:** `conversion/convert_to_mla.py`

**Problems:**
- **Wrong constants:**
  - `HEAD_DIM = 128` (Should be `64` per config.json)
  - `ROPE_DIM = 64` (Correct)
  - `LATENT_DIM = 512` (Arbitrary, needs validation)

- **Fatal logic flaw:**
  ```python
  k_rope = k_reshaped[:, :ROPE_DIM, :]     # [:, :64, :]
  k_content = k_reshaped[:, ROPE_DIM:, :]  # [:, 64:, :] → EMPTY!
  ```
  If `HEAD_DIM == ROPE_DIM == 64`, then `k_content` is empty (slice 64:64).
  The entire "preserve RoPE + compress content" strategy fails.

**Impact:**
- If the script was run with these values, the conversion is invalid
- Need to verify what constants were ACTUALLY used for `gpt-oss-20b-mla-init`

---

### 2. ⚠️ Not True MLA Architecture
**Finding:** The converted model has:
- ✅ Modified K/V projection weights (lossy compressed via SVD)
- ❌ No new projection layers (down_proj, up_proj)
- ❌ No config changes for MLA-specific parameters
- ❌ Same architecture as original GQA

**What this means:**
- This is **weight compression**, not architectural MLA
- **No KV cache reduction** (still stores full n_heads * head_dim)
- Roadmap goal "Measure KV Cache size reduction (Target: >50%)" is **impossible** with current approach

---

### 3. ⚠️ Distillation Script Assumptions
**File:** `training/train_distill.py`

**Potential Issues:**
```python
params = [p for n, p in student.named_parameters() if "proj" in n or "norm" in n]
```

- Assumes MLA has `proj` layers to train
- But current model has no new projection layers
- Will end up training all Q/K/V projections + norms (which is actually fine for this approach)

**KL Divergence approach:**
- Temperature = 2.0 (reasonable)
- Teacher: 4-bit NF4
- Student: BF16
- Only trains projection/norm layers
- **This should work** for healing weight-compressed model

---

## What Actually Exists

### ✅ Good Components

1. **Data Pipeline** (`data_pipeline/`)
   - `generate_rfc_data.py`: Sophisticated embedding-based hard negative mining
   - Uses GTE-Large-v1.5 (8192 context)
   - Splits: SWA / FULL / MIXED
   - Smart length-based batching
   - **Status:** Ready to use (needs L40S/A100)

2. **Analysis Scripts** (`analysis/`)
   - `1_check_sink.py`: Attention sink diagnostics
   - `2_check_routing.py`: MoE routing health
   - `3_check_geometry.py`: Q/K scaling checks
   - `4_eval_ppl.py`: WikiText-2 perplexity evaluation
   - `5_measure_kv_cache.py`: KV cache measurement (just created)
   - **Status:** Ready to use

3. **Calibration Data**
   - `calib_swa_golden.jsonl` (exists)
   - `calib_mixed_golden.jsonl` (exists)
   - **Status:** Already generated

4. **Converted Model**
   - `gpt-oss-20b-mla-init/` (exists locally)
   - Uploaded to HF: `keypa/gpt-oss-20b-mla-init`
   - **Status:** Unknown validity (depends on conversion constants used)

---

## Critical Questions to Answer

### Q1: What constants were used for the actual conversion?
**Action:** Check if there's conversion log or re-read the script that was executed.

### Q2: Is the converted model actually usable?
**Test:** Run PPL evaluation to see degradation level.

### Q3: What is the actual goal?
**Options:**
- **A. Weight Compression Only:** Accept no KV cache reduction, just make model smaller via SVD
- **B. True MLA:** Requires architectural changes, new projection layers, custom inference

### Q4: Can distillation heal weight-compressed model?
**Hypothesis:** Yes, if the compression wasn't too aggressive.
**Test:** PPL before/after distillation.

---

## Proposed Action Plan

### Phase 0: Validation (Do Now)
1. ✅ **Audit complete** (this document)
2. 🔲 **Check conversion history:** What constants were actually used?
3. 🔲 **Fix conversion script:** Update constants to correct values
4. 🔲 **Document approach decision:** Weight compression OR true MLA
5. 🔲 **Verify data exists:** Check if calibration files are valid

### Phase 1: Baseline Measurements
1. 🔲 **Run PPL eval** on converted model (needs 16GB GPU)
2. 🔲 **Compare with original** GPT-OSS-20B baseline (if possible)
3. 🔲 **Skip KV cache check** (not meaningful for weight compression)

### Phase 2: Distillation (if PPL is acceptable)
1. 🔲 **Review distillation params:** Learning rate, batch size, steps
2. 🔲 **Run training:** Use `calib_mixed_golden.jsonl`
3. 🔲 **Evaluate healed model:** Check PPL recovery

### Phase 3: Release (if successful)
1. 🔲 **Upload healed model** to HF
2. 🔲 **Update README** with actual results
3. 🔲 **Document limitations:** No KV cache reduction

---

## Immediate Next Steps

### Step 1: Determine Actual Conversion Parameters
Check what was used to create `gpt-oss-20b-mla-init`.

### Step 2: Fix Conversion Script
Update `convert_to_mla.py` with correct approach:

**Option A: Full Head Compression (if ROPE_DIM == HEAD_DIM)**
- Compress all of K + V together
- No RoPE/content split
- Simpler approach

**Option B: Investigate Actual RoPE Application**
- Check GPT-OSS model code
- Determine if RoPE is really applied to full 64 dims
- If not, find the actual split point

### Step 3: Run Baseline PPL
Once conversion validity is confirmed.

---

## Decision Required

**For User:**
What is the actual goal of this project?

1. **Weight Compression:** Smaller model weights, same KV cache
   - Current approach achieves this
   - Just need to fix constants and validate

2. **True MLA:** Smaller model + smaller KV cache
   - Requires major architectural changes
   - Much more complex implementation
   - Needs custom inference kernels

**Recommendation:** Start with Option 1 (weight compression), validate it works, then consider Option 2 as future work.
