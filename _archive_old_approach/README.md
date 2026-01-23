# Archive - Old Weight Compression Approach

This folder contains files from the initial (incorrect) approach that attempted MLA via in-place weight modification.

## Why Archived

**Problem:** The original approach only compressed weights but didn't change the architecture, so:
- No KV cache reduction achieved
- Just lossy weight compression
- Doesn't meet project goal of "drastically reduce KV cache usage"

**Decision:** Implement true MLA architecture instead (Option A)

## Contents

- `conversion_v1/convert_to_mla.py` - Original broken conversion script
  - Wrong constants (HEAD_DIM=128 should be 64)
  - Fatal logic bug (empty k_content slice)
  - No architectural changes

- `5_measure_kv_cache.py` - KV cache measurement script
  - Not useful for weight-only compression
  - Can be restored once true MLA is implemented

## Note on gpt-oss-20b-mla-init

The model in `gpt-oss-20b-mla-init/` folder (and uploaded to HF) was created with the broken script.
**Validity unknown** - do not use for experiments until proper MLA architecture is implemented.

## Can Be Useful For

- Reference for SVD compression logic (the math is correct)
- Code snippets for model loading/saving
- Understanding what NOT to do

---
**Archived:** 2026-01-23  
**Reason:** Implementing true MLA architecture with projection layers
