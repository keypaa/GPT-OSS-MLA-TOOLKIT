# 🗺️ Project Roadmap: GPT-OSS MLA Conversion

**Goal:** Convert the GPT-OSS-20B (GQA) model into a Multi-Head Latent Attention (MLA) architecture to drastically reduce KV cache usage while preserving Perplexity (PPL < 3.0).

---

## 🟢 Phase 1: Diagnostics & Data Engineering (COMPLETED)
*Objective: Verify the baseline model's health and generate SOTA calibration data.*

- [x] **Attention Sink Analysis:** Debunked the "massive sink" hypothesis. Proved sinks are normal (Ratio ~1.0).
- [x] **MoE Routing Health:** Confirmed experts are not collapsing (Load balanced ~12-20%).
- [x] **Geometric Analysis:** Verified Q/K scaling norms are within healthy ranges.
- [x] **Calibration Data Generation:**
    - [x] Used `GTE-Large-v1.5` (8192 context) to capture long-range dependencies.
    - [x] Implemented "Hard Negative Mining" (Centroids + Top 5% Outliers).
    - [x] Generated **SWA Split** (Short/Dense) for sliding window layers.
    - [x] Generated **Full Split** (Long/Complex) for full attention layers.
- [x] **Artifact Hosting:** Published `gpt-oss-calibration-data` to Hugging Face.

---

## 🟡 Phase 2: Architecture Conversion (CURRENT)
*Objective: Perform the mathematical compression of weights.*

- [ ] **Implement `convert_to_mla.py`:**
    - [ ] Load GPT-OSS-20B (Teacher).
    - [ ] Implement "TransMLA v2" logic:
        - [ ] **RoPE Identity:** Keep the Rotary Positional Embedding dimensions (first 64) exact/uncompressed.
        - [ ] **SVD Compression:** Perform Singular Value Decomposition on the remaining Key/Value dimensions.
    - [ ] Initialize the "Student" (MLA) model structure.
- [ ] **Verify Compression Ratio:**
    - [ ] Measure KV Cache size reduction (Target: >50% reduction vs GQA).
    - [ ] Baseline PPL check (Expect ~5.0+ PPL before healing).

---

## 🟠 Phase 3: The Healing (Distillation)
*Objective: Recover the "lost" intelligence using the Golden Data.*

- [ ] **Setup Distillation Pipeline (`train_distill.py`):**
    - [ ] Teacher: 4-bit NF4 Frozen.
    - [ ] Student: BF16 Trainable (Projectors only).
    - [ ] Loss: KL-Divergence (Soft Targets).
- [ ] **Run Training (L40S / A100):**
    - [ ] Step 1: General Healing using `calib_mixed_golden.jsonl`.
    - [ ] Step 2 (Optional): Targeted SWA/Full fine-tuning using the splits.
- [ ] **Evaluation:**
    - [ ] Monitor PPL convergence.
    - [ ] Target PPL: < 3.0 (Baseline is ~2.9).

---

## 🔵 Phase 4: Inference & Release
*Objective: Make it usable for the community.*

- [ ] **Inference Integration:**
    - [ ] Map weights to **SGLang** / **FlashInfer** MLA kernels.
    - [ ] Benchmark tokens/sec and VRAM usage vs original model.
- [ ] **Release:**
    - [ ] Upload `gpt-oss-20b-mla` weights to Hugging Face.
    - [ ] Publish technical report/blog post on `r/LocalLLaMA`.

---

## 🧊 Future / Experimental
*Ideas for after the 20B conversion is stable.*

- [ ] **Scale to 120B:** Apply the same pipeline to the full-sized model.
- [ ] **Diffusion Decoder:** Experiment with replacing the AR head with a diffusion process (Experimental).