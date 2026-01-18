# GPT-OSS MLA Conversion Toolkit

**An open-source suite for analyzing, calibrating, and compressing GPT-OSS (MoE) models into Multi-Head Latent Attention (MLA).**

This repository contains the tools used to generate the [SWA/Full Split Calibration Dataset](https://huggingface.co/datasets/YOUR_HF_USERNAME/gpt-oss-calibration-data) and the analysis scripts that debunked early theories about quantization failure modes.

## 🏆 The Artifacts (Ready to Use)
We have generated high-signal calibration data using **GTE-Large-v1.5** on **L40S** compute, strictly strictly preserving 8192-token context windows to capture long-range dependencies in code and math.

- **[HuggingFace Dataset Link](https://huggingface.co/datasets/YOUR_HF_USERNAME/gpt-oss-calibration-data)**

| Split | Description | Target Layer Type |
| :--- | :--- | :--- |
| `calib_swa_golden.jsonl` | Dense local coherence (< 2k tokens) | Sliding Window Attention |
| `calib_full_golden.jsonl` | Long-range outliers (> 4k tokens) | Full Attention |
| `calib_mixed_golden.jsonl` | Baseline mix | General Calibration |

## 🧪 Analysis Tools
Scripts to diagnose why MLA conversion might degrade Perplexity (PPL).

1.  **`analysis/1_check_sinks.py`**: Measures Attention Sink norms.
    *   *Finding:* GPT-OSS-20B does **not** exhibit massive sink outliers (Ratio ~1.0). "Sink-Faithful" calibration is likely unnecessary.
2.  **`analysis/2_check_routing.py`**: Diagnostics for MoE Routing collapse.
    *   *Finding:* Expert routing remains healthy (Top expert load ~12-20%) post-quantization.
3.  **`analysis/3_check_geometry.py`**: Checks Q/K Scaling norms.

## 🛠️ Data Pipeline
Located in `data_pipeline/`.
Uses **Embedding-Based Hard Negative Mining** to find the "hardest" samples for the model (Centroids + Top 5% Outliers).
*   **Model:** Alibaba-NLP/gte-large-en-v1.5 (8192 Context)
*   **Hardware:** Optimized for L40S / A100 (48GB+ VRAM).
*   **Logic:** Sort-by-Length batching to eliminate padding overhead.

## 🚀 Distillation (The Fix)
Located in `training/`.
A dedicated `train_distill.py` script to heal PPL degradation via KL-Divergence Distillation.
*   **Teacher:** GPT-OSS-20B (4-bit NF4)
*   **Student:** Converted MLA Model (BF16)
*   **Objective:** Minimize distributional shift between Teacher and Student logits.

## 🚧 Roadmap & Status

**Current Phase: Phase 1 (SVD Conversion)**

- [x] **Data Engineering:** Generate SOTA calibration datasets (SWA/Full/Mixed).
- [x] **Diagnostics:** Confirm 20B model architecture is healthy (No sink/routing collapse).
- [ ] **SVD Conversion:** Implement `convert_to_mla.py` using "TransMLA v2" logic (Exact RoPE-K + Compressed Content-KV).
- [ ] **Healing:** Run KL-Divergence Distillation on L40S using the `calib_mixed_golden` dataset.
- [ ] **Inference:** Integrate with SGLang/FlashInfer for efficient MLA decoding.
- [ ] **Release:** Publish `gpt-oss-20b-mla` weights with < 3.0 PPL.

## Usage
```bash
pip install -r requirements.txt

# Run Data Gen
python data_pipeline/generate_rfc_data.py

# Run Analysis
python analysis/1_check_sinks.py --model "openai/gpt-oss-20b"
```
## License 
MIT