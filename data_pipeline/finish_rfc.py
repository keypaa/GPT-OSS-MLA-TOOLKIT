import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import json
from datasets import load_dataset
from huggingface_hub import HfApi
import gc
import os

os.environ["OMP_NUM_THREADS"] = "1"

def finish_rfc_full_only(output_file="calib_full_golden.jsonl", hf_token=None):
    # 1. Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # --- CONFIGURATION ---
    MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
    
    # CRITICAL FIX: Reduced Batch Size for Long Context
    # 256 worked for short text, but 8k tokens needs much less.
    # 32 is extremely safe for 48GB VRAM.
    BATCH_SIZE = 80
    
    # We want a solid pool of long docs
    TARGET_FULL_DOCS = 40_000 
    
    LENGTH_THRESHOLD_CHAR = 12000 
    
    print(f"--- FINISHING RFC: FULL SET ONLY ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch: {BATCH_SIZE} (Safe Mode)")
    print(f"Goal:  Collect {TARGET_FULL_DOCS} Long Documents")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n1. Loading Model...")
    embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    embedder.max_seq_length = 8192
    if torch.cuda.is_bf16_supported():
        embedder.half()

    print("\n2. Streaming Data (Skipping short docs)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    full_texts = []
    
    iterator = iter(dataset)
    while len(full_texts) < TARGET_FULL_DOCS:
        try:
            item = next(iterator)
            text = item['text']
            
            # We ONLY keep the long ones now
            if len(text) >= LENGTH_THRESHOLD_CHAR:
                full_texts.append(text)
                
            if len(full_texts) % 1000 == 0:
                print(f"   Collected: {len(full_texts)} / {TARGET_FULL_DOCS}", end="\r")
        except StopIteration:
            break
            
    print(f"\n   Collection Complete.")

    # --- PROCESS FULL ---
    print(f"\n3. Processing FULL Set...")
    
    print("   Generating Embeddings (Batch Size 32)...")
    embeddings = embedder.encode(
        full_texts, 
        show_progress_bar=True, 
        batch_size=BATCH_SIZE, # <--- The OOM Fix
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    
    print("   Clustering...")
    n_clusters = 512
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init="auto")
    kmeans.fit(embeddings)
    
    selected_indices = set()
    
    # Centroids
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    for idx in closest: selected_indices.add(idx)
    
    # Outliers
    dists = kmeans.transform(embeddings)
    min_dists = np.min(dists, axis=1)
    
    # Select top 5% hardest
    n_outliers = int(len(full_texts) * 0.05)
    outlier_indices = np.argsort(min_dists)[-n_outliers:]
    for idx in outlier_indices: selected_indices.add(idx)
    
    print(f"   Saving {len(selected_indices)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in selected_indices:
            record = {
                "text": full_texts[idx],
                "meta": {
                    "type": "full_golden",
                    "model": "gte-large-v1.5",
                    "length": len(full_texts[idx])
                }
            }
            json.dump(record, f)
            f.write("\n")

    print(f"\n4. Uploading to HuggingFace...")
    try:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=output_file,
            path_in_repo=output_file,
            repo_id="keypa/gpt-oss-calibration-data",
            repo_type="dataset",
            token=hf_token
        )
        print(f"   ✓ Successfully uploaded to HuggingFace dataset")
    except Exception as e:
        print(f"   ✗ Upload failed: {e}")
        print(f"   Note: File is still saved locally at {output_file}")
        print(f"   Tip: Make sure you provided a valid HF token with write access")

    print("\n--- Mission Complete ---")

if __name__ == "__main__":
    import sys
    
    # Get HF token from command line argument or environment variable
    hf_token = None
    if len(sys.argv) > 1:
        hf_token = sys.argv[1]
    else:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: No HuggingFace token provided.")
            print("Usage: python finish_rfc.py <HF_TOKEN>")
            print("Or set HF_TOKEN environment variable")
            print("Proceeding without upload capability...\n")
    
    finish_rfc_full_only(hf_token=hf_token)