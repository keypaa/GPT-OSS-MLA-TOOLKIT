import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import json
from datasets import load_dataset
import gc
import os

# Thread safety fix
os.environ["OMP_NUM_THREADS"] = "1"

def generate_rfc_artifacts():
    # 1. Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # --- CONFIGURATION ---
    # GTE-Large: 0.4B Params, 8192 Context. Fast & Smart.
    MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
    BATCH_SIZE = 48 # Safe for 48GB VRAM with 8k tokens
    INPUT_POOL_SIZE = 100_000 
    
    # RFC Definition: SWA < 12k chars, FULL > 12k chars
    LENGTH_THRESHOLD_CHAR = 12000 
    
    print(f"--- RFC DATA GENERATION (Smart Sort Mode) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"1. Loading {MODEL_NAME} on {device}...")
    embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    embedder.max_seq_length = 8192
    if torch.cuda.is_bf16_supported(): embedder.half()

    print("2. Streaming Data...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    swa_texts = []
    full_texts = []
    
    iterator = iter(dataset)
    while len(swa_texts) + len(full_texts) < INPUT_POOL_SIZE:
        try:
            item = next(iterator)
            text = item['text']
            if len(text) < 1000: continue # Skip noise
            
            if len(text) < LENGTH_THRESHOLD_CHAR: swa_texts.append(text)
            else: full_texts.append(text)
            
            if (len(swa_texts) + len(full_texts)) % 10000 == 0:
                print(f"   Collected: {len(swa_texts)} SWA | {len(full_texts)} FULL", end="\r")
        except StopIteration: break

    # Process Splits
    process_split(embedder, swa_texts, "calib_swa_golden.jsonl", "swa", BATCH_SIZE * 2) # Short text = bigger batch
    process_split(embedder, full_texts, "calib_full_golden.jsonl", "full", BATCH_SIZE)

    # Generate Mixed (Optional fallback)
    # We sample 5k from each to make a mixed set
    mixed = swa_texts[:5000] + full_texts[:5000]
    process_split(embedder, mixed, "calib_mixed_golden.jsonl", "mixed", BATCH_SIZE)

def process_split(embedder, texts, filename, tag, batch_size):
    print(f"\nProcessing {tag.upper()} ({len(texts)} docs)...")
    
    # SMART SORT: Sort by length to eliminate padding overhead
    # This makes processing 3x faster
    texts.sort(key=len)
    
    embeddings = embedder.encode(
        texts, show_progress_bar=True, batch_size=batch_size, 
        convert_to_numpy=True, normalize_embeddings=True
    )
    
    print("   Clustering...")
    n_clusters = min(512, len(texts)//50)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init="auto").fit(embeddings)
    
    selected = set()
    # Centroids
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    for idx in closest: selected.add(idx)
    # Outliers (Hard Negatives)
    min_dists = np.min(kmeans.transform(embeddings), axis=1)
    outliers = np.argsort(min_dists)[-int(len(texts)*0.05):]
    for idx in outliers: selected.add(idx)
    
    print(f"   Saving {len(selected)} samples to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for idx in selected:
            json.dump({"text": texts[idx], "meta": {"type": tag, "model": "gte-large"}}, f)
            f.write("\n")

if __name__ == "__main__":
    generate_rfc_artifacts()