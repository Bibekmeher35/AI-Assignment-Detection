#!/usr/bin/env python3
"""
extract_embeddings.py
Extracts 768-dimensional paragraph embeddings from frozen DeBERTa-v3-base with masked mean pooling.
Caches embeddings for ultra-fast training, hyperparameter exploration, and reproducible ablation studies.
"""

import os
import json
import argparse
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from model import masked_mean_pooling


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_split_embeddings(
    jsonl_path: str,
    output_pt_path: str,
    tokenizer,
    deberta_model,
    device: torch.device,
    batch_size: int = 32,
    max_tokens_per_para: int = 192,
    max_samples: int = None,
    balanced: bool = True
):
    print(f"\n--- Extracting embeddings from: {jsonl_path} ---")
    if os.path.exists(output_pt_path):
        print(f"File {output_pt_path} already exists! Skipping.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        all_lines = [json.loads(l.strip()) for l in f if l.strip()]

    if balanced and max_samples:
        human_docs = [d for d in all_lines if d['label'] == 0]
        ai_docs = [d for d in all_lines if d['label'] == 1]
        
        n_half = max_samples // 2
        random.seed(42)
        random.shuffle(human_docs)
        random.shuffle(ai_docs)
        
        selected_human = human_docs[:min(len(human_docs), n_half)]
        selected_ai = ai_docs[:min(len(ai_docs), n_half)]
        docs = selected_human + selected_ai
        random.shuffle(docs)
    elif max_samples:
        docs = all_lines[:max_samples]
    else:
        docs = all_lines

    print(f"Total documents to process: {len(docs)} (Human={sum(d['label']==0 for d in docs)}, AI={sum(d['label']==1 for d in docs)})")
    
    deberta_model.eval()
    deberta_model.to(device)

    all_doc_embeddings = []
    all_lengths = []
    all_labels = []
    all_ids = []
    all_generators = []

    # Flatten paragraphs across documents
    doc_meta = []
    flat_paragraphs = []
    doc_para_slices = []

    curr_idx = 0
    for data in docs:
        paras = data['paragraphs']
        if len(paras) == 0:
            paras = [data.get('prompt', 'Assignment text')]

        num_p = len(paras)
        doc_para_slices.append((curr_idx, curr_idx + num_p))
        flat_paragraphs.extend(paras)
        curr_idx += num_p

        doc_meta.append({
            'id': data['id'],
            'label': data['label'],
            'generator': data.get('generator', 'unknown'),
            'num_paras': num_p
        })

    print(f"Total paragraphs to encode: {len(flat_paragraphs)}")

    # Batched encoding with torch.inference_mode()
    flat_embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(flat_paragraphs), batch_size), desc="Encoding paragraphs"):
            batch_text = flat_paragraphs[i:i + batch_size]
            encoded = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_tokens_per_para,
                return_tensors="pt"
            ).to(device)

            outputs = deberta_model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            para_embs = masked_mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            flat_embeddings.append(para_embs.cpu())

    all_flat_embs = torch.cat(flat_embeddings, dim=0)  # [total_paras, 768]

    # Regroup paragraphs by document
    for doc_idx, (start, end) in enumerate(doc_para_slices):
        meta = doc_meta[doc_idx]
        doc_emb = all_flat_embs[start:end]  # [num_paras_i, 768]
        
        all_doc_embeddings.append(doc_emb)
        all_lengths.append(doc_emb.shape[0])
        all_labels.append(meta['label'])
        all_ids.append(meta['id'])
        all_generators.append(meta['generator'])

    payload = {
        'embeddings': all_doc_embeddings,  # List of [num_paras, 768]
        'lengths': torch.tensor(all_lengths, dtype=torch.long),
        'labels': torch.tensor(all_labels, dtype=torch.long),
        'ids': all_ids,
        'generators': all_generators
    }

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(payload, output_pt_path)
    print(f"Saved {len(all_doc_embeddings)} document embedding sequences to {output_pt_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract DeBERTa paragraph embeddings")
    parser.add_argument("--data_dir", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_samples", type=int, default=1200)
    parser.add_argument("--max_val_samples", type=int, default=300)
    parser.add_argument("--max_test_samples", type=int, default=400)
    args = parser.parse_args()

    device = get_device()
    print(f"Using compute device: {device}")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    deberta = AutoModel.from_pretrained(args.model_name)

    # 1. Train Split
    extract_split_embeddings(
        jsonl_path=os.path.join(args.data_dir, "train.jsonl"),
        output_pt_path=os.path.join(args.data_dir, "train_embeddings.pt"),
        tokenizer=tokenizer,
        deberta_model=deberta,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        balanced=True
    )

    # 2. Val Split
    extract_split_embeddings(
        jsonl_path=os.path.join(args.data_dir, "val.jsonl"),
        output_pt_path=os.path.join(args.data_dir, "val_embeddings.pt"),
        tokenizer=tokenizer,
        deberta_model=deberta,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        balanced=True
    )

    # 3. Test Split
    extract_split_embeddings(
        jsonl_path=os.path.join(args.data_dir, "test.jsonl"),
        output_pt_path=os.path.join(args.data_dir, "test_embeddings.pt"),
        tokenizer=tokenizer,
        deberta_model=deberta,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_test_samples,
        balanced=True
    )


if __name__ == "__main__":
    main()
