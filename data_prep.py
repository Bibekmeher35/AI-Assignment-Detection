#!/usr/bin/env python3
"""
data_prep.py
Prepares train, val, and test splits from Human_and_diff_AIs.csv for Model B.
Splitting is performed strictly at the prompt/document level to prevent prompt & paragraph leakage.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def extract_paragraphs(text: str, max_paras: int = 15, min_chars: int = 25) -> list[str]:
    """
    Extracts coherent paragraphs from assignment text.
    Handles double newlines, single newlines, hard wrapping, and sentence grouping.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 1. First attempt splitting by double or multiple newlines
    blocks = [b.strip() for b in re.split(r'\n\s*\n+', text) if b.strip()]
    
    paras = []
    if len(blocks) > 1:
        for b in blocks:
            # Flatten internal line-wrapping
            clean_b = ' '.join(b.split()).strip()
            clean_b = re.sub(r'^(#+|\*+|-+)\s*', '', clean_b).strip()
            if len(clean_b) >= min_chars:
                paras.append(clean_b)
    else:
        # 2. If single block with no blank lines, join lines and group sentences into coherent paragraphs
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        full_text = ' '.join(lines)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
        
        curr = []
        for s in sentences:
            curr.append(s)
            if len(curr) >= 4 or sum(len(x) for x in curr) >= 350:
                p_text = ' '.join(curr).strip()
                if len(p_text) >= min_chars:
                    paras.append(p_text)
                curr = []
        if curr:
            p_text = ' '.join(curr).strip()
            if len(p_text) >= min_chars:
                paras.append(p_text)

    # Fallback
    if len(paras) == 0 and len(text.strip()) >= min_chars:
        paras = [text.strip()]

    # Limit to max_paras to avoid redundant tail segments
    if max_paras and len(paras) > max_paras:
        paras = paras[:max_paras]

    return paras


def prepare_dataset(
    input_csv: str,
    output_dir: str,
    max_samples_per_split: int = None,
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Raw shape: {df.shape}")

    # Identify AI columns
    ai_columns = [col for col in df.columns if col not in ['prompt', 'Human_story']]
    print(f"Detected AI generator columns: {ai_columns}")

    # Unique prompts for document-level splitting
    unique_prompts = df['prompt'].dropna().unique()
    print(f"Total unique prompts: {len(unique_prompts)}")

    # Split prompts: 70% train, 15% val, 15% test
    train_prompts, temp_prompts = train_test_split(unique_prompts, test_size=0.30, random_state=seed)
    val_prompts, test_prompts = train_test_split(temp_prompts, test_size=0.50, random_state=seed)

    prompt_split_map = {}
    for p in train_prompts:
        prompt_split_map[p] = 'train'
    for p in val_prompts:
        prompt_split_map[p] = 'val'
    for p in test_prompts:
        prompt_split_map[p] = 'test'

    print(f"Prompt split sizes -> Train: {len(train_prompts)}, Val: {len(val_prompts)}, Test: {len(test_prompts)}")

    records = {'train': [], 'val': [], 'test': []}
    doc_id = 1

    for idx, row in df.iterrows():
        prompt = row['prompt']
        if prompt not in prompt_split_map:
            continue
        split = prompt_split_map[prompt]

        # 1. Process Human Story (Label 0)
        human_text = row.get('Human_story')
        if isinstance(human_text, str) and human_text.strip():
            paras = extract_paragraphs(human_text)
            if len(paras) >= 1:
                records[split].append({
                    'id': f"doc_{doc_id:06d}",
                    'prompt': prompt,
                    'text': human_text.strip(),
                    'paragraphs': json.dumps(paras),
                    'num_paragraphs': len(paras),
                    'label': 0,
                    'generator': 'human'
                })
                doc_id += 1

        # 2. Process AI Stories (Label 1)
        for ai_col in ai_columns:
            ai_text = row.get(ai_col)
            if isinstance(ai_text, str) and ai_text.strip():
                paras = extract_paragraphs(ai_text)
                if len(paras) >= 1:
                    clean_gen_name = ai_col.split('/')[-1]
                    records[split].append({
                        'id': f"doc_{doc_id:06d}",
                        'prompt': prompt,
                        'text': ai_text.strip(),
                        'paragraphs': json.dumps(paras),
                        'num_paragraphs': len(paras),
                        'label': 1,
                        'generator': clean_gen_name
                    })
                    doc_id += 1

    # Save splits
    for split_name, split_records in records.items():
        split_df = pd.DataFrame(split_records)
        
        # Shuffle
        split_df = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        if max_samples_per_split and len(split_df) > max_samples_per_split:
            human_sub = split_df[split_df['label'] == 0]
            ai_sub = split_df[split_df['label'] == 1]
            n_per_class = max_samples_per_split // 2
            human_sampled = human_sub.sample(n=min(len(human_sub), n_per_class), random_state=seed)
            ai_sampled = ai_sub.sample(n=min(len(ai_sub), n_per_class), random_state=seed)
            split_df = pd.concat([human_sampled, ai_sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
        
        split_df.to_csv(csv_path, index=False)
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for _, r in split_df.iterrows():
                f.write(json.dumps({
                    'id': r['id'],
                    'prompt': r['prompt'],
                    'paragraphs': json.loads(r['paragraphs']),
                    'num_paragraphs': int(r['num_paragraphs']),
                    'label': int(r['label']),
                    'generator': r['generator']
                }) + '\n')

        print(f"[{split_name.upper()}] Saved {len(split_df)} samples to {csv_path} & {jsonl_path}")
        print(f"   Class balance: Human (0)={sum(split_df['label']==0)}, AI (1)={sum(split_df['label']==1)}")
        print(f"   Avg paragraphs per doc: {split_df['num_paragraphs'].mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset splits for Model B")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Project Work/Human_and_diff_AIs.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.input_csv, args.output_dir, seed=args.seed)
