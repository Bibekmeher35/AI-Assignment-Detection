#!/usr/bin/env python3
"""
robustness_test.py
Robustness & Semantic Transition Disruption Experiment for Model B vs Model C.

Evaluates how Model B (Standard GRU Memory) vs Model C (Proposed CCRM) behave under:
  1. Intact Coherent Text (Standard test sequence)
  2. Disrupted Paragraph Transitions (Shuffled paragraph order)
  3. Reverse Chronological Flow (Disrupted discourse direction)

Analyzes whether explicit cognitive relationship features in CCRM (diff, interaction)
make Model C more sensitive to disrupted discourse structures.
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from compare_models import load_model
from train import get_device


def evaluate_with_perturbation(model, embeddings_list, lengths_list, labels_list, perturbation_mode="intact", seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_probs, all_targets = [], [], []

    for i in range(len(embeddings_list)):
        emb = embeddings_list[i].clone()  # [num_paras, 768]
        seq_len = lengths_list[i].item()
        label = labels_list[i].item()

        if seq_len > 1:
            if perturbation_mode == "shuffle":
                perm = torch.randperm(seq_len)
                emb[:seq_len] = emb[perm]
            elif perturbation_mode == "reverse":
                perm = torch.arange(seq_len - 1, -1, -1)
                emb[:seq_len] = emb[perm]
            # 'intact' does nothing

        batch_emb = emb.unsqueeze(0).to(device)  # [1, max_len, 768]
        batch_len = torch.tensor([seq_len], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(batch_emb, batch_len)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = torch.argmax(logits, dim=-1).item()

        all_preds.append(pred)
        all_probs.append(prob)
        all_targets.append(label)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': auc,
        'avg_ai_prob_on_human': float(all_probs[all_targets == 0].mean()),
        'avg_ai_prob_on_ai': float(all_probs[all_targets == 1].mean())
    }


def resolve_path(p: str, fallback_subdir: str = "Model B") -> str:
    if os.path.exists(p):
        return p
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(parent, fallback_subdir, os.path.basename(p))
    if os.path.exists(fallback):
        return fallback
    return p


def run_robustness_study(
    model_b_path: str = "model_b_gru.pth",
    model_c_path: str = "model_c_ccrm.pth",
    test_pt: str = "data/test_embeddings.pt",
    output_json: str = "robustness_study_results.json"
):
    model_b_path = resolve_path(model_b_path, "Model B")
    device = get_device()
    print(f"Running Robustness Study on device: {device}")

    data = torch.load(test_pt, map_location="cpu", weights_only=False)
    embeddings = data['embeddings']
    lengths = data['lengths']
    labels = data['labels']

    model_b, _, _, _ = load_model(model_b_path, device)
    model_c, _, _, _ = load_model(model_c_path, device)

    modes = [
        ("Intact Coherent Text", "intact"),
        ("Disrupted / Shuffled Paragraphs", "shuffle"),
        ("Reversed Discourse Flow", "reverse"),
    ]

    records = []
    for mode_name, mode_key in modes:
        res_b = evaluate_with_perturbation(model_b, embeddings, lengths, labels, mode_key)
        res_c = evaluate_with_perturbation(model_c, embeddings, lengths, labels, mode_key)

        records.append({
            'Scenario': mode_name,
            'Model B Acc (%)': round(res_b['accuracy'] * 100, 2),
            'Model C Acc (%)': round(res_c['accuracy'] * 100, 2),
            'Model B F1 (%)': round(res_b['f1'] * 100, 2),
            'Model C F1 (%)': round(res_c['f1'] * 100, 2),
            'Model B AUC': round(res_b['roc_auc'], 4),
            'Model C AUC': round(res_c['roc_auc'], 4),
            'B Human P(AI)': round(res_b['avg_ai_prob_on_human'] * 100, 2),
            'C Human P(AI)': round(res_c['avg_ai_prob_on_human'] * 100, 2),
            'B AI P(AI)': round(res_b['avg_ai_prob_on_ai'] * 100, 2),
            'C AI P(AI)': round(res_c['avg_ai_prob_on_ai'] * 100, 2),
        })

    df = pd.DataFrame(records)

    print("\n" + "="*85)
    print("        ROBUSTNESS & DISCOURSE DISRUPTION ANALYSIS: MODEL B vs MODEL C")
    print("="*85)
    print(tabulate(df[[
        'Scenario', 'Model B Acc (%)', 'Model C Acc (%)', 'Model B F1 (%)', 'Model C F1 (%)', 'Model B AUC', 'Model C AUC'
    ]], headers='keys', tablefmt='github', showindex=False))

    print("\n" + "="*85)
    print("               PROBABILITY SENSITIVITY UNDER PERTURBATION")
    print("="*85)
    print(tabulate(df[[
        'Scenario', 'B Human P(AI)', 'C Human P(AI)', 'B AI P(AI)', 'C AI P(AI)'
    ]], headers='keys', tablefmt='github', showindex=False))

    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved robustness study to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness Evaluation of Model B vs Model C")
    parser.add_argument("--model_b", type=str, default="model_b_gru.pth")
    parser.add_argument("--model_c", type=str, default="model_c_ccrm.pth")
    parser.add_argument("--test_pt", type=str, default="data/test_embeddings.pt")
    parser.add_argument("--output_json", type=str, default="robustness_study_results.json")
    args = parser.parse_args()

    run_robustness_study(args.model_b, args.model_c, args.test_pt, args.output_json)
