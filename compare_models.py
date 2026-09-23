#!/usr/bin/env python3
"""
compare_models.py
Direct ablation comparison between Model B (Baseline GRU Memory)
and Model C (Proposed CCRM Cognitive Memory Module).

Evaluates both models on the exact same test split (test_embeddings.pt).
Outputs side-by-side tables of metrics, parameter counts, and latency.
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tabulate import tabulate

from model import ModelB_GRU, ModelC_CCRM
from train import ParagraphEmbeddingDataset, collate_variable_sequences, get_device


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_type = checkpoint.get('model_type', 'model_c' if 'cognitive_dropout' in config else 'model_b')

    if model_type == "model_c":
        model = ModelC_CCRM(
            embedding_dim=config.get('embedding_dim', 768),
            hidden_size=config.get('hidden_size', 256),
            mlp_hidden=config.get('mlp_hidden', 128),
            cognitive_dropout=config.get('cognitive_dropout', 0.2),
            classifier_dropout=config.get('classifier_dropout', 0.3)
        ).to(device)
        name = "Model C (CCRM + GRU + MLP)"
    else:
        model = ModelB_GRU(
            embedding_dim=config.get('embedding_dim', 768),
            hidden_size=config.get('hidden_size', 256),
            num_layers=1,
            mlp_hidden=config.get('mlp_hidden', 128),
            dropout=config.get('dropout', 0.3)
        ).to(device)
        name = "Model B (GRU + MLP)"

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, name, model_type, config


def run_model_inference(model, data_loader, device):
    all_preds, all_targets, all_probs = [], [], []

    # Warmup
    with torch.no_grad():
        for batch in data_loader:
            embs = batch['embeddings'].to(device)
            lens = batch['lengths'].to(device)
            _ = model(embs, lens)
            break

    if device.type == 'mps':
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in data_loader:
            embs = batch['embeddings'].to(device)
            lens = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embs, lens)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    if device.type == 'mps':
        torch.mps.synchronize()
    total_time = (time.perf_counter() - t0) * 1000  # ms
    avg_latency = total_time / len(data_loader.dataset)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'avg_latency_ms': avg_latency
    }


def resolve_path(p: str, fallback_subdir: str = "Model B") -> str:
    if os.path.exists(p):
        return p
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(parent, fallback_subdir, os.path.basename(p))
    if os.path.exists(fallback):
        return fallback
    return p


def compare_models(
    model_b_path: str = "model_b_gru.pth",
    model_c_path: str = "model_c_ccrm.pth",
    test_pt: str = "data/test_embeddings.pt",
    output_json: str = "model_b_vs_c_comparison.json"
):
    model_b_path = resolve_path(model_b_path, "Model B")
    device = get_device()
    print(f"Comparing models on device: {device}")
    print(f"Test dataset path: {test_pt}")

    test_dataset = ParagraphEmbeddingDataset(test_pt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_variable_sequences
    )

    models_info = [
        ("Model B (Baseline)", model_b_path),
        ("Model C (Proposed CCRM)", model_c_path),
    ]

    results = []

    for label, ckpt_path in models_info:
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        model, name, m_type, cfg = load_model(ckpt_path, device)
        total_params = sum(p.numel() for p in model.parameters())
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        metrics = run_model_inference(model, test_loader, device)

        results.append({
            'Model': label,
            'Architecture Description': name,
            'Trainable Params': total_params,
            'Checkpoint Size (MB)': round(ckpt_size_mb, 2),
            'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
            'Precision (%)': round(metrics['precision'] * 100, 2),
            'Recall (%)': round(metrics['recall'] * 100, 2),
            'F1-Score (%)': round(metrics['f1'] * 100, 2),
            'ROC-AUC': round(metrics['roc_auc'], 4),
            'Inference Latency (ms/doc)': round(metrics['avg_latency_ms'], 2),
            'TN (Human Correct)': metrics['confusion_matrix'][0][0],
            'FP (False Alarm)': metrics['confusion_matrix'][0][1],
            'FN (Missed AI)': metrics['confusion_matrix'][1][0],
            'TP (AI Caught)': metrics['confusion_matrix'][1][1]
        })

    df = pd.DataFrame(results)

    print("\n" + "="*85)
    print("         DIRECT ABLATION COMPARISON: MODEL B vs. MODEL C")
    print("="*85)
    print(tabulate(df[[
        'Model', 'Trainable Params', 'Checkpoint Size (MB)',
        'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC', 'Inference Latency (ms/doc)'
    ]], headers='keys', tablefmt='github', showindex=False))

    print("\n" + "="*85)
    print("                     CONFUSION MATRIX COMPARISON")
    print("="*85)
    print(tabulate(df[[
        'Model', 'TN (Human Correct)', 'FP (False Alarm)', 'FN (Missed AI)', 'TP (AI Caught)'
    ]], headers='keys', tablefmt='github', showindex=False))

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison report to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Model B and Model C")
    parser.add_argument("--model_b", type=str, default="model_b_gru.pth")
    parser.add_argument("--model_c", type=str, default="model_c_ccrm.pth")
    parser.add_argument("--test_pt", type=str, default="data/test_embeddings.pt")
    parser.add_argument("--output_json", type=str, default="model_b_vs_c_comparison.json")
    args = parser.parse_args()

    compare_models(args.model_b, args.model_c, args.test_pt, args.output_json)
