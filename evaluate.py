#!/usr/bin/env python3
"""
evaluate.py
Evaluation suite for Model C (DeBERTa + CCRM + GRU + MLP) & Model B baseline.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Generator-specific Breakdown.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from model import ModelC_CCRM, ModelB_GRU
from train import ParagraphEmbeddingDataset, collate_variable_sequences, get_device


def evaluate_test_set(
    checkpoint_path: str,
    test_pt: str,
    output_report_path: str = None,
    batch_size: int = 32
):
    device = get_device()
    print(f"Loading checkpoint from {checkpoint_path}...")
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
        model_display_name = "Model C (CCRM Cognitive Memory + GRU + MLP)"
    else:
        model = ModelB_GRU(
            embedding_dim=config.get('embedding_dim', 768),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 1),
            mlp_hidden=config.get('mlp_hidden', 128),
            dropout=config.get('dropout', 0.3)
        ).to(device)
        model_display_name = "Model B (Baseline GRU Memory + MLP)"

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded {model_display_name} successfully. Config: {config}")

    test_dataset = ParagraphEmbeddingDataset(test_pt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_sequences
    )

    all_preds = []
    all_targets = []
    all_probs = []
    all_ids = []
    all_generators = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embeddings, lengths)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_ids.extend(batch['ids'])
            all_generators.extend(batch['generators'])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute overall metrics
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds).tolist()

    print("\n" + "="*65)
    print(f"       EVALUATION RESULTS: {model_display_name.upper()}")
    print("="*65)
    print(f"Test Set Size : {len(all_targets)} documents")
    print(f"Accuracy      : {acc*100:.2f}%")
    print(f"Precision     : {precision*100:.2f}%")
    print(f"Recall        : {recall*100:.2f}%")
    print(f"F1-Score      : {f1*100:.2f}%")
    print(f"ROC-AUC       : {auc:.4f}")
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"   [Human (TN): {cm[0][0]:>5}, AI (FP): {cm[0][1]:>5}]")
    print(f"   [Human (FN): {cm[1][0]:>5}, AI (TP): {cm[1][1]:>5}]")
    print("="*65)

    # Per-Generator Breakdown
    df_eval = pd.DataFrame({
        'id': all_ids,
        'generator': all_generators,
        'target': all_targets,
        'pred': all_preds,
        'prob': all_probs
    })

    gen_breakdown = {}
    print("\n>>> Per-Generator Performance Breakdown:")
    for gen, group in df_eval.groupby('generator'):
        g_acc = accuracy_score(group['target'], group['pred'])
        gen_breakdown[gen] = {
            'count': len(group),
            'accuracy': float(g_acc),
            'avg_ai_prob': float(group['prob'].mean()),
            'predicted_ai_count': int((group['pred'] == 1).sum()),
            'predicted_human_count': int((group['pred'] == 0).sum())
        }
        print(f"  • {gen:<25} (N={len(group):>4}): Acc={g_acc*100:>6.2f}% | Avg P(AI)={group['prob'].mean():.4f}")

    results = {
        'model_type': model_type,
        'model_display_name': model_display_name,
        'model_config': config,
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'metrics': {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(auc)
        },
        'confusion_matrix': cm,
        'generator_breakdown': gen_breakdown
    }

    if output_report_path:
        with open(output_report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comprehensive evaluation report to {output_report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model C / Model B on Test Set")
    parser.add_argument("--checkpoint", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model C/model_c_ccrm.pth")
    parser.add_argument("--test_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model C/data/test_embeddings.pt")
    parser.add_argument("--output_report", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model C/test_evaluation_report_model_c.json")
    args = parser.parse_args()

    evaluate_test_set(args.checkpoint, args.test_pt, args.output_report)
