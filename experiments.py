#!/usr/bin/env python3
"""
experiments.py
Runs the systematic ablation study suite for Model B:
- Experiment B1: Standard Baseline (DeBERTa + GRU 256 + MLP)
- Experiment B2: GRU Hidden Size Ablation (128 vs 256 vs 512)
- Experiment B3: GRU Layer Depth Ablation (1-Layer vs 2-Layer)
Generates consolidated comparative tables for the research paper.
"""

import os
import json
import argparse
import pandas as pd
from train import train_model
from evaluate import evaluate_test_set


def run_all_experiments(
    train_pt: str,
    val_pt: str,
    test_pt: str,
    output_dir: str = "/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/experiments_output",
    epochs: int = 10,
    batch_size: int = 32
):
    os.makedirs(output_dir, exist_ok=True)
    results_list = []

    experiments = [
        # (Exp ID, Description, hidden_size, num_layers)
        ("B1_Baseline", "DeBERTa + GRU (256) + MLP", 256, 1),
        ("B2_Hidden_128", "DeBERTa + GRU (128) + MLP", 128, 1),
        ("B2_Hidden_512", "DeBERTa + GRU (512) + MLP", 512, 1),
        ("B3_Layers_2", "DeBERTa + GRU (256, 2-layers) + MLP", 256, 2),
    ]

    print("\n" + "#"*70)
    print("      RUNNING MODEL B SYSTEMATIC EXPERIMENTAL SUITE")
    print("#"*70)

    for exp_id, exp_name, h_size, n_layers in experiments:
        print(f"\n>>> Running Experiment: {exp_id} ({exp_name})")
        ckpt_path = os.path.join(output_dir, f"{exp_id}.pth")
        report_path = os.path.join(output_dir, f"{exp_id}_report.json")

        # 1. Train model
        train_model(
            train_pt=train_pt,
            val_pt=val_pt,
            save_path=ckpt_path,
            hidden_size=h_size,
            num_layers=n_layers,
            epochs=epochs,
            batch_size=batch_size
        )

        # 2. Evaluate on test set
        eval_res = evaluate_test_set(
            checkpoint_path=ckpt_path,
            test_pt=test_pt,
            output_report_path=report_path,
            batch_size=batch_size
        )

        metrics = eval_res['metrics']
        results_list.append({
            'Experiment ID': exp_id,
            'Model Configuration': exp_name,
            'Hidden Size': h_size,
            'GRU Layers': n_layers,
            'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
            'Precision (%)': round(metrics['precision'] * 100, 2),
            'Recall (%)': round(metrics['recall'] * 100, 2),
            'F1-Score (%)': round(metrics['f1'] * 100, 2),
            'ROC-AUC': round(metrics['roc_auc'], 4)
        })

    summary_df = pd.DataFrame(results_list)
    summary_csv = os.path.join(output_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_json = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(results_list, f, indent=2)

    print("\n" + "="*80)
    print("                    CONSOLIDATED EXPERIMENTAL RESULTS TABLE")
    print("="*80)
    print(summary_df.to_markdown(index=False))
    print("="*80)
    print(f"Summary table saved to {summary_csv} and {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model B Ablation Experiments")
    parser.add_argument("--train_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data/train_embeddings.pt")
    parser.add_argument("--val_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data/val_embeddings.pt")
    parser.add_argument("--test_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data/test_embeddings.pt")
    parser.add_argument("--output_dir", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/experiments_output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_all_experiments(
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        test_pt=args.test_pt,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
