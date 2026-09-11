#!/usr/bin/env python3
"""
train.py
Training engine for Model B: Memory-Only Baseline (DeBERTa-v3 -> Paragraph Embeddings -> GRU -> MLP -> Human/AI).

Includes:
- Milestone 1: Verification with synthetic random embeddings.
- End-to-end training loop with variable-length batch collation.
- Evaluation metrics (Loss, Accuracy, Precision, Recall, F1, ROC-AUC).
- Best checkpoint saving (model_b_gru.pth).
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from model import ModelB_GRU


class ParagraphEmbeddingDataset(Dataset):
    """
    Dataset wrapping cached document paragraph embedding sequences.
    """
    def __init__(self, pt_path: str):
        print(f"Loading cached embeddings from {pt_path}...")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.embeddings = data['embeddings']  # List of [num_paras_i, 768]
        self.lengths = data['lengths']        # LongTensor of lengths
        self.labels = data['labels']          # LongTensor of labels (0 or 1)
        self.ids = data['ids']
        self.generators = data['generators']
        print(f"Loaded {len(self.embeddings)} documents. (Human={sum(self.labels == 0).item()}, AI={sum(self.labels == 1).item()})")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],  # [num_paras, 768]
            'length': self.lengths[idx],
            'label': self.labels[idx],
            'id': self.ids[idx],
            'generator': self.generators[idx]
        }


def collate_variable_sequences(batch):
    """
    Collate function to dynamically pad variable paragraph sequences within a batch.
    """
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    max_len = max(lengths).item()
    batch_size = len(batch)
    embedding_dim = batch[0]['embedding'].shape[-1]

    padded_embeddings = torch.zeros(batch_size, max_len, embedding_dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        seq_len = item['length']
        if seq_len > 0:
            padded_embeddings[i, :seq_len] = item['embedding']

    ids = [item['id'] for item in batch]
    generators = [item['generator'] for item in batch]

    return {
        'embeddings': padded_embeddings,  # [B, max_len, 768]
        'lengths': lengths,                # [B]
        'labels': labels,                  # [B]
        'ids': ids,
        'generators': generators
    }


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_milestone_1_verification(device: torch.device):
    """
    Milestone 1: Verify GRU -> MLP architecture with synthetic random embeddings.
    """
    print("\n" + "="*50)
    print(">>> MILESTONE 1: VERIFYING GRU + MLP ON SYNTHETIC DATA")
    print("="*50)

    torch.manual_seed(42)
    batch_size = 8
    max_paras = 10
    embedding_dim = 768
    hidden_size = 256

    model = ModelB_GRU(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=1, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Create synthetic batch with variable paragraph lengths (e.g. 3 to 10 paragraphs)
    lengths = torch.tensor([3, 5, 8, 10, 4, 6, 7, 9], dtype=torch.long)
    mock_embeddings = torch.randn(batch_size, max_paras, embedding_dim, device=device)
    mock_labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=torch.long, device=device)

    model.train()
    print(f"Input Shape: {mock_embeddings.shape}")
    print(f"Lengths: {lengths.tolist()}")
    print(f"Labels: {mock_labels.tolist()}")

    initial_loss = None
    for step in range(15):
        optimizer.zero_grad()
        logits = model(mock_embeddings, lengths)
        loss = criterion(logits, mock_labels)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
            print(f"Step 0 Initial Loss: {initial_loss:.4f} | Output Logits Shape: {logits.shape}")

    final_loss = loss.item()
    print(f"Step 15 Final Loss: {final_loss:.4f}")
    assert logits.shape == (batch_size, 2), f"Expected logits shape (8, 2), got {logits.shape}"
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    print("[PASSED] Milestone 1 Verified: GRU forward pass, backpropagation, and loss reduction successful!")
    print("="*50 + "\n")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embeddings'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embeddings, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'roc_auc': auc,
        'preds': all_preds,
        'targets': all_targets,
        'probs': all_probs
    }


def train_model(
    train_pt: str,
    val_pt: str,
    save_path: str = "model_b_gru.pth",
    hidden_size: int = 256,
    num_layers: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    epochs: int = 15,
    batch_size: int = 32,
    patience: int = 4
):
    device = get_device()
    print(f"Training on device: {device}")

    train_dataset = ParagraphEmbeddingDataset(train_pt)
    val_dataset = ParagraphEmbeddingDataset(val_pt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_sequences
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_sequences
    )

    # Compute class weights for imbalanced classification if needed
    labels = train_dataset.labels.numpy()
    n_samples = len(labels)
    n_human = (labels == 0).sum()
    n_ai = (labels == 1).sum()
    weight_human = n_samples / (2.0 * max(n_human, 1))
    weight_ai = n_samples / (2.0 * max(n_ai, 1))
    class_weights = torch.tensor([weight_human, weight_ai], dtype=torch.float32).to(device)
    print(f"Class counts -> Human: {n_human}, AI: {n_ai} | Class weights: [Human={weight_human:.2f}, AI={weight_ai:.2f}]")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = ModelB_GRU(
        embedding_dim=768,
        hidden_size=hidden_size,
        num_layers=num_layers,
        mlp_hidden=128,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\n" + "="*70)
    print(f"Starting Training: Model B (GRU hidden={hidden_size}, layers={num_layers}, lr={lr})")
    print("="*70)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            lengths = batch['lengths'].to(device)
            batch_labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(embeddings, lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * len(batch_labels)
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())

        train_loss = total_train_loss / len(train_dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_p, train_r, train_f1, _ = precision_recall_fscore_support(train_targets, train_preds, average='binary', zero_division=0)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['f1'])

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, F1: {train_f1:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc']
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'embedding_dim': 768,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'mlp_hidden': 128,
                    'dropout': dropout
                },
                'val_metrics': val_metrics,
                'epoch': epoch
            }
            torch.save(checkpoint, save_path)
            print(f"   >>> [*] Best model saved with Val F1: {best_val_f1:.4f} to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Best Epoch was {best_epoch} with Val F1: {best_val_f1:.4f}")
                break

    # Save training history
    hist_path = save_path.replace('.pth', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}. Checkpoint: {save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model B: Memory-Only Baseline")
    parser.add_argument("--train_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data/train_embeddings.pt")
    parser.add_argument("--val_pt", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/data/val_embeddings.pt")
    parser.add_argument("--save_path", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/model_b_gru.pth")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_mock", action="store_true", help="Run Milestone 1 synthetic verification")
    args = parser.parse_args()

    device = get_device()
    if args.test_mock:
        run_milestone_1_verification(device)
    else:
        train_model(
            train_pt=args.train_pt,
            val_pt=args.val_pt,
            save_path=args.save_path,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
