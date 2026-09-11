#!/usr/bin/env python3
"""
predict.py
End-to-End Inference CLI for Model B (DeBERTa-v3 -> Paragraph Embeddings -> GRU -> MLP).

Usage:
  python3 predict.py --text "Paragraph 1...\n\nParagraph 2...\n\nParagraph 3..."
  python3 predict.py --file path/to/assignment.txt
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from model import ModelB_GRU, masked_mean_pooling
from data_prep import extract_paragraphs
from train import get_device


class ModelBPredictor:
    def __init__(
        self,
        checkpoint_path: str = "/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/model_b_gru.pth",
        deberta_model_name: str = "microsoft/deberta-v3-base",
        device: torch.device = None
    ):
        self.device = device or get_device()
        print(f"Loading Model B on device: {self.device}")

        # 1. Load DeBERTa
        print(f"Loading feature extractor: {deberta_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
        self.deberta = AutoModel.from_pretrained(deberta_model_name).to(self.device)
        self.deberta.eval()

        # 2. Load trained Model B GRU + MLP
        print(f"Loading trained weights: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        self.model = ModelB_GRU(
            embedding_dim=config.get('embedding_dim', 768),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 1),
            mlp_hidden=config.get('mlp_hidden', 128),
            dropout=config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model B ready for inference.")

    def predict(self, text: str) -> dict:
        paragraphs = extract_paragraphs(text)
        if len(paragraphs) == 0:
            paragraphs = [text.strip()] if text.strip() else ["Empty document"]

        # Extract DeBERTa embeddings for each paragraph
        encoded = self.tokenizer(
            paragraphs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.deberta(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            para_embeddings = masked_mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])  # [num_paras, 768]
            
            # Format as batch of size 1: [1, num_paras, 768]
            batch_emb = para_embeddings.unsqueeze(0)
            lengths = torch.tensor([len(paragraphs)], dtype=torch.long, device=self.device)

            logits = self.model(batch_emb, lengths)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_class = torch.argmax(logits, dim=-1).item()

        human_prob = probs[0].item()
        ai_prob = probs[1].item()
        label_str = "AI-Generated" if pred_class == 1 else "Human-Written"

        return {
            'prediction': label_str,
            'class_id': pred_class,
            'human_confidence': round(human_prob * 100, 2),
            'ai_confidence': round(ai_prob * 100, 2),
            'num_paragraphs': len(paragraphs),
            'paragraphs': paragraphs
        }


def main():
    parser = argparse.ArgumentParser(description="Classify assignment text with Model B")
    parser.add_argument("--checkpoint", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/model_b_gru.pth")
    parser.add_argument("--text", type=str, help="Raw text of the assignment")
    parser.add_argument("--file", type=str, help="Path to text file containing assignment")
    args = parser.parse_args()

    if not args.text and not args.file:
        # Sample demonstration text
        args.text = (
            "Climate change is accelerating due to global greenhouse gas emissions.\n\n"
            "Rising temperatures significantly disrupt agricultural cycles and crop yields worldwide.\n\n"
            "Consequently, farming communities face compounding economic challenges and supply chain disruptions."
        )
        print("No input specified. Using sample input text.\n")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            args.text = f.read()

    predictor = ModelBPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.text)

    print("\n" + "="*50)
    print("           MODEL B INFERENCE RESULT")
    print("="*50)
    print(f"Classification : {result['prediction']}")
    print(f"Human Prob     : {result['human_confidence']}%")
    print(f"AI Prob        : {result['ai_confidence']}%")
    print(f"Paragraphs ({result['num_paragraphs']}):")
    for i, p in enumerate(result['paragraphs'], 1):
        print(f"   [P{i}] {p[:80]}..." if len(p) > 80 else f"   [P{i}] {p}")
    print("="*50)


if __name__ == "__main__":
    main()
