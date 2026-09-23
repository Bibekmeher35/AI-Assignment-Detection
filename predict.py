#!/usr/bin/env python3
"""
predict.py
End-to-End Inference CLI for Model C (CCRM: DeBERTa-v3 -> Paragraph Embeddings -> Cognitive Memory Module -> MLP).

Also supports Model B baseline comparison on the same document.

Usage:
  python3 predict.py --text "Paragraph 1...\n\nParagraph 2...\n\nParagraph 3..."
  python3 predict.py --file path/to/assignment.txt
  python3 predict.py --file sample_essay.txt --compare
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model import ModelC_CCRM, ModelB_GRU, masked_mean_pooling
from data_prep import extract_paragraphs
from train import get_device


class AssignmentPredictor:
    """
    Unified predictor supporting Model C (CCRM) and Model B (GRU Baseline)
    with frozen DeBERTa-v3-base feature extractor.
    """
    def __init__(
        self,
        model_type: str = "c",
        checkpoint_path: str = None,
        deberta_model_name: str = "microsoft/deberta-v3-base",
        device: torch.device = None
    ):
        self.device = device or get_device()
        self.model_type = model_type.lower()
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if checkpoint_path is None:
            if self.model_type == "c":
                checkpoint_path = os.path.join(base_dir, "model_c_ccrm.pth")
            else:
                local_b = os.path.join(base_dir, "model_b_gru.pth")
                fallback_b = os.path.join(os.path.dirname(base_dir), "Model B", "model_b_gru.pth")
                checkpoint_path = local_b if os.path.exists(local_b) else fallback_b

        self.checkpoint_path = checkpoint_path

        # 1. Load DeBERTa
        print(f"Loading feature extractor: {deberta_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
        self.deberta = AutoModel.from_pretrained(deberta_model_name).to(self.device)
        self.deberta.eval()

        # 2. Load specified classification model
        print(f"Loading {self.model_type.upper()} weights from: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        if self.model_type == "c":
            self.model = ModelC_CCRM(
                embedding_dim=config.get('embedding_dim', 768),
                hidden_size=config.get('hidden_size', 256),
                mlp_hidden=config.get('mlp_hidden', 128),
                num_classes=config.get('num_classes', 2),
                cognitive_dropout=config.get('cognitive_dropout', 0.2),
                classifier_dropout=config.get('classifier_dropout', 0.3)
            ).to(self.device)
        else:
            self.model = ModelB_GRU(
                embedding_dim=config.get('embedding_dim', 768),
                hidden_size=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 1),
                mlp_hidden=config.get('mlp_hidden', 128),
                dropout=config.get('dropout', 0.3)
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model {self.model_type.upper()} successfully loaded on {self.device}.")

    def extract_paragraph_embeddings(self, paragraphs: list) -> torch.Tensor:
        """Runs DeBERTa-v3 tokenization + masked mean pooling for all paragraphs."""
        encoded = self.tokenizer(
            paragraphs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.deberta(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            para_embeddings = masked_mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
        return para_embeddings  # [num_paras, 768]

    def predict(self, text: str) -> dict:
        paragraphs = extract_paragraphs(text)
        if len(paragraphs) == 0:
            paragraphs = [text.strip()] if text.strip() else ["Empty document"]

        para_embeddings = self.extract_paragraph_embeddings(paragraphs)
        batch_emb = para_embeddings.unsqueeze(0)  # [1, num_paras, 768]
        lengths = torch.tensor([len(paragraphs)], dtype=torch.long, device=self.device)

        cognitive_norms = []
        with torch.no_grad():
            if self.model_type == "c":
                logits, cog_history = self.model(batch_emb, lengths, return_cognitive_states=True)
                for c_t in cog_history:
                    c_norm = torch.norm(c_t, p=2, dim=-1).item()
                    cognitive_norms.append(round(c_norm, 4))
            else:
                logits = self.model(batch_emb, lengths)

            probs = torch.softmax(logits, dim=-1)[0]
            pred_class = torch.argmax(logits, dim=-1).item()

        human_prob = probs[0].item()
        ai_prob = probs[1].item()
        label_str = "AI-Generated" if pred_class == 1 else "Human-Written"

        res = {
            'model_type': self.model_type.upper(),
            'prediction': label_str,
            'class_id': pred_class,
            'human_confidence': round(human_prob * 100, 2),
            'ai_confidence': round(ai_prob * 100, 2),
            'num_paragraphs': len(paragraphs),
            'paragraphs': paragraphs
        }

        if self.model_type == "c":
            res['cognitive_shift_magnitudes'] = cognitive_norms
            res['mean_cognitive_magnitude'] = round(sum(cognitive_norms) / len(cognitive_norms), 4) if cognitive_norms else 0.0

        return res


def run_comparison(text: str, device: torch.device):
    """Executes both Model B and Model C on the input text and prints side-by-side results."""
    print("\n--- Running Comparative Inference (Model B vs Model C) ---")
    pred_b = AssignmentPredictor(model_type="b", device=device)
    res_b = pred_b.predict(text)

    pred_c = AssignmentPredictor(model_type="c", device=device)
    res_c = pred_c.predict(text)

    print("\n" + "="*70)
    print("                COMPARATIVE INFERENCE ANALYSIS")
    print("="*70)
    print(f"Total Paragraphs Segmented: {res_c['num_paragraphs']}")
    print("-" * 70)
    print(f"{'Metric / Feature':<30} | {'Model B (GRU Baseline)':<17} | {'Model C (Proposed CCRM)':<17}")
    print("-" * 70)
    print(f"{'Predicted Class':<30} | {res_b['prediction']:<17} | {res_c['prediction']:<17}")
    print(f"{'Human Confidence':<30} | {res_b['human_confidence']:>15.2f}% | {res_c['human_confidence']:>15.2f}%")
    print(f"{'AI Confidence':<30} | {res_b['ai_confidence']:>15.2f}% | {res_c['ai_confidence']:>15.2f}%")
    print("-" * 70)
    if 'cognitive_shift_magnitudes' in res_c:
        print("Cognitive Transition Dynamics (Model C CCRM):")
        for i, norm in enumerate(res_c['cognitive_shift_magnitudes'], 1):
            bar = "█" * int(norm * 4)
            print(f"  Paragraph {i} Shift Magnitude ||C_{i}||: {norm:6.3f}  {bar}")
        print(f"  Mean Shift Across Discourse: {res_c['mean_cognitive_magnitude']:6.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Classify academic assignment using Model C (CCRM) or Model B")
    parser.add_argument("--model", type=str, choices=["c", "b"], default="c", help="Model architecture ('c' for CCRM, 'b' for GRU baseline)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom checkpoint file")
    parser.add_argument("--text", type=str, help="Raw text of the assignment")
    parser.add_argument("--file", type=str, help="Path to text file containing assignment")
    parser.add_argument("--compare", action="store_true", help="Compare both Model B and Model C side-by-side")
    args = parser.parse_args()

    device = get_device()

    if not args.text and not args.file:
        # Sample academic essay
        args.text = (
            "Climate change is accelerating due to human-induced greenhouse gas emissions across the industrial sector.\n\n"
            "Rising global temperatures significantly disrupt agricultural cycles, reducing crop yields and destabilizing regional food security.\n\n"
            "Consequently, vulnerable communities face compounding socioeconomic distress, necessitating adaptive infrastructure and international policy reforms."
        )
        print("No input specified. Using sample academic assignment text.\n")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            args.text = f.read()

    if args.compare:
        run_comparison(args.text, device)
        return

    predictor = AssignmentPredictor(model_type=args.model, checkpoint_path=args.checkpoint, device=device)
    result = predictor.predict(args.text)

    print("\n" + "="*60)
    print(f"           MODEL {result['model_type']} INFERENCE REPORT")
    print("="*60)
    print(f"Classification : {result['prediction']}")
    print(f"Human Prob     : {result['human_confidence']}%")
    print(f"AI Prob        : {result['ai_confidence']}%")
    print(f"Paragraphs     : {result['num_paragraphs']}")
    print("-" * 60)
    for i, p in enumerate(result['paragraphs'], 1):
        preview = p[:80] + "..." if len(p) > 80 else p
        print(f"   [P{i}] {preview}")
    
    if 'cognitive_shift_magnitudes' in result:
        print("-" * 60)
        print("Cognitive Memory Module Transition Analysis:")
        for idx, norm in enumerate(result['cognitive_shift_magnitudes'], 1):
            bar = "■" * int(norm * 5)
            print(f"   Para {idx} Shift ||C_{idx}||: {norm:6.3f}  {bar}")
        print(f"   Mean Cognitive Shift: {result['mean_cognitive_magnitude']:6.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
