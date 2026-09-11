#!/usr/bin/env python3
"""
step_by_step_demo.py
Interactive Step-by-Step Pipeline Demonstration for Model B.
Displays the exact data, tensor shapes, and state transitions at each stage:
  1. Document Processing & Paragraph Extraction (P1, P2, P3, ...)
  2. Tokenization & Attention Masks
  3. DeBERTa-v3 Semantic Paragraph Embeddings (E1, E2, E3, ...) [768-D]
  4. Step-by-step Sequential GRU Memory Updates (M1, M2, ..., Mn) [256-D]
  5. Final Memory Vector (Mn)
  6. MLP Classification Layers (Linear -> ReLU -> Dropout -> Linear)
  7. Final Softmax Probability & Classification (Human vs AI)
"""

import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from model import ModelB_GRU, masked_mean_pooling
from data_prep import extract_paragraphs
from train import get_device


def print_banner(title: str, step_num: int):
    print("\n" + "=" * 78)
    print(f"  STEP {step_num}: {title.upper()}")
    print("=" * 78)


def run_step_by_step_demo(
    input_text: str = None,
    file_path: str = None,
    checkpoint_path: str = "model_b_gru.pth",
    model_name: str = "microsoft/deberta-v3-base"
):
    device = get_device()
    print(f"\n[Compute Device]: {device}")

    # Load input text
    if file_path and os.path.exists(file_path):
        print(f"[Loading Document]: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    elif input_text:
        raw_text = input_text
    else:
        # Default sample text
        default_file = "/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/sample_essay.txt"
        if os.path.exists(default_file):
            with open(default_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            raw_text = (
                "Artificial intelligence is transforming education by providing personalized learning.\n\n"
                "However, students may rely too heavily on automated writing tools rather than developing critical thinking.\n\n"
                "Therefore, educators must establish clear guidelines to balance technological benefits with authentic learning."
            )

    # =========================================================================
    # STEP 1: DOCUMENT PROCESSING & PARAGRAPH EXTRACTION
    # =========================================================================
    print_banner("Document Processing & Paragraph Segmentation", 1)
    print(f"Raw Document Length: {len(raw_text)} characters")
    
    paragraphs = extract_paragraphs(raw_text)
    print(f"Extracted {len(paragraphs)} Paragraphs:\n")
    for idx, p in enumerate(paragraphs, 1):
        preview = p if len(p) <= 110 else p[:107] + "..."
        print(f"  [P{idx}] ({len(p.split())} words): \"{preview}\"")

    # =========================================================================
    # STEP 2: TOKENIZATION & ATTENTION MASKS
    # =========================================================================
    print_banner("Tokenization & Input Formatting", 2)
    print(f"Loading Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded = tokenizer(
        paragraphs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    print(f"Input IDs Tensor Shape       : {list(encoded['input_ids'].shape)}  ([num_paras, max_tokens])")
    print(f"Attention Mask Tensor Shape  : {list(encoded['attention_mask'].shape)}")
    
    for idx in range(len(paragraphs)):
        token_count = encoded['attention_mask'][idx].sum().item()
        first_few_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][idx][:8].tolist())
        print(f"  • P{idx+1}: {token_count:>3} active tokens | First tokens: {first_few_tokens}")

    # =========================================================================
    # STEP 3: DeBERTa-v3 SEMANTIC PARAGRAPH EMBEDDINGS (E1, E2, ..., En)
    # =========================================================================
    print_banner("DeBERTa-v3 Encoding & Masked Mean Pooling", 3)
    print(f"Loading Transformer: {model_name}...")
    deberta = AutoModel.from_pretrained(model_name).to(device)
    deberta.eval()

    with torch.no_grad():
        transformer_outputs = deberta(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
        token_embeddings = transformer_outputs.last_hidden_state  # [num_paras, seq_len, 768]
        paragraph_embeddings = masked_mean_pooling(token_embeddings, encoded['attention_mask'])  # [num_paras, 768]

    print(f"Last Hidden States Shape     : {list(token_embeddings.shape)} ([num_paras, seq_len, 768])")
    print(f"Masked Mean Pooling Formula  : E_i = (Σ token_emb * mask) / (Σ mask)")
    print(f"Paragraph Embeddings Matrix  : {list(paragraph_embeddings.shape)} (n = {len(paragraphs)}, d = 768)\n")

    for idx in range(len(paragraphs)):
        emb_sample = paragraph_embeddings[idx][:5].tolist()
        formatted_sample = ", ".join(f"{x:+.4f}" for x in emb_sample)
        print(f"  • E{idx+1} ∈ ℝ⁷⁶⁸ : [{formatted_sample}, ...]")

    # =========================================================================
    # STEP 4: STEP-BY-STEP SEQUENTIAL GRU MEMORY
    # =========================================================================
    print_banner("Sequential GRU Memory Updates", 4)
    
    # Load trained model checkpoint
    print(f"Loading Trained Checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = ModelB_GRU(
        embedding_dim=config.get('embedding_dim', 768),
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 1),
        mlp_hidden=config.get('mlp_hidden', 128),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # We step through the GRU cell manually for each paragraph to display step-by-step memory
    gru = model.gru
    current_memory = torch.zeros(1, 1, 256, device=device)  # Initial empty memory M0
    print(f"Initial Memory State (M0)    : zeros(1, 256)")
    print("-" * 78)

    all_memories = []
    with torch.no_grad():
        for t in range(len(paragraphs)):
            # Paragraph embedding input for step t: [1, 1, 768]
            e_t = paragraph_embeddings[t:t+1].unsqueeze(1)
            
            # Forward one step through GRU: E_t + M_(t-1) -> M_t
            gru_out, current_memory = gru(e_t, current_memory)
            m_t = current_memory[-1, 0]  # [256]
            all_memories.append(m_t)
            
            m_sample = m_t[:4].tolist()
            formatted_m = ", ".join(f"{x:+.4f}" for x in m_sample)
            print(f"  Step {t+1}: E{t+1} (768-D) + M{t} (256-D) ──► GRU ──► M{t+1} (256-D): [{formatted_m}, ...]")

    # =========================================================================
    # STEP 5: FINAL MEMORY VECTOR (Mn)
    # =========================================================================
    print_banner("Final Document Memory Vector (Mn)", 5)
    final_memory = all_memories[-1].unsqueeze(0)  # [1, 256]
    print(f"Document Representation Shape : {list(final_memory.shape)} ([batch_size=1, hidden_size=256])")
    print(f"Memory Vector Norm (L2)       : {final_memory.norm().item():.4f}")
    sample_mn = final_memory[0][:8].tolist()
    print(f"First 8 dimensions of Mn      : {[round(x, 4) for x in sample_mn]}")

    # =========================================================================
    # STEP 6: MLP CLASSIFICATION HEAD
    # =========================================================================
    print_banner("MLP Classification Head", 6)
    
    # Trace through each layer in MLP
    linear1 = model.classifier[0]  # Linear(256 -> 128)
    relu = model.classifier[1]     # ReLU
    # Dropout is identity during eval
    linear2 = model.classifier[3]  # Linear(128 -> 2)

    with torch.no_grad():
        # Layer 1: Linear 256 -> 128
        h1 = linear1(final_memory)
        print(f"  1. Linear(256 ──► 128) Output Shape : {list(h1.shape)}")
        print(f"     Sample H1 activations             : {[round(x, 4) for x in h1[0][:6].tolist()]}")
        
        # Layer 2: ReLU
        h1_act = relu(h1)
        print(f"  2. ReLU Activation Output Shape     : {list(h1_act.shape)}")
        print(f"     Positive activations count        : {(h1_act > 0).sum().item()} / 128")
        
        # Layer 3: Linear 128 -> 2 (Logits)
        logits = linear2(h1_act)
        print(f"  3. Linear(128 ──► 2) Logits Shape   : {list(logits.shape)}")
        print(f"     Raw Output Logits                 : [Human: {logits[0][0].item():+.4f}, AI: {logits[0][1].item():+.4f}]")

    # =========================================================================
    # STEP 7: SOFTMAX & FINAL CLASSIFICATION
    # =========================================================================
    print_banner("Final Classification Decision", 7)
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)[0]
        human_prob = probabilities[0].item()
        ai_prob = probabilities[1].item()
        predicted_class = torch.argmax(logits, dim=-1).item()

    pred_label = "AI-Generated" if predicted_class == 1 else "Human-Written"
    
    print(f"Softmax Probabilities:")
    print(f"  • P(Human-Written) = {human_prob * 100:>6.2f}%")
    print(f"  • P(AI-Generated)  = {ai_prob * 100:>6.2f}%\n")
    print(f"╔════════════════════════════════════════════════════════════════════╗")
    print(f"║  FINAL DECISION: {pred_label.upper():<47} ║")
    print(f"║  Confidence    : {max(human_prob, ai_prob)*100:>5.2f}%                                             ║")
    print(f"╚════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-step Model B Pipeline Execution")
    parser.add_argument("--file", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/sample_essay.txt")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="/Users/bibekmeher/Documents/COLLEGE/4th Year/7th Sem/MAJOR/Model/model_b_gru.pth")
    args = parser.parse_args()

    run_step_by_step_demo(
        input_text=args.text,
        file_path=args.file,
        checkpoint_path=args.checkpoint
    )
