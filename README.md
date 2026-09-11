# AI-Generated Assignment Detection (Model B: Memory-Only Baseline)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue.svg)]()
[![Model Size](https://img.shields.io/badge/Model%20Weights-3.18%20MB-green.svg)]()
[![Test F1](https://img.shields.io/badge/Test%20F1--Score-99.75%25-brightgreen.svg)]()

This repository contains the official implementation of **Model B: Memory-Only Baseline** (`DeBERTa-v3 → Paragraph Embeddings → GRU → MLP → Human/AI`) for detecting AI-generated academic assignments.

Model B answers the core research question:
> *"Does maintaining sequential recurrent memory over paragraph embeddings improve AI-generated assignment detection compared to flat document representations?"*

---

## 🏗️ Architecture Pipeline

```
                         ┌────────────────────────────────────────┐
                         │          Academic Assignment           │
                         │          Raw Text / Document           │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │           Document Parser              │
                         │        Paragraph Segmentation          │
                         │                                        │
                         │  • Line-wrap normalization             │
                         │  • Discourse boundary detection        │
                         │  • Output: [P₁, P₂, P₃, ..., Pₙ]       │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │        Paragraph Preprocessing         │
                         │                                        │
                         │  • SentencePiece Tokenization          │
                         │  • Truncation / Padding (max 512 tok)  │
                         │  • Attention Mask Generation           │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │      DeBERTa-v3-base (Frozen)          │
                         │          Semantic Encoder              │
                         │                                        │
                         │  • Hidden dimension: d = 768           │
                         │  • Masked Mean Pooling                 │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │        Paragraph Embeddings            │
                         │                                        │
                         │  P₁  ──►  E₁ ∈ ℝ⁷⁶⁸                    │
                         │  P₂  ──►  E₂ ∈ ℝ⁷⁶⁸                    │
                         │  P₃  ──►  E₃ ∈ ℝ⁷⁶⁸                    │
                         │  ... ──►  ...                          │
                         │  Pₙ  ──►  Eₙ ∈ ℝ⁷⁶⁸                    │
                         │                                        │
                         │  Sequence Shape: [Batch, n, 768]       │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
     ╔═════════════════════════════════════════════════════════════════════╗
     ║                  ⭐ RECURRENT MEMORY MODULE                         ║
     ║                       (1-Layer GRU)                                 ║
     ║                                                                     ║
     ║     Step 1:   E₁  +  M₀(0)   ──►  GRU  ──►  M₁ ∈ ℝ²⁵⁶               ║
     ║     Step 2:   E₂  +  M₁      ──►  GRU  ──►  M₂ ∈ ℝ²⁵⁶               ║
     ║     Step 3:   E₃  +  M₂      ──►  GRU  ──►  M₃ ∈ ℝ²⁵⁶               ║
     ║       ...      ...   ...           ...      ...                 ║
     ║     Step n:   Eₙ  +  Mₙ₋₁    ──►  GRU  ──►  Mₙ ∈ ℝ²⁵⁶               ║
     ║                                                                     ║
     ║  • input_size = 768,  hidden_size = 256                             ║
     ║  • Dynamically extracts valid final step for variable lengths       ║
     ╚═══════════════════════════════════════╤═════════════════════════════╝
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │       Final Document Memory            │
                         │                                        │
                         │  M_final = Mₙ ∈ ℝ²⁵⁶                   │
                         │  • Sequential Context Representation   │
                         │  • Dropout (p = 0.3)                   │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │         MLP Classification Head        │
                         │                                        │
                         │  • Linear(256 ──► 128)                 │
                         │  • ReLU Activation                     │
                         │  • Dropout(p = 0.3)                    │
                         │  • Linear(128 ──► 2)                   │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │            Classification              │
                         │                                        │
                         │  Logits: [Human Score, AI Score]       │
                         │  Softmax Probability Output            │
                         │                                        │
                         │      ├── Human-Written (0)             │
                         │      └── AI-Generated  (1)             │
                         └────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Model/
├── data_prep.py                  # Parses raw CSV, extracts paragraphs, creates leak-free splits
├── extract_embeddings.py         # Batch extracts 768-dim paragraph vectors via DeBERTa-v3
├── model.py                      # ModelB_GRU architecture & EndToEndModelB pipeline
├── train.py                      # Training loop with validation metrics, early stopping & checkpointing
├── evaluate.py                   # Test set evaluation suite with per-generator breakdown
├── experiments.py                # Systematic ablation study suite (B1, B2, B3)
├── predict.py                    # Real-time inference CLI for text or text files
├── step_by_step_demo.py          # Interactive demonstration logging data at every pipeline step
├── model_b_gru.pth               # Best trained model weights (3.18 MB)
├── test_evaluation_report.json   # Full test evaluation report
├── requirements.txt              # Python dependencies
├── sample_essay.txt              # Sample academic assignment for testing
└── data/                         # Train, val, and test splits (CSV & JSONL)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd Model
pip install -r requirements.txt
```

### 2. Run Step-by-Step Pipeline Demonstration

Inspect the exact data, tokens, and tensor state transitions across all 7 pipeline steps:

```bash
python3 step_by_step_demo.py --file sample_essay.txt
```

### 3. Real-Time Inference on Any Text

```bash
# Pass raw text
python3 predict.py --text "First paragraph...\n\nSecond paragraph...\n\nThird paragraph..."

# Pass an assignment file
python3 predict.py --file path/to/assignment.txt
```

### 4. Re-Train Model B

```bash
# 1. Prepare data splits
python3 data_prep.py

# 2. Pre-extract DeBERTa embeddings (fast caching)
python3 extract_embeddings.py --batch_size 32

# 3. Train Model B
python3 train.py --epochs 12 --batch_size 32
```

### 5. Evaluate on Test Set

```bash
python3 evaluate.py --checkpoint model_b_gru.pth --test_pt data/test_embeddings.pt
```

---

## 📊 Empirical Results

### Test Set Performance ($N = 400$)

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **99.75%** |
| **Precision** | **100.00%** |
| **Recall** | **99.50%** |
| **F1-Score** | **99.75%** |
| **ROC-AUC** | **0.9975** |

### Confusion Matrix
```
                 Predicted Human    Predicted AI
True Human (200)      200                 0     (0 False Positives / 100% Specificity)
True AI (200)           1               199     (1 False Negative  / 99.5% Sensitivity)
```

### Breakdown Across AI Generators

| Generator | Sample Count | Accuracy | Average P(AI) |
| :--- | :---: | :---: | :---: |
| **Human Essays** | 200 | **100.00%** | $0.44\%$ |
| **GPT-4o** | 36 | **100.00%** | $96.01\%$ |
| **Gemma-2-9B** | 27 | **100.00%** | $99.94\%$ |
| **Llama-3-8B** | 32 | **100.00%** | $99.96\%$ |
| **Mistral-7B** | 43 | **100.00%** | $99.98\%$ |
| **Qwen-2-72B** | 23 | **100.00%** | $99.97\%$ |
| **Yi-Large** | 39 | **97.44%** | $97.42\%$ |

---

## 🔬 Ablation Studies

| Experiment ID | Architecture Configuration | Hidden Size | GRU Layers | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_Baseline** | DeBERTa + GRU (256) + MLP | 256 | 1 | 98.75 | 100.00 | 97.50 | 98.73 | 0.9945 |
| **B2_Hidden_128** | DeBERTa + GRU (128) + MLP | 128 | 1 | 99.50 | 99.50 | 99.50 | 99.50 | 0.9990 |
| **B2_Hidden_512** | DeBERTa + GRU (512) + MLP | 512 | 1 | **99.75** | **100.00** | **99.50** | **99.75** | **0.9984** |
| **B3_Layers_2** | DeBERTa + GRU (256, 2-layer) + MLP | 256 | 2 | 99.50 | 100.00 | 99.00 | 99.50 | 0.9972 |

---

## ⏱️ Complexity & Resource Metrics

* **Trainable Parameters**: $821,122$ ($0.82\text{ M}$)
* **Checkpoint File Size**: $3.18\text{ MB}$
* **End-to-End Latency**: $\approx 73.5\text{ ms}$ per assignment
* **GRU + MLP Standalone Latency**: $\approx 2.48\text{ ms}$ (for 10 paragraphs)
* **Runtime Memory**: $\approx 1.2\text{ GB}$ RAM / VRAM

---

## 📜 Citation / Major Project Context

This model forms the **Model B (Memory-Only Baseline)** milestone for the research study:
1. **Model A**: DeBERTa + MLP (Bag-of-tokens / Document Baseline)
2. **Model B**: DeBERTa + GRU + MLP (Sequential Memory Baseline - *This Model*)
3. **Model C**: DeBERTa + CCRM + MLP (Consistency-Aware Cognitive Reasoning Module - *Final Proposed Model*)
