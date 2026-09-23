# Model C: Cognitive Memory Module (CCRM) for AI-Generated Assignment Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue.svg)]()
[![Model Size](https://img.shields.io/badge/Model%20Weights-6.19%20MB-green.svg)]()
[![Test F1](https://img.shields.io/badge/Test%20F1--Score-99.75%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9990-brightgreen.svg)]()

Official implementation of **Model C: Cognitive Contextual Representation & Memory Module (CCRM)** for the research project:
> **"Detecting AI-Generated Assignments Using Cognitive Pattern Analysis"**

Model C is the proposed, advanced architecture of this research work. It extends the sequential recurrent baseline (**Model B**) by introducing explicit paragraph-to-memory interaction modeling to capture subtle discourse transitions, conceptual evolution, and semantic progression across academic essays.

---

## 📌 Executive Summary & Key Results

| Metric | Model B (Baseline GRU) | Model C (Proposed CCRM) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Architecture** | Frozen DeBERTa + GRU + MLP | Frozen DeBERTa + CCRM + GRU + MLP | Explicit Cognitive Operation |
| **Trainable Parameters** | 821,122 (~0.82 M) | 1,609,346 (~1.61 M) | +788,224 params (~0.79 M) |
| **Checkpoint Size** | 3.18 MB | 6.19 MB | Lightweight & deployable |
| **Test Accuracy** | 99.75% | **99.75%** | Robust generalization |
| **Test Precision (Human)** | 100.0% | **100.0%** | Zero false accusations |
| **Test Recall (AI Caught)** | 99.50% | **99.50%** | 199 / 200 AI caught |
| **Test F1-Score** | 99.75% | **99.75%** | Optimal balance |
| **ROC-AUC** | 0.9975 | **0.9990** | **Superior threshold separability (+0.0015)** |
| **Inference Latency** | 5.08 ms/doc | **0.44 ms/doc** | Real-time capable |
| **Shuffled Text AUC** | 0.9980 | **0.9994** | **Higher discourse robustness** |

---

## 🔬 What Exactly is New in Model C Compared to Model B?

While Model B answers whether *maintaining sequential recurrent memory* outperforms flat bag-of-words or mean-pooled models, Model C investigates:
> *"Does adding an explicit cognitive/semantic transition operator ($E_i \leftrightarrow M_{i-1}$) improve AI-generated assignment detection and discourse robustness beyond standard recurrent transitions alone?"*

| Dimension | Model B (Sequential Baseline) | Model C (Proposed CCRM) |
| :--- | :--- | :--- |
| **Core Input to Memory** | Raw paragraph vector $E_i \in \mathbb{R}^{768}$ is fed directly to GRU | Residual cognitively modulated vector $\widetilde{E}_i = E_i + W_c C_i$ is fed to GRU |
| **Discourse Comparison** | Implicit only inside standard GRU reset/update gates | Explicit multi-perspective interaction: difference ($E_i' - M_{i-1}'$) and Hadamard product ($E_i' \odot M_{i-1}'$) |
| **Semantic Drift Modeling** | Passive accumulation in hidden state | Active computation of conceptual jump magnitude $\|C_i\|_2$ |
| **Discourse Perturbation** | Sensitive to sequence reversal | Maintains higher ranking separation (AUC 0.9994 on shuffled text) |
| **Interpretability** | Black-box hidden state sequence | Paragraph-by-paragraph cognitive transition dynamics inspectable via $\|C_i\|_2$ |

> [!NOTE]
> **Scientific Clarification**: In this research work, the term **"Cognitive Pattern"** refers to the computational structure of discourse: how ideas develop, how context is accumulated, and how consecutive paragraphs transition semantically. It does **not** claim to simulate biological human neurons or cognitive neurobiology.

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
     ║             ⭐ NOVEL COGNITIVE MEMORY MODULE (CCRM)                 ║
     ║                                                                     ║
     ║  For each paragraph step i = 1, 2, ..., n:                          ║
     ║                                                                     ║
     ║  1. Dual Linear Projections:                                        ║
     ║     Eᵢ'     = Linear(768 ──► 256)(Eᵢ)                               ║
     ║     Mᵢ₋₁'   = Linear(256 ──► 256)(Mᵢ₋₁)                             ║
     ║                                                                     ║
     ║  2. Multi-Perspective Interaction Vector:                           ║
     ║     Diff    = Eᵢ' - Mᵢ₋₁'                                           ║
     ║     Inter   = Eᵢ' ⊙ Mᵢ₋₁'  (Element-wise Hadamard product)          ║
     ║     Rᵢ      = [ Eᵢ' ∥ Mᵢ₋₁' ∥ Diff ∥ Inter ]  ∈ ℝ¹⁰²⁴               ║
     ║                                                                     ║
     ║  3. Cognitive Transition Representation:                            ║
     ║     Cᵢ      = CognitiveMLP(1024 ──► 256)(Rᵢ)                        ║
     ║                                                                     ║
     ║  4. Residual Additive Modulation:                                   ║
     ║     ΔEᵢ     = Linear(256 ──► 768)(Cᵢ)                               ║
     ║     Ẽᵢ      = Eᵢ + ΔEᵢ  ∈ ℝ⁷⁶⁸                                      ║
     ║                                                                     ║
     ║  5. Recurrent Context Update:                                       ║
     ║     Mᵢ      = GRUCell(Ẽᵢ, Mᵢ₋₁)  ∈ ℝ²⁵⁶                             ║
     ╚═══════════════════════════════════════╤═════════════════════════════╝
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │       Final Document Memory            │
                         │                                        │
                         │  M_final = Mₙ ∈ ℝ²⁵⁶                   │
                         │  • Contextual Discourse State          │
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
                         │             Output Logits              │
                         │                                        │
                         │  Softmax(z) ──► [P(Human), P(AI)]      │
                         │  argmax(z)  ──► 0: Human | 1: AI       │
                         └────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation of CCRM

For an assignment composed of $n$ ordered paragraphs $\{P_1, P_2, \dots, P_n\}$:

### 1. Semantic Embedding Extraction
Each paragraph $P_i$ is encoded by frozen DeBERTa-v3-base and pooled using attention-masked mean pooling:
$$E_i = \text{MaskedMeanPool}(\text{DeBERTa}(P_i)) \in \mathbb{R}^{768}$$

### 2. Cognitive State Projection
At step $i$, with accumulated discourse memory $M_{i-1} \in \mathbb{R}^{256}$ (initialized to $M_0 = \mathbf{0}$):
$$E_i' = W_e E_i + b_e \in \mathbb{R}^{256}$$
$$M_{i-1}' = W_m M_{i-1} + b_m \in \mathbb{R}^{256}$$

### 3. Multi-Perspective Interaction Vector ($R_i$)
To capture both absolute states and relative shifts between the incoming paragraph and the historical context:
$$R_i = \Big[ E_i' \;\parallel\; M_{i-1}' \;\parallel\; (E_i' - M_{i-1}') \;\parallel\; (E_i' \odot M_{i-1}') \Big] \in \mathbb{R}^{1024}$$
where $\parallel$ denotes tensor concatenation, $(E_i' - M_{i-1}')$ measures directional semantic displacement, and $(E_i' \odot M_{i-1}')$ captures semantic alignment.

### 4. Cognitive Operation ($C_i$)
The concatenated relationship vector passes through a nonlinear feedforward network:
$$C_i = \text{Dropout}\Big(\text{ReLU}\big(W_r R_i + b_r\big)\Big) \in \mathbb{R}^{256}$$
The vector $C_i$ explicitly represents the **cognitive transition** introduced by paragraph $P_i$.

### 5. Residual Feature Modulation ($\widetilde{E}_i$)
Rather than replacing the dense semantic information from DeBERTa, $C_i$ acts as an additive residual modulation:
$$\Delta E_i = W_c C_i + b_c \in \mathbb{R}^{768}$$
$$\widetilde{E}_i = E_i + \Delta E_i \in \mathbb{R}^{768}$$

### 6. Contextual Memory Update ($M_i$)
The modulated paragraph representation updates the sequential memory via a Gated Recurrent Unit:
$$M_i = \text{GRUCell}(\widetilde{E}_i, M_{i-1}) \in \mathbb{R}^{256}$$

### 7. Document Classification
After processing all $n$ paragraphs, the final memory vector $M_n \in \mathbb{R}^{256}$ summarizes the entire discourse:
$$h = \text{Dropout}\big(\text{ReLU}(W_1 M_n + b_1)\big) \in \mathbb{R}^{128}$$
$$z = W_2 h + b_2 \in \mathbb{R}^2$$
$$P(\text{class}) = \text{Softmax}(z)$$

---

## 📊 Experimental Dataset

All models were trained, validated, and evaluated on the identical stratified benchmark:

| Split | Total Documents | Human-Written | AI-Generated | Generators Covered |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 1,200 | 600 (50.0%) | 600 (50.0%) | GPT-4o, Gemma-2-9B, Llama-3-8B, Mistral-7B, Qwen-2-72B, Yi-Large |
| **Validation** | 300 | 150 (50.0%) | 150 (50.0%) | Balanced distribution across all generators |
| **Test** | 400 | 200 (50.0%) | 200 (50.0%) | 200 Human + 200 AI across 6 LLMs |

### Test Set Generator Breakdown (Model C CCRM)

| Generator / Family | Test Count | Detected as AI | Misclassified | Accuracy (%) | Mean $P(\text{AI})$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Human-Written** | 200 | 0 | 0 | **100.0%** | **0.64%** |
| **Mistral-7B** | 43 | 43 | 0 | **100.0%** | **99.91%** |
| **Qwen-2-72B** | 23 | 23 | 0 | **100.0%** | **99.91%** |
| **Gemma-2-9B** | 27 | 27 | 0 | **100.0%** | **99.82%** |
| **Llama-3-8B** | 32 | 32 | 0 | **100.0%** | **99.57%** |
| **GPT-4o** | 36 | 36 | 0 | **100.0%** | **97.34%** |
| **Yi-Large** | 39 | 38 | 1 | **97.44%** | **97.28%** |

---

## 🧪 Discourse Perturbation & Robustness Study

To evaluate whether the models rely on superficial sequence order or genuine discourse coherence, we conducted perturbation stress testing across three experimental conditions:
1. **Intact Discourse**: Natural, coherent paragraph ordering as written.
2. **Shuffled Discourse**: Paragraphs randomly permuted, breaking rhetorical flow.
3. **Reversed Discourse**: Paragraph order inverted ($P_n \to P_1$).

### Robustness Results Summary

```
ROC-AUC Under Discourse Perturbations:
Intact Coherent Text:
  Model B (GRU Baseline)   [███████████████████████████████████████ 0.9975]
  Model C (Proposed CCRM)  [████████████████████████████████████████ 0.9990]  (+0.0015)

Disrupted / Shuffled Paragraphs:
  Model B (GRU Baseline)   [███████████████████████████████████████ 0.9980]
  Model C (Proposed CCRM)  [████████████████████████████████████████ 0.9994]  (+0.0014)

Reversed Discourse Flow:
  Model B (GRU Baseline)   [███████████████████████████████████████ 0.9978]
  Model C (Proposed CCRM)  [████████████████████████████████████████ 0.9990]  (+0.0012)
```

### Key Discourse Insights
- **Model C maintains higher ROC-AUC in every condition ($0.9990 - 0.9994$)**, demonstrating superior probabilistic separability between human and machine text.
- Because Model C computes explicit difference $(E_i' - M_{i-1}')$ and interaction $(E_i' \odot M_{i-1}')$ vectors, it immediately flags unnatural semantic jumps or robotic uniform transitions, even when the overall sequence order is scrambled.

---

## 🗂️ Repository Directory Structure

```
Model C/
├── model.py                     # PyTorch architecture (CognitiveOperation, ModelC_CCRM, ModelB_GRU)
├── train.py                     # Training script with AdamW, Cosine Annealing, early stopping
├── evaluate.py                  # Full test evaluation with generator breakdown and metrics
├── compare_models.py            # Automated ablation benchmarking (Model B vs Model C)
├── robustness_test.py           # Discourse perturbation testing (Intact, Shuffled, Reversed)
├── step_by_step_demo.py         # Step-by-step mathematical tensor tracing on sample essay
├── predict.py                   # Live CLI inference with cognitive transition dynamics
├── data_prep.py                 # Paragraph segmenter and DeBERTa tokenization utilities
├── sample_essay.txt             # Multi-paragraph sample academic essay
├── model_c_ccrm.pth             # Trained Model C checkpoint (6.19 MB)
├── test_evaluation_report_model_c.json  # Comprehensive test report
├── model_b_vs_c_comparison.json        # Comparative benchmark JSON (Model B vs Model C)
├── robustness_study_results.json       # Perturbation test report JSON
├── data/                        # Pre-extracted DeBERTa paragraph embeddings
│   ├── train_embeddings.pt      # 1,200 documents
│   ├── val_embeddings.pt        # 300 documents
│   └── test_embeddings.pt       # 400 documents
└── README.md                    # Documentation
```

---

## 🚀 Step-by-Step Usage & Replication

### 1. Verify Architecture Shapes
```bash
python3 train.py --test_mock
```

### 2. Train Model C
```bash
python3 train.py --epochs 10 --batch_size 16 --lr 0.0003
```
*Best checkpoint will be saved to `model_c_ccrm.pth`.*

### 3. Evaluate Model C on Test Set
```bash
python3 evaluate.py --checkpoint model_c_ccrm.pth
```

### 4. Run Model B vs Model C Ablation Comparison
```bash
python3 compare_models.py
```

### 5. Run Discourse Robustness Perturbation Study
```bash
python3 robustness_test.py
```

### 6. Verify Mathematical Formulations Step-by-Step
```bash
python3 step_by_step_demo.py --file sample_essay.txt
```

### 7. Run Live Document Inference with Side-by-Side Comparison
```bash
python3 predict.py --file sample_essay.txt --compare
```
*Sample output:*
```text
======================================================================
                COMPARATIVE INFERENCE ANALYSIS
======================================================================
Total Paragraphs Segmented: 5
----------------------------------------------------------------------
Metric / Feature               | Model B (GRU Baseline) | Model C (Proposed CCRM)
----------------------------------------------------------------------
Predicted Class                | AI-Generated      | AI-Generated     
Human Confidence               |            0.05% |            0.41%
AI Confidence                  |           99.95% |           99.59%
----------------------------------------------------------------------
Cognitive Transition Dynamics (Model C CCRM):
  Paragraph 1 Shift Magnitude ||C_1||: 11.143  ████████████████████████████████████████████
  Paragraph 2 Shift Magnitude ||C_2||:  7.603  ██████████████████████████████
  Paragraph 3 Shift Magnitude ||C_3||:  6.199  ████████████████████████
  Paragraph 4 Shift Magnitude ||C_4||:  4.015  ████████████████
  Paragraph 5 Shift Magnitude ||C_5||:  2.304  █████████
  Mean Shift Across Discourse:  6.253
======================================================================
```

---

## ⚖️ Ethical Guidelines & Responsible Use

1. **Zero False Accusations Target**: Model C achieved **100.0% precision** on human-written academic assignments (0 false alarms out of 200). In real educational deployments, any detection tool should be used as an *assistive flag for human instructor review*, never for automated disciplinary action.
2. **Discourse Focus**: Model C analyzes paragraph-level flow and semantic transitions rather than penalizing advanced academic vocabulary, ensuring native and non-native English writers are treated equitably.
