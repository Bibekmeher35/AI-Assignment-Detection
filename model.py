#!/usr/bin/env python3
"""
model.py
Architecture for Model C: Proposed Cognitive Memory Module (CCRM)
Pipeline: DeBERTa-v3 -> Paragraph Embeddings -> Cognitive Operation -> GRU -> MLP -> Human/AI.

Also retains ModelB_GRU for direct baseline ablation comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def masked_mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes masked mean pooling over token embeddings.
    token_embeddings: [batch_size, seq_len, hidden_dim]
    attention_mask: [batch_size, seq_len]
    Returns: [batch_size, hidden_dim]
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class CognitiveOperation(nn.Module):
    """
    Cognitive Operation Module (CCRM):
    Explicitly analyzes the semantic transition and consistency between:
      1. Current paragraph embedding E_i in R^768
      2. Accumulated previous document memory M_(i-1) in R^256

    Steps:
      - Project E_i -> E_i' in R^256
      - Project M_(i-1) -> M_(i-1)' in R^256
      - Construct relationship features: [E_i', M_(i-1)', E_i' - M_(i-1)', E_i' * M_(i-1)'] in R^1024
      - CognitiveMLP(R_i) produces cognitive transition representation C_i in R^256
      - Project C_i back to R^768: combined_input = E_i + W_c(C_i)
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_size: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 1. Linear projections to shared cognitive space
        self.proj_e = nn.Linear(embedding_dim, hidden_size)  # 768 -> 256
        self.proj_m = nn.Linear(hidden_size, hidden_size)     # 256 -> 256

        # 2. Cognitive MLP over relationship features (256 * 4 = 1024)
        relationship_dim = hidden_size * 4
        self.cognitive_mlp = nn.Sequential(
            nn.Linear(relationship_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # 3. Controlled integration projection back to embedding space
        self.proj_c = nn.Linear(hidden_size, embedding_dim)  # 256 -> 768

    def forward(self, e_i: torch.Tensor, m_prev: torch.Tensor):
        """
        e_i: [batch_size, 768] (current paragraph embedding)
        m_prev: [batch_size, 256] (accumulated previous document memory)
        Returns:
          combined_input: [batch_size, 768]
          c_i: [batch_size, 256] (cognitive transition representation)
        """
        # Project into shared cognitive dimension
        e_proj = self.proj_e(e_i)       # [B, 256]
        m_proj = self.proj_m(m_prev)    # [B, 256]

        # Construct explicit semantic relationship features
        diff = e_proj - m_proj          # [B, 256]
        inter = e_proj * m_proj         # [B, 256]

        # Concatenate relationship representation Ri
        r_i = torch.cat([e_proj, m_proj, diff, inter], dim=-1)  # [B, 1024]

        # Pass through Cognitive MLP to extract cognitive pattern
        c_i = self.cognitive_mlp(r_i)   # [B, 256]

        # Project cognitive representation back to embedding space (residual modulation)
        delta_e = self.proj_c(c_i)      # [B, 768]
        combined_input = e_i + delta_e  # [B, 768]

        return combined_input, c_i


class ModelC_CCRM(nn.Module):
    """
    Model C: Proposed Cognitive Memory Module Architecture
    Pipeline:
      E_i + M_(i-1) -> CognitiveOperation -> C_i -> Memory Update (GRU) -> M_i
      M_final -> Dropout -> MLP Classifier -> Human / AI
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_size: int = 256,
        mlp_hidden: int = 128,
        num_classes: int = 2,
        cognitive_dropout: float = 0.2,
        classifier_dropout: float = 0.3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # 1. Novel Cognitive Operation Component
        self.cognitive_op = CognitiveOperation(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=cognitive_dropout
        )

        # 2. Sequential GRU Memory Cell
        self.gru_cell = nn.GRUCell(
            input_size=embedding_dim,
            hidden_size=hidden_size
        )

        # 3. MLP Classifier Head (Identical to Model B for fair ablation)
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(
        self,
        paragraph_embeddings: torch.Tensor,
        lengths: torch.Tensor = None,
        return_cognitive_states: bool = False
    ):
        """
        paragraph_embeddings: [batch_size, max_paras, embedding_dim]
        lengths: [batch_size] tensor of actual number of paragraphs per document
        return_cognitive_states: if True, returns cognitive vectors C_i along with logits
        """
        batch_size, max_paras, _ = paragraph_embeddings.shape
        device = paragraph_embeddings.device

        if lengths is None:
            lengths = torch.full((batch_size,), max_paras, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)

        # Initial memory M0 = 0
        current_memory = torch.zeros(batch_size, self.hidden_size, device=device)
        cognitive_history = []

        # Sequential processing paragraph by paragraph
        for step in range(max_paras):
            # Active mask for documents that have valid paragraphs at this step
            active_mask = (lengths > step).unsqueeze(-1)  # [batch_size, 1]

            e_i = paragraph_embeddings[:, step, :]  # [batch_size, 768]

            # 1. Cognitive Operation
            combined_input, c_i = self.cognitive_op(e_i, current_memory)
            if return_cognitive_states:
                cognitive_history.append(c_i)

            # 2. Memory Update via GRU Cell
            cand_memory = self.gru_cell(combined_input, current_memory)

            # 3. Apply active mask (retain previous state if document has ended)
            current_memory = torch.where(active_mask, cand_memory, current_memory)

        final_memory = current_memory  # M_final = M_n in R^256

        # MLP Classification Head
        dropped_memory = self.dropout(final_memory)
        logits = self.classifier(dropped_memory)  # [batch_size, num_classes]

        if return_cognitive_states:
            return logits, cognitive_history
        return logits


class ModelB_GRU(nn.Module):
    """
    Model B: Memory-Only Baseline Classifier (Retained for ablation comparison)
    Takes pre-extracted paragraph embeddings [batch_size, max_num_paras, 768]
    Passes directly through GRU -> Extracts final memory vector -> MLP -> Logits [Human, AI].
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_size: int = 256,
        num_layers: int = 1,
        mlp_hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        gru_out_dim = hidden_size * self.num_directions
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, paragraph_embeddings: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch_size, max_paras, _ = paragraph_embeddings.shape

        if lengths is not None and (lengths < max_paras).any():
            lengths_clamped = torch.clamp(lengths, min=1).cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                paragraph_embeddings,
                lengths_clamped,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed_input)
            final_memory = hidden[-1]
        else:
            gru_output, hidden = self.gru(paragraph_embeddings)
            final_memory = hidden[-1]

        final_memory = self.dropout(final_memory)
        logits = self.classifier(final_memory)
        return logits


class EndToEndModelC(nn.Module):
    """
    End-to-End Pipeline combining DeBERTa-v3 Feature Extractor + ModelC_CCRM
    Used for raw text inference and live demonstrations.
    """
    def __init__(
        self,
        deberta_model_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 256,
        cognitive_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
        freeze_deberta: bool = True
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(deberta_model_name)
        if freeze_deberta:
            for param in self.deberta.parameters():
                param.requires_grad = False

        self.ccrm_classifier = ModelC_CCRM(
            embedding_dim=self.deberta.config.hidden_size,
            hidden_size=hidden_size,
            cognitive_dropout=cognitive_dropout,
            classifier_dropout=classifier_dropout
        )

    def extract_paragraph_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return masked_mean_pooling(token_embeddings, attention_mask)

    def forward(self, batch_para_ids: list[torch.Tensor], batch_para_masks: list[torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
        batch_size = len(batch_para_ids)
        max_paras = max(lengths).item()
        hidden_dim = self.deberta.config.hidden_size
        device = next(self.parameters()).device

        padded_embeddings = torch.zeros(batch_size, max_paras, hidden_dim, device=device)

        for i in range(batch_size):
            if lengths[i] > 0:
                p_ids = batch_para_ids[i].to(device)
                p_masks = batch_para_masks[i].to(device)
                p_embs = self.extract_paragraph_embedding(p_ids, p_masks)
                padded_embeddings[i, :lengths[i]] = p_embs

        logits = self.ccrm_classifier(padded_embeddings, lengths)
        return logits
