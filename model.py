#!/usr/bin/env python3
"""
model.py
Architecture for Model B: DeBERTa-v3 -> Paragraph Embeddings -> GRU -> MLP -> Human/AI Classifier.

Model B acts as the memory-only baseline.
It processes an assignment paragraph-by-paragraph sequentially using a GRU to maintain document-level memory.
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


class ModelB_GRU(nn.Module):
    """
    Model B: Memory-Only Baseline Classifier
    Takes pre-extracted paragraph embeddings [batch_size, max_num_paras, 768]
    and paragraph sequence lengths [batch_size].
    Passes through GRU -> Extracts final valid memory vector -> MLP -> Logits [Human, AI].
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

        # 1. GRU Sequential Memory Module
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 2. MLP Classifier Module
        gru_out_dim = hidden_size * self.num_directions
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, paragraph_embeddings: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Model B.
        paragraph_embeddings: [batch_size, num_paras, embedding_dim]
        lengths: [batch_size] tensor of actual number of paragraphs per document
        Returns: logits [batch_size, num_classes]
        """
        batch_size, max_paras, _ = paragraph_embeddings.shape

        if lengths is not None and (lengths < max_paras).any():
            # Variable sequence length handling via pack_padded_sequence
            lengths_clamped = torch.clamp(lengths, min=1).cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                paragraph_embeddings,
                lengths_clamped,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed_input)
            
            if not self.bidirectional:
                # For single-direction GRU, hidden[-1] is the final hidden state of the last valid step
                final_memory = hidden[-1]  # [batch_size, hidden_size]
            else:
                # For bidirectional GRU, concatenate forward & backward last states
                final_memory = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            gru_output, hidden = self.gru(paragraph_embeddings)
            if not self.bidirectional:
                final_memory = hidden[-1]  # [batch_size, hidden_size]
            else:
                final_memory = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        # Apply dropout on document memory representation
        final_memory = self.dropout(final_memory)

        # Pass through MLP classifier
        logits = self.classifier(final_memory)  # [batch_size, 2]
        return logits


class EndToEndModelB(nn.Module):
    """
    End-to-End Pipeline combining DeBERTa-v3 Feature Extractor + ModelB_GRU
    Primarily used for raw text inference and full end-to-end evaluation.
    """
    def __init__(
        self,
        deberta_model_name: str = "microsoft/deberta-v3-base",
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
        dropout: float = 0.3,
        freeze_deberta: bool = True
    ):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(deberta_model_name)
        if freeze_deberta:
            for param in self.deberta.parameters():
                param.requires_grad = False
                
        self.gru_classifier = ModelB_GRU(
            embedding_dim=self.deberta.config.hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=dropout
        )

    def extract_paragraph_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts 768-dim masked mean pooled embedding for a batch of paragraphs.
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return masked_mean_pooling(token_embeddings, attention_mask)

    def forward(self, batch_para_ids: list[torch.Tensor], batch_para_masks: list[torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
        """
        batch_para_ids: list of [num_paras_i, max_len]
        batch_para_masks: list of [num_paras_i, max_len]
        lengths: [batch_size]
        """
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

        logits = self.gru_classifier(padded_embeddings, lengths)
        return logits
