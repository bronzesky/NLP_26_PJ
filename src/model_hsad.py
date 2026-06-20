"""
src/model_hsad.py

Hierarchical Sentence-Aware Detector (HSAD).
Architecture:
  1. RoBERTa-base encodes full document (≤512 tokens), output_hidden_states=True
  2. Token hidden states → sentence vectors via mean pooling over token spans
  3. 2-layer CrossSentence TransformerEncoder adds inter-sentence context
  4. Sentence-level attention → sentence AI scores (supervised by subtask C)
  5. Attention-weighted pooling → document vector → document classification head
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import RobertaModel, AutoConfig


class HSAD(nn.Module):
    def __init__(
        self,
        roberta_model_dir: str,
        cross_sent_layers: int = 2,
        cross_sent_heads: int = 8,
        dropout: float = 0.1,
        freeze_layers: int = 8,  # freeze bottom N layers of RoBERTa
    ):
        super().__init__()
        self.roberta_model_dir = str(roberta_model_dir)

        # RoBERTa backbone (load with output_hidden_states capability)
        self.encoder = RobertaModel.from_pretrained(
            roberta_model_dir,
            add_pooling_layer=False,
        )
        hidden = self.encoder.config.hidden_size  # 768

        # Freeze bottom layers to preserve fine-tuned knowledge
        if freeze_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Cross-sentence Transformer (lightweight: 2 layers)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cross_sent_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cross_sent = nn.TransformerEncoder(enc_layer, num_layers=cross_sent_layers)

        # Sentence-level attention for pooling
        self.sent_attn = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Two classification heads
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden, 2)   # sentence-level AI classifier
        self.doc_head = nn.Linear(hidden, 2)    # document-level AI classifier

        # Positional embedding for sentences (up to 128 sentences)
        self.sent_pos_emb = nn.Embedding(129, hidden)

        self._init_new_params()

    def _init_new_params(self):
        for module in [self.cross_sent, self.sent_attn, self.sent_head,
                       self.doc_head, self.sent_pos_emb]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,          # (seq_len,) — single document
        attention_mask: torch.Tensor,     # (seq_len,)
        sent_spans: list[tuple[int, int]], # list of (token_start, token_end) per sentence
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: tokenized document, shape (seq_len,), NOT batched
            attention_mask: shape (seq_len,)
            sent_spans: token index spans for each sentence in the tokenized doc
        Returns:
            doc_logits: (2,)
            sent_logits: (n_sents, 2)
            sent_attn_weights: (n_sents,)
        """
        # RoBERTa forward — add batch dim
        outputs = self.encoder(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        # token_hidden: (1, seq_len, 768) → (seq_len, 768)
        token_hidden = outputs.last_hidden_state.squeeze(0)

        # Sentence mean pooling from token spans
        sent_vecs = []
        for start, end in sent_spans:
            if end > start:
                sent_vecs.append(token_hidden[start:end].mean(0))
            else:
                sent_vecs.append(token_hidden[start:start+1].mean(0))

        if not sent_vecs:
            sent_vecs = [token_hidden[0]]  # fallback: use [CLS]

        n_sents = len(sent_vecs)
        sent_repr = torch.stack(sent_vecs)  # (n_sents, 768)

        # Add positional embeddings
        positions = torch.arange(n_sents, device=sent_repr.device).clamp(max=128)
        sent_repr = sent_repr + self.sent_pos_emb(positions)

        # Cross-sentence Transformer
        ctx = self.cross_sent(sent_repr.unsqueeze(0)).squeeze(0)  # (n_sents, 768)

        # Sentence-level classification
        sent_logits = self.sent_head(self.dropout(ctx))  # (n_sents, 2)

        # Attention pooling → document vector
        attn_scores = self.sent_attn(ctx).squeeze(-1)    # (n_sents,)
        attn_weights = torch.softmax(attn_scores, dim=0) # (n_sents,)
        doc_vec = (ctx * attn_weights.unsqueeze(-1)).sum(0)  # (768,)

        # Document-level classification
        doc_logits = self.doc_head(self.dropout(doc_vec))  # (2,)

        return {
            "doc_logits": doc_logits,
            "sent_logits": sent_logits,
            "sent_attn_weights": attn_weights,
            "doc_repr": doc_vec,
        }

    def save_pretrained(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "roberta_model_dir": self.roberta_model_dir,
                "cross_sent_layers": len(self.cross_sent.layers),
                "cross_sent_heads": self.cross_sent.layers[0].self_attn.num_heads,
            },
        }, os.path.join(model_dir, "hsad_checkpoint.pt"))

    @classmethod
    def from_pretrained(cls, model_dir: str, roberta_model_dir: Optional[str] = None) -> "HSAD":
        ckpt = torch.load(os.path.join(model_dir, "hsad_checkpoint.pt"), map_location="cpu")
        cfg = ckpt["config"]
        if roberta_model_dir:
            cfg["roberta_model_dir"] = roberta_model_dir
        model = cls(**cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model
