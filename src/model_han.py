"""
src/model_han.py

Hierarchical Attention Network over DeBERTa-v3-large.
Architecture:
  1. DeBERTa-v3-large encodes each sentence independently (CLS vector)
  2. 4-layer CrossSentence TransformerEncoder adds inter-sentence context
  3. Sentence-level attention → sentence AI scores
  4. Paragraph-level grouping + attention → paragraph AI scores
  5. Weighted pooling → document vector → classification head
  6. 3 auxiliary heads for MTL (discourse / contraction / first_person buckets)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import DebertaV2Model, AutoConfig


class HierarchicalDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "models/deberta-v3-large",
        max_sent_len: int = 96,
        max_sents: int = 128,
        cross_sent_layers: int = 4,
        cross_sent_heads: int = 8,
        dropout: float = 0.1,
        num_aux_classes: int = 3,
    ):
        super().__init__()
        self.max_sent_len = max_sent_len
        self.max_sents = max_sents

        # ── Sentence encoder: DeBERTa-v3-large ────────────────────────────────
        self.sentence_encoder = DebertaV2Model.from_pretrained(model_name)
        hidden_size = self.sentence_encoder.config.hidden_size  # 1024

        # ── Cross-sentence Transformer ─────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=cross_sent_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.cross_sent_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cross_sent_layers
        )

        # ── Attention heads ────────────────────────────────────────────────────
        self.sent_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.para_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # ── Classification head ────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.doc_classifier = nn.Linear(hidden_size, 2)

        # ── MTL auxiliary heads ────────────────────────────────────────────────
        # One head per feature bucket: discourse / contraction / first_person
        self.aux_heads = nn.ModuleDict({
            "discourse": nn.Linear(hidden_size, num_aux_classes),
            "contraction": nn.Linear(hidden_size, num_aux_classes),
            "first_person": nn.Linear(hidden_size, num_aux_classes),
        })

        # ── Positional encoding for cross-sent transformer ─────────────────────
        self.pos_embedding = nn.Embedding(max_sents + 1, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        self._init_cross_sent_weights()

    def _init_cross_sent_weights(self):
        """Initialize cross-sentence transformer with small weights for stability."""
        for module in self.cross_sent_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_sentences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a batch of sentences through DeBERTa.
        Args:
            input_ids: (num_sents, seq_len)
            attention_mask: (num_sents, seq_len)
        Returns:
            sent_reprs: (num_sents, hidden_size) — CLS vectors
        """
        # Process in mini-batches to avoid OOM on long documents
        chunk_size = 32
        all_cls = []
        n = input_ids.size(0)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end]
            kwargs = {"input_ids": ids_chunk, "attention_mask": mask_chunk}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids[start:end]
            out = self.sentence_encoder(**kwargs)
            cls = out.last_hidden_state[:, 0, :]  # (chunk, hidden)
            all_cls.append(cls)
        return torch.cat(all_cls, dim=0)  # (num_sents, hidden)

    def forward(
        self,
        sent_input_ids: torch.Tensor,
        sent_attention_mask: torch.Tensor,
        para_boundaries: list[tuple[int, int]],
        sent_token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sent_input_ids: (num_sents, max_sent_len)
            sent_attention_mask: (num_sents, max_sent_len)
            para_boundaries: list of (start_idx, end_idx) for each paragraph
            sent_token_type_ids: optional (num_sents, max_sent_len)
        Returns:
            dict with keys:
                doc_logits: (2,)
                sent_attention_weights: (num_sents,)
                para_attention_weights: (num_paras,)
                aux_logits: dict[str, (num_aux_classes,)]
        """
        num_sents = sent_input_ids.size(0)

        # 1. Encode each sentence
        sent_reprs = self.encode_sentences(
            sent_input_ids, sent_attention_mask, sent_token_type_ids
        )  # (num_sents, hidden)

        # 2. Add positional embeddings
        positions = torch.arange(num_sents, device=sent_reprs.device)
        positions = positions.clamp(max=self.max_sents)
        sent_reprs = sent_reprs + self.pos_embedding(positions)
        sent_reprs = self.input_ln(sent_reprs)  # stabilize before cross-sent encoder

        # 3. Cross-sentence context (batch_first=True, add batch dim)
        ctx_reprs = self.cross_sent_encoder(
            sent_reprs.unsqueeze(0)
        ).squeeze(0)  # (num_sents, hidden)

        # 4. Sentence-level attention weights
        sent_scores = self.sent_attention(ctx_reprs)  # (num_sents, 1)
        sent_weights = torch.softmax(sent_scores.squeeze(-1), dim=0)  # (num_sents,)

        # 5. Paragraph-level representations
        para_reprs_list: list[torch.Tensor] = []
        for start, end in para_boundaries:
            para_ctx = ctx_reprs[start:end]  # (para_len, hidden)
            para_sent_w = sent_weights[start:end].unsqueeze(-1)  # (para_len, 1)
            para_repr = (para_ctx * para_sent_w).sum(0)  # (hidden,)
            para_reprs_list.append(para_repr)

        if para_reprs_list:
            para_reprs = torch.stack(para_reprs_list)  # (num_paras, hidden)
            para_scores = self.para_attention(para_reprs)  # (num_paras, 1)
            para_weights = torch.softmax(para_scores.squeeze(-1), dim=0)  # (num_paras,)
            doc_repr = (para_reprs * para_weights.unsqueeze(-1)).sum(0)  # (hidden,)
        else:
            # Fallback: direct sentence pooling
            doc_repr = (ctx_reprs * sent_weights.unsqueeze(-1)).sum(0)
            para_weights = torch.ones(1, device=sent_reprs.device)

        # 6. Classification
        doc_repr = self.dropout(doc_repr)
        doc_logits = self.doc_classifier(doc_repr)  # (2,)

        # 7. MTL auxiliary heads
        aux_logits = {k: head(doc_repr) for k, head in self.aux_heads.items()}

        return {
            "doc_logits": doc_logits,
            "sent_attention_weights": sent_weights,
            "para_attention_weights": para_weights,
            "aux_logits": aux_logits,
            "doc_repr": doc_repr,
            "ctx_reprs": ctx_reprs,
        }

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs) -> "HierarchicalDetector":
        """Load a saved HierarchicalDetector from a checkpoint directory."""
        import os
        checkpoint = torch.load(
            os.path.join(model_dir, "han_checkpoint.pt"),
            map_location="cpu",
        )
        config = checkpoint["config"]
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, model_dir: str) -> None:
        """Save model and config to a directory."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "model_name": "/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline/models/deberta-v3-large",
                    "max_sent_len": self.max_sent_len,
                    "max_sents": self.max_sents,
                },
            },
            os.path.join(model_dir, "han_checkpoint.pt"),
        )
        print(f"Saved model to {model_dir}/han_checkpoint.pt")
