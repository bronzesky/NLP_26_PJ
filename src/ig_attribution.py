"""
src/ig_attribution.py

Token-level attribution via Integrated Gradients for HSAD.
Uses captum.attr.IntegratedGradients on the document-level classifier.
Only called at demo/inference time — not during training.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


def compute_token_attribution(
    model: "HSAD",
    tokenizer,
    text: str,
    sent_spans: list[tuple[int, int]],
    target_class: int = 1,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> list[tuple[str, float]]:
    """
    Compute Integrated Gradients attribution for each token.

    Returns:
        List of (token_str, attribution_score) — score > 0 means AI-indicative,
        score < 0 means human-indicative.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise RuntimeError("captum is required: pip install captum")

    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].squeeze(0).to(device)
    attention_mask = enc["attention_mask"].squeeze(0).to(device)

    # Wrap forward to accept embedding input for IG
    embeddings = model.encoder.embeddings.word_embeddings(input_ids)  # (seq_len, 768)

    def forward_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
        # embeds: (1, seq_len, 768) — IG adds batch dim
        embeds = embeds.squeeze(0)
        # Run RoBERTa with custom embeddings
        # We need to hook into the embedding layer
        outputs = model.encoder(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        token_hidden = outputs.last_hidden_state.squeeze(0)
        # Sentence pooling
        sent_vecs = []
        for start, end in sent_spans:
            if end > start:
                sent_vecs.append(token_hidden[start:end].mean(0))
            else:
                sent_vecs.append(token_hidden[max(0, start):start+1].mean(0))
        if not sent_vecs:
            sent_vecs = [token_hidden[0]]
        sent_repr = torch.stack(sent_vecs)
        n_sents = len(sent_vecs)
        positions = torch.arange(n_sents, device=device).clamp(max=128)
        sent_repr = sent_repr + model.sent_pos_emb(positions)
        ctx = model.cross_sent(sent_repr.unsqueeze(0)).squeeze(0)
        attn_w = torch.softmax(model.sent_attn(ctx).squeeze(-1), dim=0)
        doc_vec = (ctx * attn_w.unsqueeze(-1)).sum(0)
        doc_logits = model.doc_head(model.dropout(doc_vec))
        return doc_logits[target_class].unsqueeze(0)

    ig = IntegratedGradients(forward_from_embeds)
    baseline = torch.zeros_like(embeddings)  # zero embedding baseline

    # IG expects (batch, seq_len, hidden) — add batch dim
    attributions, _ = ig.attribute(
        embeddings.unsqueeze(0),
        baseline.unsqueeze(0),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # attribution per token: L2 norm over embedding dim
    token_attr = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()

    # Normalize to [-1, 1]
    max_abs = max(abs(token_attr).max(), 1e-8)
    token_attr = token_attr / max_abs

    # Map back to token strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())
    result = [(tok, float(score)) for tok, score in zip(tokens, token_attr)]
    return result


def attribution_to_html_spans(token_attributions: list[tuple[str, float]]) -> str:
    """Convert token attribution scores to inline HTML with background colors."""
    import html as _html
    parts = []
    for tok, score in token_attributions:
        # Skip special tokens
        if tok in ("<s>", "</s>", "<pad>"):
            continue
        # Clean up RoBERTa tokenization artifacts
        display = tok.replace("Ġ", " ").replace("Ċ", "\n")
        if score > 0.15:
            intensity = min(int(score * 200), 200)
            color = f"rgba(255,80,80,{score:.2f})"
        elif score < -0.15:
            intensity = min(int(abs(score) * 200), 200)
            color = f"rgba(80,160,255,{abs(score):.2f})"
        else:
            color = "transparent"
        escaped = _html.escape(display)
        if color != "transparent":
            parts.append(f'<span style="background:{color};border-radius:2px">{escaped}</span>')
        else:
            parts.append(escaped)
    return "".join(parts)
