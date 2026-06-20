"""
src/features_v2.py
5-layer linguistic feature extractor for AI vs human text analysis.
Layers: lexical, syntactic, discourse, pragmatic, rhetorical.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Optional

import numpy as np

# ── regex helpers ──────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_CONTRACTION_RE = re.compile(
    r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
    r"isn't|aren't|wasn't|weren't|I'm|I've|I'll|I'd|"
    r"it's|that's|there's|they're|we're|you're|he's|she's|"
    r"could've|would've|should've|I'd|they'd|we'd)\b",
    re.IGNORECASE,
)
_LATINATE_RE = re.compile(
    r"(tion|ment|ance|ence|ity|ous|ive|ize|ise|ism|ist|ate|ify)$",
    re.IGNORECASE,
)

# ── word sets ──────────────────────────────────────────────────────────────────
_FIRST_PERSON = frozenset([
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
])
_MODALS = frozenset([
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would",
])
_HEDGES = [
    "it seems", "it appears", "it is suggested", "research indicates",
    "studies show", "it is worth noting", "it should be noted",
    "one might argue", "it could be argued", "arguably",
    "to some extent", "in many cases", "generally speaking",
    "it is generally", "it has been shown", "evidence suggests",
]
_DISCOURSE_MARKERS = {
    "additive": [
        "furthermore", "moreover", "additionally", "in addition",
        "besides", "likewise", "also",
    ],
    "contrastive": [
        "however", "nevertheless", "on the other hand",
        "in contrast", "although", "despite", "nonetheless",
        "yet", "conversely",
    ],
    "causal": [
        "therefore", "thus", "consequently", "as a result",
        "because of this", "hence", "accordingly",
    ],
    "conclusive": [
        "in conclusion", "to summarize", "in summary",
        "overall", "in short", "to conclude", "to sum up",
        "in closing",
    ],
}
_ALL_MARKERS = [m for ms in _DISCOURSE_MARKERS.values() for m in ms]


# ── text splitting helpers ─────────────────────────────────────────────────────

def split_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    parts = _PARAGRAPH_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Layer 1: Lexical ───────────────────────────────────────────────────────────

def lexical_features(tokens: list[str]) -> dict[str, float]:
    n = len(tokens)
    if n == 0:
        return {
            "mattr": 0.0, "hapax_ratio": 0.0, "latinate_ratio": 0.0,
            "avg_word_length": 0.0,
        }

    # MATTR: Moving Average TTR over window=50
    window = min(50, n)
    if n <= window:
        mattr_val = len(set(tokens)) / n
    else:
        ttrs = []
        for i in range(n - window + 1):
            w = tokens[i : i + window]
            ttrs.append(len(set(w)) / window)
        mattr_val = float(np.mean(ttrs))

    # Hapax ratio: words appearing exactly once / total unique words
    counts = Counter(tokens)
    unique = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = hapax / unique if unique > 0 else 0.0

    # Latinate ratio
    latinate = sum(1 for t in tokens if _LATINATE_RE.search(t))
    latinate_ratio = latinate / n

    # Average word length
    avg_word_len = sum(len(t) for t in tokens) / n

    return {
        "mattr": round(mattr_val, 6),
        "hapax_ratio": round(hapax_ratio, 6),
        "latinate_ratio": round(latinate_ratio, 6),
        "avg_word_length": round(avg_word_len, 6),
    }


# ── Layer 2: Syntactic (requires spaCy) ───────────────────────────────────────

def syntactic_features(text: str) -> dict[str, float]:
    """Requires spaCy en_core_web_sm loaded globally via _get_nlp()."""
    try:
        nlp = _get_nlp()
        doc = nlp(text)
    except Exception:
        # Fallback: return neutral values if spaCy unavailable
        return {
            "passive_ratio": 0.0,
            "subordination_ratio": 0.0,
            "pos_bigram_entropy": 0.0,
        }

    verbs = [t for t in doc if t.pos_ == "VERB"]
    passive = [t for t in doc if t.dep_ == "auxpass"]
    passive_ratio = len(passive) / len(verbs) if verbs else 0.0

    sents = list(doc.sents)
    clause_deps = {"advcl", "relcl", "ccomp", "acl"}
    clause_count = sum(1 for t in doc if t.dep_ in clause_deps)
    subordination_ratio = clause_count / len(sents) if sents else 0.0

    pos_tags = [t.pos_ for t in doc if not t.is_space]
    bigrams = list(zip(pos_tags, pos_tags[1:]))
    if bigrams:
        bg_counts = Counter(bigrams)
        total = sum(bg_counts.values())
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in bg_counts.values()
        )
    else:
        entropy = 0.0

    return {
        "passive_ratio": round(passive_ratio, 6),
        "subordination_ratio": round(subordination_ratio, 6),
        "pos_bigram_entropy": round(entropy, 6),
    }


_nlp_cache: Optional[object] = None


def _get_nlp():
    global _nlp_cache
    if _nlp_cache is None:
        import spacy
        _nlp_cache = spacy.load("en_core_web_sm")
    return _nlp_cache


# ── Layer 3: Discourse ─────────────────────────────────────────────────────────

def discourse_features(
    text: str,
    sentences: list[str],
    paragraphs: list[str],
) -> dict[str, float]:
    text_lower = text.lower()
    n_sents = max(len(sentences), 1)

    # Per-category density
    by_cat: dict[str, float] = {}
    for cat, markers in _DISCOURSE_MARKERS.items():
        count = sum(
            1 for i, s in enumerate(sentences)
            if any(m in s.lower() for m in markers)
        )
        by_cat[f"discourse_{cat}"] = count / n_sents

    # Total discourse marker density
    marker_positions: list[float] = []
    for i, s in enumerate(sentences):
        if any(m in s.lower() for m in _ALL_MARKERS):
            marker_positions.append(i / n_sents)
    total_density = len(marker_positions) / n_sents

    # Position std: low value = markers clustered at start/end (AI signal)
    pos_std = float(np.std(marker_positions)) if len(marker_positions) > 1 else 0.0

    # Structural completeness: fraction of paragraphs with topic sentence + transition
    completeness_scores: list[float] = []
    for para in paragraphs:
        para_lower = para.lower()
        para_sents = split_sentences(para)
        if len(para_sents) < 2:
            completeness_scores.append(0.5)
            continue
        # Topic sentence heuristic: starts with the/this/these/ai/it + verb
        first_lower = para_sents[0].lower()
        has_topic = any(
            first_lower.startswith(w)
            for w in ["the ", "this ", "these ", "it ", "ai ", "one ", "another "]
        )
        # Transition heuristic: last sentence contains a discourse marker
        last_lower = para_sents[-1].lower()
        has_transition = any(m in last_lower for m in _ALL_MARKERS)
        completeness_scores.append(0.5 * int(has_topic) + 0.5 * int(has_transition))

    structural_completeness = (
        float(np.mean(completeness_scores)) if completeness_scores else 0.0
    )

    result = {
        "discourse_total_density": round(total_density, 6),
        "discourse_position_std": round(pos_std, 6),
        "structural_completeness": round(structural_completeness, 6),
    }
    for k, v in by_cat.items():
        result[k] = round(v, 6)
    return result


# ── Layer 4: Pragmatic ─────────────────────────────────────────────────────────

def pragmatic_features(
    text: str,
    sentences: list[str],
    tokens: list[str],
) -> dict[str, float]:
    n_tokens = max(len(tokens), 1)
    n_sents = max(len(sentences), 1)

    # Contraction ratio (raw text match, case-insensitive)
    contractions = _CONTRACTION_RE.findall(text)
    contraction_ratio = len(contractions) / n_tokens

    # First-person ratio
    first_person_count = sum(1 for t in tokens if t in _FIRST_PERSON)
    first_person_ratio = first_person_count / n_tokens

    # Hedge density + position distribution
    hedge_positions: list[float] = []
    for i, s in enumerate(sentences):
        s_lower = s.lower()
        if any(h in s_lower for h in _HEDGES):
            hedge_positions.append(i / n_sents)
    hedge_density = len(hedge_positions) / n_sents
    # Low std = hedges clustered at boundaries = AI signal
    hedge_pos_std = (
        float(np.std(hedge_positions)) if len(hedge_positions) > 1 else 0.0
    )

    # Modal verb ratio
    modal_count = sum(1 for t in tokens if t in _MODALS)
    modal_ratio = modal_count / n_tokens

    return {
        "contraction_ratio": round(contraction_ratio, 6),
        "first_person_ratio": round(first_person_ratio, 6),
        "hedge_density": round(hedge_density, 6),
        "hedge_position_std": round(hedge_pos_std, 6),
        "modal_ratio": round(modal_ratio, 6),
    }


# ── Layer 5: Rhetorical ────────────────────────────────────────────────────────

def rhetorical_features(
    sentences: list[str],
    paragraphs: list[str],
) -> dict[str, float]:
    # Sentence length CV (coefficient of variation): low = AI (uniform length)
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    if sent_lengths:
        mean_len = float(np.mean(sent_lengths))
        std_len = float(np.std(sent_lengths))
        sent_length_cv = std_len / mean_len if mean_len > 0 else 0.0
        sent_length_std = std_len
    else:
        sent_length_cv = 0.0
        sent_length_std = 0.0

    # Paragraph length std (in sentences): low = AI (uniform para length)
    para_lengths: list[int] = []
    for para in paragraphs:
        para_sents = split_sentences(para)
        para_lengths.append(max(len(para_sents), 1))
    para_length_std = float(np.std(para_lengths)) if len(para_lengths) > 1 else 0.0

    # Rhetorical question density
    rhet_q_markers = [
        "isn't it", "don't you", "wouldn't", "aren't we",
        "can we not", "could we not", "isn't that",
    ]
    rhet_q_count = sum(
        1 for s in sentences
        if s.strip().endswith("?")
        and any(m in s.lower() for m in rhet_q_markers)
    )
    rhet_q_ratio = rhet_q_count / max(len(sentences), 1)

    return {
        "sentence_length_std": round(sent_length_std, 6),
        "sentence_length_cv": round(sent_length_cv, 6),
        "para_length_std": round(para_length_std, 6),
        "rhetorical_question_ratio": round(rhet_q_ratio, 6),
    }


# ── Unified entry point ────────────────────────────────────────────────────────

def full_features(text: str) -> dict[str, float]:
    """Compute all 5-layer features for a given text string."""
    text = text or ""
    tokens = split_tokens(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text]

    result: dict[str, float] = {}
    result.update(lexical_features(tokens))
    result.update(syntactic_features(text))
    result.update(discourse_features(text, sentences, paragraphs))
    result.update(pragmatic_features(text, sentences, tokens))
    result.update(rhetorical_features(sentences, paragraphs))
    return result


def feature_names() -> list[str]:
    """Return the canonical list of all feature names."""
    return list(full_features("This is a test sentence. Another sentence here.").keys())
