from __future__ import annotations

import re
import string
from collections import Counter

import pandas as pd


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)
PARAGRAPH_RE = re.compile(r"\n\s*\n+")

_CONTRACTIONS = re.compile(r"n't|'re|'ve|'ll|'d|'m|'s\b", re.IGNORECASE)
_FIRST_PERSON = frozenset(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"])
_SECOND_PERSON = frozenset(["you", "your", "yours", "yourself", "yourselves"])
_MODALS = frozenset(["can", "could", "may", "might", "must", "shall", "should", "will", "would"])
_DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "nevertheless", "nonetheless",
    "consequently", "additionally", "subsequently", "alternatively",
    "in conclusion", "in summary", "as a result", "on the other hand",
    "it is important", "it is essential", "it is worth noting",
    "first of all", "in addition", "in contrast", "for instance", "for example",
]


def text_features(text: str) -> dict[str, float | int]:
    text = "" if pd.isna(text) else str(text)
    text_lower = text.lower()
    tokens = TOKEN_RE.findall(text_lower)
    sentences = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    token_count = len(tokens)
    sentence_count = max(len(sentences), 1)
    char_count = len(text)

    # sentence length stats
    sent_lengths = [len(TOKEN_RE.findall(s)) for s in sentences]
    sl_mean = sum(sent_lengths) / sentence_count
    sl_var = sum((l - sl_mean) ** 2 for l in sent_lengths) / sentence_count
    sl_std = sl_var ** 0.5

    # paragraph count
    paragraphs = [p.strip() for p in PARAGRAPH_RE.split(text) if p.strip()]
    para_count = max(len(paragraphs), 1)

    # pronoun ratios
    first_person_count = sum(1 for t in tokens if t in _FIRST_PERSON)
    second_person_count = sum(1 for t in tokens if t in _SECOND_PERSON)

    # contraction ratio (based on raw text matches, not tokens)
    contraction_count = len(_CONTRACTIONS.findall(text))

    # modal verb ratio
    modal_count = sum(1 for t in tokens if t in _MODALS)

    # discourse marker ratio (count occurrences in lower text)
    discourse_count = sum(text_lower.count(marker) for marker in _DISCOURSE_MARKERS)

    return {
        "text_length": char_count,
        "word_count": token_count,
        "sentence_count": sentence_count,
        "paragraph_count": para_count,
        "avg_sentence_length": sl_mean,
        "sentence_length_std": sl_std,
        "ttr": len(set(tokens)) / token_count if token_count else 0.0,
        "punctuation_per_token": _punctuation_count(text) / token_count if token_count else 0.0,
        "repeated_bigram_ratio": repeated_ngram_ratio(tokens, 2),
        "repeated_trigram_ratio": repeated_ngram_ratio(tokens, 3),
        "avg_word_length": sum(len(tok) for tok in tokens) / token_count if token_count else 0.0,
        "first_person_ratio": first_person_count / token_count if token_count else 0.0,
        "second_person_ratio": second_person_count / token_count if token_count else 0.0,
        "contraction_ratio": contraction_count / token_count if token_count else 0.0,
        "modal_verb_ratio": modal_count / token_count if token_count else 0.0,
        "discourse_marker_ratio": discourse_count / sentence_count,
    }


def add_text_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Missing required text column: {text_col}")
    feature_df = pd.DataFrame([text_features(text) for text in df[text_col]])
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)


def repeated_ngram_ratio(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def high_confidence_errors(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    min_confidence: float = 0.9,
) -> pd.DataFrame:
    if "confidence" not in df.columns:
        return df.iloc[0:0].copy()
    truth = pd.to_numeric(df[true_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    return df[(truth.notna()) & (pred.notna()) & (truth != pred) & (conf >= min_confidence)].copy()


def _punctuation_count(text: str) -> int:
    punct = set(string.punctuation)
    return sum(1 for char in text if char in punct)
