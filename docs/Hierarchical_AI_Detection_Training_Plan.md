# Hierarchical AI-Generated Text Detection Training Plan

## 1. Project Goal

The project should move beyond a plain document-level baseline. The final system should detect whether a text is human-written or AI-generated, output a calibrated confidence score, and localize AI-like evidence at multiple levels:

- Article level: final AI probability for the whole document.
- Paragraph level: AI-likelihood score for each paragraph.
- Phrase/span level: highlighted spans that contribute to the AI decision.

The intended research question is:

> What linguistic and contextual signals distinguish human writing from AI-generated writing, and can these signals be used for both detection and interpretable localization?

## 2. Existing Baselines

The current project already contains two trained baselines.

### 2.1 TF-IDF + Logistic Regression

Path:

`outputs/tfidf/model.joblib`

Method:

- Convert text into word-level unigram/bigram TF-IDF features.
- Train Logistic Regression for binary classification.
- Labels: `0 = human`, `1 = machine`.

Strength:

- Strong test accuracy and macro-F1.
- Interpretable lexical features through TF-IDF coefficients.

Weakness:

- Mainly captures surface lexical and n-gram patterns.
- Word-level highlights can easily over-mark normal human academic writing if used naively.

### 2.2 RoBERTa-base Fine-tuning

Path:

`outputs/roberta_base/best_model`

Method:

- Load pretrained `roberta-base`.
- Tokenize text with byte-level BPE subword tokenizer.
- Truncate input to `max_length = 512`.
- Fine-tune full `RobertaForSequenceClassification`, not a frozen encoder.
- Classification labels: `0 = human`, `1 = machine`.

Strength:

- Learns contextual semantic representations.
- High test AUROC, meaning the ranking signal is strong.

Weakness:

- Default threshold over-predicts AI on the test set.
- Long documents are truncated to 512 subword tokens.
- Hidden semantic features are not directly human-readable.

## 3. Unified Modeling Direction

The next model should not separate feature analysis from classification. Instead, linguistic features should enter the classifier pipeline.

Recommended final direction:

```text
Raw text
  -> TF-IDF lexical signal
  -> RoBERTa contextual signal
  -> Human-readable linguistic features
  -> Fusion classifier
  -> Article / paragraph / phrase-level AI probability and evidence
```

The fusion classifier can use:

- `tfidf_prob_ai` or `tfidf_logit`
- `roberta_prob_ai`, `roberta_logit`, confidence, entropy
- manual linguistic features
- optional paragraph/span aggregation features

Recommended simple fusion models:

- Logistic Regression for interpretability.
- LightGBM or XGBoost if stronger performance is needed.

## 4. Manual Linguistic Features

Manual features should be used as classifier inputs, not only as post-hoc statistics.

Feature groups:

### 4.1 Length and Structure

- character count
- word count
- sentence count
- paragraph count
- average sentence length
- sentence length standard deviation
- max/min sentence length

### 4.2 Lexical Diversity

- type-token ratio
- unique word ratio
- average word length
- rare word ratio
- repeated bigram ratio
- repeated trigram ratio

### 4.3 Style and Discourse

- punctuation density
- comma / semicolon / colon / question mark / exclamation mark density
- first-person pronoun ratio
- second-person pronoun ratio
- contraction ratio
- modal verb ratio
- discourse marker ratio

Example discourse markers:

```text
however, therefore, moreover, furthermore, in conclusion, as a result,
on the other hand, it is important to, it is essential to
```

These features should be standardized and used to train a classifier:

```text
manual_features -> Logistic Regression / LightGBM -> manual_prob_ai
```

Then combine:

```text
[tfidf_logit, roberta_logit, manual_logit, manual_features] -> fusion classifier
```

## 5. Article / Paragraph / Phrase Hierarchy

The final detector should be hierarchical.

### 5.1 Article Level

For the whole document:

```text
article_prob_ai = fusion_classifier(article)
```

Output:

- final AI probability
- final label
- confidence
- branch contributions from TF-IDF / RoBERTa / manual features

### 5.2 Paragraph Level

Split article by blank lines or paragraph boundaries:

```text
article -> paragraph_1, paragraph_2, ..., paragraph_n
```

Score each paragraph:

```text
paragraph_prob_ai_i = fusion_classifier(paragraph_i)
```

Use paragraph scores for a heatmap. This is the most stable localization layer and should be the main demo.

### 5.3 Phrase / Span Level

Phrase-level labels are not directly available in the original SemEval data. Therefore, use weak supervision or span scoring.

Candidate span generation:

- sentence-level spans
- punctuation-separated clauses
- 5-12 word sliding windows
- stride 3-5 words
- optional repeated n-grams or discourse-marker spans

Recommended basic approach:

```text
for suspicious paragraphs:
    generate candidate spans
    score each span with TF-IDF / RoBERTa / fusion classifier
    keep top-k non-overlapping spans
    highlight them in HTML
```

Important gating:

- Only enable phrase-level red highlighting when paragraph probability is high.
- Prefer phrase spans over single words.
- Limit each paragraph to top 3-5 highlighted spans.
- Avoid full-text red coloring.

## 6. Teacher-Suggested Mixed-Text Training

The teacher suggested constructing synthetic mixed documents:

- Insert human-written spans into AI-generated documents.
- Insert AI-generated spans into human-written documents.
- Label the inserted spans during training.
- Evaluate whether the model can locate and classify mixed-origin text.

This is a strong idea because it creates phrase/span-level supervision from document-level data.

## 7. Mixed-Text Data Construction

### 7.1 Source Pools

Use existing SemEval examples:

- Human pool: documents with `label = 0`.
- AI pool: documents with `label = 1`.

Split each document into paragraphs or sentence chunks.

### 7.2 Construct AI Document with Human Insertions

Input:

- Base document: AI-generated.
- Inserted spans: human-written paragraph/sentence/span.

Output labels:

- Document label can remain AI or be marked as mixed.
- Span labels:
  - base AI spans: `1`
  - inserted human spans: `0`

Example:

```text
[AI paragraph 1]        span_label = 1
[Inserted human span]   span_label = 0
[AI paragraph 2]        span_label = 1
```

### 7.3 Construct Human Document with AI Insertions

Input:

- Base document: human-written.
- Inserted spans: AI-generated paragraph/sentence/span.

Output labels:

```text
[Human paragraph 1]     span_label = 0
[Inserted AI span]      span_label = 1
[Human paragraph 2]     span_label = 0
```

### 7.4 Mixing Granularity

Use multiple granularities:

- paragraph insertion
- sentence insertion
- clause or phrase insertion

Recommended first version:

- paragraph insertion and sentence insertion
- avoid very short phrase insertion at first, because labels become noisy

### 7.5 Mixing Ratio

For each mixed document:

- replace or insert 10-40% of spans
- sample insertion position randomly
- preserve document length distribution where possible

Metadata to save:

```text
document_id
mixed_text
document_label
is_mixed
base_origin
inserted_origin
span_start_char
span_end_char
span_text
span_label
span_granularity
source_model
source_domain
```

## 8. Training with Mixed-Text Labels

There are two levels of supervision.

### 8.1 Document-Level Training

Use the final document label:

- `0 = mostly human`
- `1 = mostly AI`
- optional `2 = mixed` if using a 3-class setup

For simplicity, keep binary classification first:

```text
document_label = 1 if AI proportion >= 0.5 else 0
```

### 8.2 Span-Level Training

Each generated span has a known label:

- `0 = human span`
- `1 = AI span`

Train a span classifier:

```text
span_text -> classifier -> span_prob_ai
```

The span classifier can use:

- TF-IDF span features
- RoBERTa span representation
- manual span-level features
- fusion features

This directly supports phrase/paragraph highlighting.

## 9. MIL Alternative

If we want weak supervision without explicit synthetic span labels, formulate the task as Multiple Instance Learning.

```text
article = bag
spans = instances
article label = bag label
span labels = latent
```

Model:

```text
span_j -> encoder -> span_logit_j
span_logits -> aggregator -> article_logit
article_logit -> BCE loss with article label
```

Aggregation options:

- mean pooling
- max pooling
- top-k mean
- noisy-OR
- attention pooling

Recommended first MIL-style aggregation:

```text
article_logit = mean(top-k span_logits)
```

This gives span scores and document scores in one model.

However, true MIL training is more complex. The mixed-text construction above is easier and gives explicit span labels.

## 10. Token / Phrase Heatmap Strategy

Token-level heatmaps should be conservative. TF-IDF and manual features can over-highlight normal human academic writing if every positive word is colored.

Recommended highlighting rules:

- Highlight spans, not isolated common words.
- Only highlight top-k evidence spans.
- Require paragraph-level probability above a threshold before phrase highlighting.
- Use multi-source agreement if possible.
- Show highlights as model evidence, not proof of AI authorship.

### 10.1 TF-IDF Evidence

For each n-gram:

```text
ngram_contribution = tfidf_value * logistic_regression_coefficient
```

Use top positive n-grams as AI-like phrase evidence.

### 10.2 RoBERTa Attention Evidence

Use attention only as qualitative visualization.

Implementation:

```text
outputs = model(..., output_attentions=True)
attention = average attention from <s> token to other tokens
```

Recommended:

- average last 4 layers
- average all heads
- map subword tokens back to readable words

Do not claim attention is causal proof.

### 10.3 Manual Feature Attribution

Manual features do not have transformer-style attention. Use feature-to-span attribution:

- discourse marker feature -> marker spans
- modal verb feature -> modal words
- repeated n-gram feature -> repeated spans
- punctuation density -> punctuation spans
- sentence length features -> sentence-level highlighting

This should be described as:

> linguistic feature attribution, not attention.

## 11. Recommended Development Stages

### Stage 1: Fusion Classifier

Implement:

- extract manual linguistic features
- read TF-IDF predictions
- read RoBERTa predictions
- train fusion classifier on dev or a train split
- evaluate on test

Outputs:

- fusion predictions
- fusion metrics
- feature coefficients / importance

### Stage 2: Mixed-Text Dataset Generator

Implement:

`scripts/build_mixed_text_dataset.py`

Inputs:

- human examples
- AI examples
- granularity: paragraph or sentence
- insertion ratio

Outputs:

- mixed document CSV/JSONL
- span label CSV/JSONL

### Stage 3: Span Classifier

Implement:

`scripts/train_span_classifier.py`

Train on synthetic span labels:

```text
span_text -> label 0/1
```

Evaluate:

- span accuracy
- span macro-F1
- document accuracy after span aggregation

### Stage 4: Hierarchical Inference

Implement:

`scripts/hierarchical_predict.py`

Outputs:

- article score
- paragraph scores
- phrase scores
- top highlighted spans

### Stage 5: HTML Demo

Implement:

`scripts/hierarchical_heatmap.py`

Output:

- article-level probability
- paragraph-level background colors
- phrase-level highlights
- top linguistic features
- branch contribution bar

## 12. Evaluation Plan

### 12.1 Standard Test Evaluation

Use SemEval test labels:

- accuracy
- macro-F1
- AUROC
- ECE
- Brier score
- NLL

### 12.2 Mixed-Text Evaluation

Because mixed-text data has span labels, evaluate:

- document accuracy
- paragraph accuracy
- span accuracy
- span macro-F1
- exact span localization if using character offsets
- top-k span recall

Suggested metric:

```text
Top-k span recall:
Does the true inserted AI/human span appear in the top-k highlighted spans?
```

### 12.3 Ablation Table

Report:

| Model | Article Acc | Article Macro-F1 | Span F1 | ECE | Notes |
|---|---:|---:|---:|---:|---|
| TF-IDF LR | existing | existing | - | existing | lexical baseline |
| RoBERTa-base | existing | existing | - | existing | contextual baseline |
| Manual features | new | new | optional | new | interpretable features |
| TF-IDF + RoBERTa | new | new | - | new | model fusion |
| TF-IDF + RoBERTa + Manual | new | new | optional | new | full fusion |
| Mixed-span classifier | new | new | new | new | hierarchical model |

## 13. Final Report Narrative

The final project should be framed as:

> A hierarchical AI-generated text detector that combines lexical, contextual, and interpretable linguistic signals. It provides article-level confidence, paragraph-level localization, and phrase-level evidence, trained and evaluated using both standard SemEval labels and synthetic mixed-origin documents with span labels.

This narrative connects:

- classification performance
- confidence calibration
- linguistic feature analysis
- phrase/span localization
- teacher-suggested mixed AI-human training

## 14. Immediate Next Steps

1. Build the manual feature extractor and fusion classifier.
2. Build the mixed-text dataset generator.
3. Train a first span classifier on synthetic paragraph/sentence labels.
4. Evaluate document accuracy and span accuracy.
5. Generate hierarchical HTML heatmap for demo.
6. Add ablation results to the final report and slides.

