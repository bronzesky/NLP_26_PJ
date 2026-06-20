# Design Discussion Notes

This document records the main design decisions and reasoning from project discussions. It is intended as future reference for implementation, slides, and the final report.

## 1. What Has Been Done So Far

The project currently implements a SemEval 2024 Task 8 Subtask A English AI-generated text detector.

Existing outputs:

- TF-IDF + Logistic Regression baseline.
- RoBERTa-base fine-tuning baseline.
- Official test prediction pipeline.
- Model calibration for RoBERTa.
- Grouped error analysis by model/domain/length/confidence.
- Paragraph-level AI-likelihood heatmap demo.
- Experiment summary and model comparison summary.

Project path on CFFF:

`/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline`

Key outputs:

- `outputs/tfidf/model.joblib`
- `outputs/roberta_base/best_model`
- `outputs/tfidf_test/metrics.json`
- `outputs/roberta_base_test/metrics.json`
- `outputs/analysis/group_by_model.csv`
- `outputs/analysis/group_by_length_bucket.csv`
- `outputs/roberta_base/calibration/reliability.html`
- `outputs/paragraph_demo_tfidf/paragraph_heatmap.html`

## 2. Current Baselines

### 2.1 TF-IDF + Logistic Regression

This is a traditional lexical baseline.

Training idea:

```text
text -> TF-IDF unigram/bigram features -> Logistic Regression -> human / machine
```

It learns statistical lexical and phrase patterns. Each n-gram has a coefficient, so it is more interpretable than neural models.

Strength:

- Fast.
- Strong test accuracy.
- Easy to inspect lexical evidence.

Limitation:

- Mainly surface-level.
- Can over-highlight normal human words or academic phrases if used directly for token heatmaps.

### 2.2 RoBERTa-base Fine-tuning

This is a pretrained Transformer baseline.

Training idea:

```text
raw text
 -> RoBERTa tokenizer
 -> subword token ids
 -> embedding layer
 -> 12-layer Transformer encoder
 -> final contextual representation
 -> classification head
 -> human / machine logits
```

Important clarification:

- RoBERTa is not mostly a tokenizer.
- The tokenizer only converts raw text into subword token ids.
- The main model is the Transformer encoder.
- We classify using the contextual representation produced by the encoder, not static word embeddings.

The implementation uses full fine-tuning:

```text
classification loss
 -> classification head
 -> RoBERTa encoder
 -> embeddings
```

No encoder-freezing code exists in the project. Therefore, the RoBERTa encoder parameters are updated during training.

Key settings:

- `model_name = roberta-base`
- `max_length = 512`
- `num_train_epochs = 3`
- `learning_rate = 2e-5`
- `weight_decay = 0.01`
- `per_device_train_batch_size = 8`
- `gradient_accumulation_steps = 2`
- `fp16 = True`
- `seed = 42`

## 3. RoBERTa Pretraining Background

RoBERTa was pretrained before our project. We do not run this pretraining.

Pretraining objective:

```text
Masked Language Modeling
```

The model sees large-scale unlabeled English text with some tokens masked and learns to predict the original tokens from context.

Compared with BERT, RoBERTa:

- removes Next Sentence Prediction
- uses dynamic masking
- trains longer
- uses larger batches
- uses more data
- uses byte-level BPE

Our project only fine-tunes this pretrained RoBERTa for AI-vs-human classification.

## 4. Why Long Text Still Matters for RoBERTa

RoBERTa uses subword tokenization, but it still has a maximum input length.

The project code uses:

```python
tokenizer(text, truncation=True, max_length=512)
```

Therefore:

- texts longer than 512 subword tokens are truncated
- later parts of long articles are not seen by RoBERTa
- long human texts may lose important evidence

Observed issue:

- RoBERTa performs poorly on long-text buckets.
- `801+` length bucket has much lower accuracy.

Potential fix:

```text
long article
 -> split into chunks or paragraphs
 -> score each chunk
 -> aggregate chunk scores
```

Aggregation candidates:

- mean probability
- max probability
- top-k mean
- majority vote
- logit average

## 5. Need for a Unified Feature-and-Classifier Mainline

Feature analysis should not be separated from classification. The desired final project should use linguistic features inside the classifier and then explain their contribution.

Recommended model:

```text
Raw text
  -> TF-IDF lexical signal
  -> RoBERTa contextual signal
  -> manual linguistic features
  -> fusion classifier
  -> final AI probability
```

Possible feature vector:

```text
[
  tfidf_prob_ai,
  roberta_prob_ai,
  roberta_confidence,
  roberta_entropy,
  word_count,
  sentence_count,
  avg_sentence_length,
  sentence_length_std,
  paragraph_count,
  type_token_ratio,
  unique_word_ratio,
  repeated_bigram_ratio,
  repeated_trigram_ratio,
  punctuation_density,
  comma_density,
  question_mark_density,
  exclamation_density,
  first_person_pronoun_ratio,
  second_person_pronoun_ratio,
  contraction_ratio,
  modal_verb_ratio,
  discourse_marker_ratio
]
```

Recommended fusion classifiers:

- Logistic Regression for interpretability.
- LightGBM / XGBoost for stronger performance if time allows.

## 6. Manual Features

Manual features should represent human-readable writing traits:

### Length and structure

- character count
- word count
- paragraph count
- sentence count
- average sentence length
- sentence length variance

### Lexical diversity

- type-token ratio
- unique word ratio
- average word length
- repeated bigram/trigram ratio

### Style

- punctuation density
- comma / colon / semicolon ratio
- question / exclamation mark ratio
- contraction ratio
- pronoun ratio
- modal verb ratio
- discourse marker ratio

Manual features are useful because they can support the final research goal:

> identify human-readable linguistic differences between AI and human writing

## 7. Manual Features and Attention

Manual features do not naturally have Transformer-style attention.

However, they can support:

```text
feature-to-span attribution
```

Examples:

- discourse marker feature -> highlight "however", "therefore", "in conclusion"
- modal verb feature -> highlight "may", "might", "could", "should"
- contraction feature -> highlight "don't", "I'm", "can't"
- repeated n-gram feature -> highlight repeated phrases
- punctuation density -> highlight punctuation spans
- sentence length features -> sentence-level highlighting

This should be described as:

```text
linguistic feature attribution
```

not as true neural attention.

## 8. Token and Phrase Heatmap Discussion

The final interface should ideally resemble plagiarism-checking systems:

- article-level probability
- paragraph-level heatmap
- phrase-level or token-level highlights

### 8.1 Paragraph Heatmap

Already implemented.

Method:

```text
article -> paragraphs -> score each paragraph -> color paragraph background
```

This is stable and should remain the main visualization layer.

### 8.2 TF-IDF Token/Phrase Evidence

For each n-gram:

```text
contribution = tfidf_value * logistic_regression_coefficient
```

High positive contribution means AI-like evidence.

Warning:

- Direct word-level highlighting can over-mark normal human academic text.
- Prefer phrase-level n-grams.
- Highlight only top-k strong evidence.

### 8.3 RoBERTa Attention Heatmap

RoBERTa can output attention:

```python
outputs = model(..., output_attentions=True)
attentions = outputs.attentions
```

For a simple token heatmap:

```text
take attention from <s> token to all tokens
average heads
optionally average last 4 layers
```

Important limitation:

```text
attention is qualitative visualization, not causal proof
```

### 8.4 Occlusion Heatmap

Occlusion is more faithful but slower:

```text
original_prob = model(text)
for each word/span:
    remove or mask it
    new_prob = model(modified_text)
    importance = original_prob - new_prob
```

It is conceptually simple but computationally expensive for long text.

### 8.5 Recommended Heatmap Strategy

Use conservative highlighting:

- paragraph background for coarse probability
- phrase-level highlights for top evidence
- avoid coloring every suspicious word
- require high paragraph probability before phrase highlighting
- use multi-source evidence if available

Preferred display:

```text
Article AI probability: 0.78
Paragraph 2: AI-like, 0.81
Highlighted phrases:
  - "it is important to recognize"
  - "offers numerous benefits"
  - "in conclusion"
```

## 9. Article / Paragraph / Phrase Detection

The target system should be hierarchical.

```text
Article-level:
    final document AI probability

Paragraph-level:
    which paragraphs are AI-like

Phrase-level:
    which spans inside suspicious paragraphs support the decision
```

Phrase-level recognition should not be simple red-word matching. It should use span scoring.

Candidate span generation:

- sentence spans
- clause spans
- punctuation-separated spans
- sliding windows of 5-12 words
- stride 3-5 words

Scoring:

```text
span -> TF-IDF / RoBERTa / fusion classifier -> span_prob_ai
```

Gating:

- only score phrases in suspicious paragraphs
- only highlight top-k non-overlapping spans
- prefer phrases over single words

## 10. Span Scoring

Span scoring is the simplest practical route toward phrase-level AI evidence.

Process:

```text
article
 -> split paragraphs
 -> score paragraphs
 -> select suspicious paragraphs
 -> generate candidate spans
 -> score spans
 -> highlight top spans
```

Advantages:

- easy to implement
- does not require phrase labels
- can reuse current TF-IDF / RoBERTa / fusion models
- easy to produce HTML demo

Weaknesses:

- span score may be less reliable for very short spans
- model sees less context if the span is scored alone

Better input:

```text
span + local sentence context
```

## 11. MIL Discussion

MIL means Multiple Instance Learning.

Mapping to this project:

```text
article = bag
spans = instances
article label = bag label
span labels = unknown
```

Architecture:

```text
span_j -> encoder -> span_logit_j
span_logits -> aggregation -> article_logit
article_logit -> binary classification loss
```

Aggregation options:

- mean pooling
- max pooling
- top-k mean
- noisy-OR
- attention pooling

Recommended first version:

```text
top-k mean over span logits
```

MIL is research-interesting because it learns span scores from document labels. However, it is more complex than direct span scoring.

## 12. Teacher-Suggested Mixed-Text Training

The teacher suggested constructing mixed-origin texts:

- Insert human-written text into AI-generated documents.
- Insert AI-generated text into human-written documents.
- Label inserted spans.
- Train and evaluate both document classification and span localization.

This is a strong method because it creates span labels without manual annotation.

Example:

```text
AI base document:
    AI paragraph 1        label 1
    inserted human span   label 0
    AI paragraph 2        label 1

Human base document:
    human paragraph 1     label 0
    inserted AI span      label 1
    human paragraph 2     label 0
```

This enables direct training of:

- paragraph classifier
- span classifier
- hierarchical detector

Recommended first granularity:

- paragraph insertion
- sentence insertion

Phrase insertion can be added later after the pipeline is stable.

## 13. Mixed-Text Evaluation

Synthetic mixed data gives span-level ground truth. Evaluation can include:

- document accuracy
- paragraph accuracy
- span accuracy
- span macro-F1
- top-k span recall

Useful metric:

```text
top-k span recall:
does the true inserted AI/human span appear in the top-k highlighted spans?
```

This directly tests whether the system can localize AI-written parts.

## 14. Recommended Next Implementation Plan

### Step 1: Fusion classifier

Build:

```text
TF-IDF signal + RoBERTa signal + manual features -> final classifier
```

Evaluate with ablation:

- TF-IDF only
- RoBERTa only
- manual features only
- TF-IDF + manual
- RoBERTa + manual
- TF-IDF + RoBERTa + manual

### Step 2: Mixed-text dataset generator

Create:

`scripts/build_mixed_text_dataset.py`

Output:

- mixed document file
- span label file
- metadata with character offsets and labels

### Step 3: Span classifier

Train:

```text
span_text -> human / AI
```

Use synthetic span labels from mixed-text construction.

### Step 4: Hierarchical inference

Create:

`scripts/hierarchical_predict.py`

Output:

- article score
- paragraph scores
- phrase/span scores

### Step 5: HTML visualization

Create:

`scripts/hierarchical_heatmap.py`

Output:

- final probability
- paragraph background colors
- top evidence phrase highlights
- optional branch contribution display

## 15. Final Narrative for Report

A strong final report should not simply say that we trained classifiers. It should present the project as:

> a hierarchical AI-writing detector combining lexical, contextual, and interpretable linguistic signals, with calibrated article-level prediction and paragraph/phrase-level evidence localization.

This narrative connects:

- existing baselines
- model improvement
- language feature analysis
- teacher-suggested mixed training
- span-level localization
- user-facing heatmap visualization

