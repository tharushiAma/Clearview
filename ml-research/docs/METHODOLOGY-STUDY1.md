# ClearView — Methodology and Technical Architecture

## Overview

ClearView does multi-aspect sentiment analysis on cosmetic product reviews. The goal is to classify sentiment (positive, negative, neutral, or not mentioned) for 7 aspects simultaneously — stayingpower, texture, smell, price, colour, shipping, packing — while handling the fact that reviews often have conflicting opinions about different aspects in the same sentence.

**Dataset:** Customer reviews from e-commerce cosmetic platforms. ~3,500 original reviews, ~8,000 after augmentation for training, ~1,200 each for val and test.

---

## 1. Aspect-Based Sentiment Analysis (ABSA)

Each review gets classified per-aspect into 4 classes:

- **Negative (0)** — negative opinion about that aspect
- **Neutral (1)** — mixed or ambivalent opinion
- **Positive (2)** — positive opinion
- **None/Null (3)** — aspect not mentioned in the review

This is a multi-label, multi-class problem: 7 aspects × 4 classes, all predicted from a single forward pass.

## 2. Multi-Aspect Sentiment Resolution (MSR)

The tricky part is when a single review says something like "The texture is good but the price is too high." A plain transformer tends to let signals bleed across aspects — the negative "price" token affects the "texture" prediction and vice versa.

MSR adds a conflict-aware gating mechanism that:

1. Detects when multiple aspects have conflicting sentiments (the conflict detector outputs a probability)
2. Uses that conflict signal to apply aspect-specific refinement to the predictions
3. Has a tunable strength parameter λ (set to 0.3 in the final model)

λ = 0 means no MSR adjustment, λ = 1.0 is maximum correction. 0.3 was chosen empirically — enough to improve minority aspects without hurting majority ones.

## 3. Class imbalance

The raw data is massively skewed. Most cosmetic reviews are positive, especially for price and packing:

| Aspect | Negative | Neutral | Positive | Pos:Neg Ratio |
| --- | --- | --- | --- | --- |
| Price | 17 | 15 | 2,244 | 132:1 |
| Packing | 29 | 6 | 2,070 | 71:1 |
| Smell | 51 | 17 | 872 | 17:1 |
| Colour | 131 | 113 | 1,506 | 11:1 |

With standard cross-entropy, the model learns to predict "positive" for everything and gets 90%+ accuracy while completely failing on minority classes. Three strategies to fix this:

1. **LLM synthetic augmentation** — generated reviews targeting the worst minority classes. Brought price from 132:1 to ~11:1 before training even starts.
2. **Hybrid loss** — Focal Loss + Class-Balanced Loss, with per-aspect γ and β tuned to the imbalance severity.
3. **Two-phase stratified split** — standard stratified split fails when you only have 9 negative price samples. Reserved minority-class rows first, split them proportionally to val/test, then ran stratified split on the rest.

---

## 2. Model Architecture

```text
Input Text
    ↓
RoBERTa-base Encoder (768-dim)
    ↓
Aspect-Aware Attention (learnable aspect embeddings as queries, 8-head MHA)
    ↓
Aspect-Oriented Dependency GCN (2-layer, aspect-gated message passing)
    ↓
7 Aspect-Specific Classifiers (768 → 384 → 3)
    ↓
Conflict Detector → conflict probability
    ↓
MSR Refinement (if enabled, λ=0.3)
    ↓
Final predictions: 7 aspects × 4 classes
```

**Total parameters: ~132M** (RoBERTa-base 125M + GCN + heads)

### RoBERTa-base

Using `roberta-base` from HuggingFace with full fine-tuning. I chose RoBERTa over BERT because it was trained longer on more data (160GB vs 16GB) and without the Next Sentence Prediction objective, which adds noise for sentiment tasks. Max 128 tokens.

### Aspect-Aware Attention

Instead of using the [CLS] token representation for classification, I use learnable aspect embeddings as the **query** in multi-head attention. The token hidden states are keys and values. This forces the model to attend specifically to what's relevant for the queried aspect rather than mixing everything together.

```python
aspect_query = self.aspect_embeddings(aspect_id)  # (batch, 768)
attended, attn_weights = self.aspect_attention(
    query=aspect_query.unsqueeze(1),
    key=hidden_states,
    value=hidden_states,
    key_padding_mask=~attention_mask.bool()
)
```

The attention weights are also used for attention-based XAI visualization.

Ablation A2 tests this by replacing with simple [CLS] pooling.

### Dependency GCN

spaCy parses the dependency tree for each review. The GCN does 2 rounds of message passing along those edges, with an aspect-specific gate controlling how much each token's representation is allowed to influence the target.

```text
gate = σ(W_gate × aspect_embedding)  # aspect-specific filter
for each edge (src → dst):
    messages[dst] += gate ⊙ H[src]   # gated message
H_new = ReLU(messages) + H           # residual
```

The point is that "awful" in "Great colour but the smell is awful" is syntactically linked to "smell", not "colour". The aspect gate for "colour" should suppress the "awful" signal. Ablation A1 tests with/without GCN.

### Classifier heads

7 separate two-layer MLPs (768→384→3), one per aspect. Separate heads let each aspect learn its own decision boundary — "smells amazing" and "delivers fast" have different linguistic patterns even though both are positive. Ablation A5 tests a single shared head.

### Conflict Detector

A small classifier that takes the sentiment probability distributions across all 7 aspects, plus entropy and sentiment contrast features, and outputs a conflict probability in [0,1]. Reviews where multiple aspects have opposing sentiments should score high.

---

## 3. Data Pipeline

**Preprocessing:** Unicode normalization, HTML/URL removal, repeated punctuation normalization, garbled token detection. Applied at preprocessing time AND replicated in `inference.py` to ensure train-test parity.

**Split:** 70/15/15. Two-phase to guarantee minority classes in val/test.

**Augmentation:** LLM-generated reviews for the most imbalanced classes. 923 generated, 810 kept after deduplication (113 removed). Also added light noise injection to make synthetic reviews less obviously clean.

**Training set:** `train_augmented.csv` — 10,050 samples total.

---

## 4. Ablation Study

8-configuration matrix isolating each component:

| Config | Data | MSR | Description |
| --- | --- | --- | --- |
| 1_base_base | Original only | No | Control |
| 1_base_msr | Original only | Yes | MSR alone |
| 2_sampler_base | Weighted sampling | No | Class balancing only |
| 2_sampler_msr | Weighted sampling | Yes | Sampling + MSR |
| 3_synth_base | Synthetic augmentation | No | Synthetic data only |
| 3_synth_msr | Synthetic augmentation | Yes | Synthetic + MSR |
| 4_full_base | All augmentations | No | Data-only |
| **4_full_msr** | **All augmentations** | **Yes** | **Full model** |

---

## 5. Results

| Model | Overall Macro-F1 | MSR Error Reduction |
| --- | --- | --- |
| RoBERTa Baseline | 0.6953 | 0 |
| Full EAGLE (MSR) | 0.7241 | 50 |

Per-aspect:

| Aspect | Baseline F1 | Full Model F1 | Gain |
| --- | --- | --- | --- |
| Texture | 0.7952 | 0.7923 | -0.29% |
| Price | 0.3871 | 0.4828 | +9.57% |
| Smell | 0.7195 | 0.7374 | +1.79% |
| Colour | 0.8093 | 0.7611 | -4.82% |
| Shipping | 0.7726 | 0.7553 | -1.73% |
| Stayingpower | 0.8114 | 0.8099 | -0.15% |
| Packing | 0.5717 | 0.7298 | +15.81% |

The gains are concentrated in the minority aspects (price, packing) which were the worst before augmentation. Some majority aspects dropped slightly — the MSR λ=0.3 is a bit conservative to avoid hurting those. Could tune per-aspect in future work.

---

## 6. Evaluation

Primary metric is **Macro-F1** — unweighted average across all classes. Critical for imbalanced data because accuracy is misleading (predicting "positive" for everything gives 90%+ accuracy).

Also tracked: per-class F1, Weighted-F1, MCC (Matthews Correlation Coefficient), confusion matrices per aspect.

For mixed sentiment specifically: `MixedSentimentEvaluator` identifies reviews where ≥2 aspects have conflicting predicted sentiments and checks if they were resolved correctly against ground truth.

---

## 7. Limitations

- English only — would need retraining or multilingual backbone for other languages
- Cosmetic-specific aspects — adapting to other product categories needs new annotations
- Max 128 tokens — very long reviews get truncated
- MSR adds ~15-20% inference overhead
- Still needs real labeled data to train; synthetic helps but doesn't replace
