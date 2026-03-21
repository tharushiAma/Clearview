# Methodology: Multi-Aspect Mixed Sentiment Resolution

ClearView Final Year Project — academic methodology reference

---

## 1. Research Overview

This project builds a multi-aspect sentiment analysis system for cosmetic product reviews, specifically designed to handle:

- Multi-aspect classification across 7 aspects: stayingpower, texture, smell, price, colour, shipping, packing
- Severe class imbalance — up to 185:1 positive-to-negative ratio before augmentation
- Mixed sentiment resolution — separating conflicting opinions within the same review
- Multi-level explainability through attention, LIME, SHAP, and Integrated Gradients

The model is RoBERTa-base with Aspect-Aware Multi-Head Attention, an Aspect-Oriented Dependency GCN, and Hybrid Loss (Focal + Class-Balanced), augmented with LLM-generated synthetic data for minority classes.

---

## 2. Problem Statement

**Multi-aspect sentiment** — reviews contain opinions about multiple aspects simultaneously, each potentially with a different polarity. "Beautiful colour but terrible smell" → colour: positive, smell: negative.

**Extreme class imbalance** (before augmentation):

| Aspect | Negative | Neutral | Positive | Pos:Neg Ratio |
| --- | --- | --- | --- | --- |
| Price | 17 | 15 | 2,244 | 132:1 |
| Packing | 29 | 6 | 2,070 | 71:1 |
| Smell | 51 | 17 | 872 | 17:1 |
| Colour | 131 | 113 | 1,506 | 11:1 |

Standard cross-entropy collapses to predicting "positive" for everything and still gets 90%+ accuracy because the majority class dominates.

**Mixed sentiment resolution** — "Great colour but awful smell" requires knowing that "awful" attaches to "smell" syntactically, not "colour". The dependency tree captures this; pooled transformer representations don't.

**Explainability** — decisions need to be interpretable for trust and academic credibility. Also the main way to prove MSR is actually working correctly.

---

## 3. Data Pipeline

### Text cleaning

Applied at preprocessing time and replicated in `inference.py` for train-inference parity:

1. Unicode normalization (NFC)
2. HTML entity and tag removal
3. URL and email removal
4. Repeated punctuation normalization (`!!!` → `!`)
5. Garbled token detection
6. Invisible/zero-width character removal

### Stratified split — two-phase rare-class guarantee

Standard stratified split fails when you have only 9 price-negative samples in 13,000 total. The solution:

- Phase 1: Identify rare-class rows (below `min_val_test_samples` threshold). Split those proportionally to val/test first.
- Phase 2: Standard stratified split on the remaining rows.

This guarantees minimum minority-class representation in val/test.

Output: `data/splits/train.csv`, `val.csv`, `test.csv`

### Synthetic augmentation

LLM-generated reviews target the most imbalanced minority classes:

| Aspect | Before ratio | After ratio |
| --- | --- | --- |
| Price | 132:1 | ~11:1 |
| Packing | 71:1 | ~12:1 |
| Smell | 17:1 | ~6:1 |

Output: `data/splits/train_augmented.csv` (10,050 samples total)

---

## 4. Model Architecture

```text
Input Text
    │
    ▼
RoBERTa-base Encoder
(12 layers, 768-dim, 125M params)
    │  last_hidden_state: (batch, seq_len, 768)
    ▼
Aspect-Aware Attention Module
  learnable aspect embeddings (7 × 768)
  aspect embedding is query, token states are K/V
  8-head Multi-Head Attention
  → aspect_representation: (batch, 768)
    │
    ▼
Aspect-Oriented Dependency GCN   [optional; use_dependency_gcn flag]
  2-layer GCN on spaCy dependency parse trees
  aspect-oriented gate: σ(W_g × aspect_emb)
  gated aggregation along dependency edges
  residual connections
  → gcn_output: (batch, 768)
    │
    ▼
Fusion: aspect_repr + gcn_output
    │
    ▼
7 Aspect-Specific Classifiers   [or 1 shared; use_shared_classifier flag]
  Linear(768 → 384) → ReLU → Dropout(0.1) → Linear(384 → 3)
    │
    ▼
Logits → Sentiment {Negative, Neutral, Positive}
```

Total trainable parameters: ~132M (with GCN), ~129M (without GCN)

### RoBERTa encoder

Pre-trained `roberta-base`, full fine-tuning (all 12 layers). Max 128 tokens. RoBERTa over BERT because it was trained longer on more data (160GB vs 16GB) and without the Next Sentence Prediction objective.

### Aspect-Aware Attention

Instead of using [CLS] for classification, I use a learnable aspect embedding as the **query** in MHA. Token hidden states are keys and values.

```python
aspect_query = self.aspect_embeddings(aspect_id)   # (batch, 768)
attended, attn_weights = self.aspect_attention(
    query=aspect_query.unsqueeze(1),
    key=hidden_states,
    value=hidden_states,
    key_padding_mask=~attention_mask.bool()
)
aspect_repr = attended.squeeze(1)  # (batch, 768)
```

The model learns which tokens matter for each aspect separately. The attention weights are also used for XAI visualization. Ablation A2 replaces this with [CLS] pooling.

### Dependency GCN

spaCy extracts dependency edges per sample at dataset creation time. GCN message passing (2 layers):

```text
gate = σ(W_gate × aspect_embedding)       # aspect-specific filter
for each edge (src → dst):
    messages[dst] += gate ⊙ H[src]        # gated message
H_new = ReLU(messages) + H               # residual
```

The gate controls information flow based on aspect relevance. "awful" in "Great colour but awful smell" is syntactically linked to "smell" — the gate for the "colour" query should suppress it. Ablation A1 tests without GCN.

### Classifier heads

7 separate two-layer MLPs per aspect. Different aspects have different linguistic patterns — a shared head forces the same decision boundary for "smells amazing" and "delivers fast". Ablation A5 tests one shared head.

---

## 5. Class imbalance handling

### Focal Loss

Formula: `FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)`

Down-weights examples the model already predicts confidently, forces attention to hard minority examples. γ is set higher for worse-imbalanced aspects:

```yaml
focal_gamma:
  default: 2.0
  smell: 2.5
  price: 3.0
  packing: 3.0
```

### Class-Balanced Loss

Formula: `w_c = (1 − β) / (1 − β^{n_c})`

More principled than inverse-frequency weighting — accounts for diminishing returns from additional samples. β is set higher for worse imbalance:

```yaml
class_balanced_beta:
  default: 0.999
  price: 0.9999
  packing: 0.9999
```

### Dice Loss

Formula: `DL = 1 − (2 · |P ∩ T| + ε) / (|P| + |T| + ε)`

Directly optimizes the Dice coefficient (equivalent to F1). Complements the cross-entropy-based losses.

### Hybrid Loss

```python
total_loss = (1.0 × focal_loss) + (0.5 × cb_loss)  # + optional dice
```

Each aspect gets its own `HybridLoss` instance, auto-configured from class counts in `config.yaml` by `AspectSpecificLossManager`. Ablation A3 tests individual loss functions in isolation.

---

## 6. Explainability

### Attention visualization

Extract MHA weights from `AspectAwareRoBERTa.aspect_attention`. Because the query is the aspect embedding, weights directly show which tokens the model uses for that aspect's prediction. Fast and always available but attention ≠ feature importance — use IG for rigorous attribution.

### LIME

Randomly masks words, observes prediction change, fits a local linear model, reports feature importances. Same code path as normal prediction so it reflects the real model behavior.

### SHAP

Shapley values — each token's average marginal contribution across all coalitions. Uses `shap.Explainer` with Partition algorithm.

### Integrated Gradients

Formula: `IG(x) = (x − x') · ∫₀¹ ∂F(x' + α(x − x'))/∂x dα`

Baseline is all-PAD embeddings. Satisfies the completeness axiom — attribution scores sum exactly to the output change. Implemented via Captum's `LayerIntegratedGradients` on RoBERTa's embedding layer, 50 interpolation steps. This is the most rigorous method and the one I use for most XAI figures in the thesis.

---

## 7. Training Strategy

### Optimizer and schedule

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

Low LR (2e-5) needed for fine-tuning pre-trained RoBERTa without catastrophic forgetting. Warmup prevents early divergence.

### Mixed precision

Roughly 2× speedup on RTX 4060. Gradient clipping at 1.0 prevents exploding gradients.

### Early stopping

Evaluated only at epoch end — not mid-epoch. Mid-epoch evaluations log metrics only and never update `patience_counter`. This prevents premature stopping from noisy mid-epoch checkpoints (this was a bug I had to fix).

Metric: validation Macro-F1. Patience: 5 epochs.

### Training config

| Parameter | Value |
| --- | --- |
| Device | NVIDIA RTX 4060 Laptop GPU |
| Batch Size | 16 |
| Learning Rate | 2.0e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| Warmup Steps | 500 |
| Scheduler | Linear warmup + linear decay |
| Epochs | 30 (early stopping, patience=5) |
| Mixed Precision | Yes (torch AMP) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping Metric | Validation Macro-F1 |

---

## 8. Evaluation

Primary metric is **Macro-F1** — unweighted average across all classes. Treats all classes equally regardless of support. Accuracy is misleading given the imbalance.

`Macro-F1 = mean(F1_negative, F1_neutral, F1_positive)`

Other metrics:

| Metric | Purpose |
| --- | --- |
| Per-class F1 | Shows minority class performance separately |
| Weighted F1 | Aggregate performance accounting for support |
| MCC | Matthews Correlation Coefficient — robust for extreme imbalance, range [-1, 1] |
| Accuracy | Reported but secondary |
| Confusion Matrix | Visual error breakdown per aspect |

Mixed sentiment resolution accuracy: computed by `MixedSentimentEvaluator` — percentage of conflicting reviews where the model correctly identifies the different sentiments per aspect.

---

## 9. Ablation Studies & Baselines

### Ablation studies (7 studies, 17 variants)

| ID | Component | Conditions |
| --- | --- | --- |
| A1 | Dependency GCN | Full model vs. No GCN |
| A2 | Aspect Attention | MHA vs. [CLS] pooling |
| A3 | Loss Function | Hybrid / Focal-only / CB-only / Dice-only / plain CE |
| A4 | Data Augmentation | With synthetic data vs. original only |
| A5 | Classifier Head | 7 aspect-specific heads vs. 1 shared head |
| A6 | Mixed Sentiment Resolution | Full model + GCN vs. No GCN |
| A7 | Hybrid Loss Weights | Focal 1.0 + CB 0.5 vs. Focal 1.0 + CB 1.0 |

### Baselines (5 models)

| ID | Model | Description |
| --- | --- | --- |
| B1 | PlainRoBERTa | RoBERTa + [CLS] head, no aspect awareness, CE loss |
| B2 | RoBERTa+CE | Full architecture but CrossEntropy only |
| B3 | BERTBaseline | BERT-base-uncased + [CLS] head, CE loss |
| B4 | TF-IDF+SVM | Classical: TF-IDF features + LinearSVC per aspect |
| B5 | Flat ABSA RoBERTa | Aspect attention, shared head, CE loss (no GCN/hybrid loss) |

### Running experiments

Experiments are run interactively via `notebooks/12_experiment_runner.ipynb`. Results are analysed and charts/LaTeX tables generated via `notebooks/13_results_analyzer.ipynb`.

---

## 10. References

1. RoBERTa: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019
2. Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
3. Class-Balanced Loss: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
4. Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
5. LIME: Ribeiro et al., "Why Should I Trust You?", KDD 2016
6. Integrated Gradients: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
7. SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
8. Captum: Kokhlikyan et al., "Captum: A unified and generic model interpretability library for PyTorch", 2020

---

ClearView FYP — Tharushi Amasha — 2025
