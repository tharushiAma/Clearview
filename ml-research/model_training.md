# Model Training Overview — ClearView

*Last updated: 2026-03-04*

This document gives a concise technical reference for the training process, architecture, and evaluation of the ClearView multi-aspect sentiment model.

---

## 1. Project Objective

**Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain**

The system classifies sentiment (Positive / Neutral / Negative) across **7 aspects** of cosmetic reviews — *stayingpower, texture, smell, price, colour, shipping, packing* — while:
- Correctly resolving **mixed sentiments** within the same review
- Handling **severe class imbalance** (up to 185:1 before augmentation)
- Providing **multi-level explainability** (Attention, LIME, SHAP, Integrated Gradients)

---

## 2. Architecture

```
Input Review
    ↓
RoBERTa-base Encoder (125M params, 768-dim)
    ↓
Aspect-Aware MHA Attention  [8 heads, aspect embedding as query]
    ↓
Aspect-Oriented Dependency GCN  [2 layers, aspect-gated messages]
    ↓
7 Aspect-Specific Classifiers  [768→384→3]
    ↓
Sentiment: Negative / Neutral / Positive
```

**Key model classes** (`src/models/model.py`):
- `AspectAwareRoBERTa` — RoBERTa + MHA attention + per-aspect classifiers
- `AspectOrientedDepGCN` — 2-layer GCN with aspect-gated message passing
- `MultiAspectSentimentModel` — full pipeline with ablation flag support

**Ablation flags** (set in `configs/config.yaml` or via override):

| Flag | Default | Purpose |
|------|---------|---------|
| `use_dependency_gcn` | `true` | Enable/disable GCN (Ablation A1) |
| `use_aspect_attention` | `true` | MHA vs CLS pooling (Ablation A2) |
| `use_shared_classifier` | `false` | 7 heads vs 1 shared head (Ablation A5) |

---

## 3. Mixed Sentiment Resolution

For a review like *"The shipping was slow but the packing was elegant"*:

1. **Dependency Parsing** (spaCy `en_core_web_sm`) extracts the syntactic tree showing *"slow"* is linked to *"shipping"* and *"elegant"* to *"packing"*
2. **Aspect-Oriented Gating** in the GCN selectively allows information relevant to the queried aspect to flow, suppressing noise from other aspects
3. **MSR Delta XAI** (`inference.py: explain_msr_delta`) proves this separation by measuring per-token confidence drop when each token is masked — tokens from one aspect should have low influence on another aspect's prediction

---

## 4. Class Imbalance Strategy

### A. LLM Synthetic Augmentation

Minority classes (Negative, Neutral) augmented using LLM-generated synthetic reviews:

| Aspect | Before (neg:pos ratio) | After |
|--------|------------------------|-------|
| Price | 174:1 | ~11:1 |
| Packing | 185:1 | ~12:1 |
| Smell | 17:1 | ~6:1 |

**Script**: `data/data_layer/create_train_aug.py`
**Output**: `data/splits/train_augmented.csv` (10,050 samples) + `augmentation_impact.md`

### B. Hybrid Loss Function (`src/models/losses.py`)

| Loss | Purpose | Weight |
|------|---------|--------|
| Focal Loss (γ per aspect) | Focus on hard/minority examples | 1.0 |
| Class-Balanced Loss (β per aspect) | Reweight by effective sample count | 0.5 |
| Dice Loss | Directly optimize F1-score | 0.3 |

**Aspect-specific parameters** (auto-configured by `AspectSpecificLossManager`):

| Aspect | Focal γ | CB β |
|--------|---------|------|
| price, packing | 3.0 | 0.9999 |
| smell | 2.5 | 0.999 |
| others | 2.0 | 0.999 |

### C. Stratified Splitting with Rare-Class Guarantee

`data/data_layer/preprocess_and_split.py` uses a **two-phase split**:
1. Reserve all rare-class rows → split proportionally to val/test first
2. Standard stratified split on remaining rows

Ensures rare aspects (price-neg, packing-neu) have ≥ a minimum count in val/test to provide reliable evaluation signal.

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Device | NVIDIA GeForce RTX 4060 Laptop GPU |
| Batch Size | 16 |
| Learning Rate | 2.0e-5 (AdamW) |
| Warmup Steps | 500 |
| Scheduler | Linear warmup + linear decay |
| Epochs | 30 (early stopping, patience=5) |
| Mixed Precision | Enabled (torch AMP) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping Metric | Validation Macro-F1 |

**Run training:**
```bash
python src/models/train.py --config configs/config.yaml
```

**Resume from checkpoint:**
```bash
python src/models/train.py --config configs/config.yaml \
    --resume outputs/cosmetic_sentiment_v1/best_model.pt
```

---

## 6. Performance Results (Test Set)

| Metric | Score |
|--------|-------|
| Overall Accuracy | **92.14%** |
| Overall Macro-F1 | **0.7981** |
| Weighted F1 | **0.9242** |
| MCC | **0.7842** |

**Per-Aspect Macro-F1:**

| Aspect | Macro-F1 | Notes |
|--------|----------|-------|
| Shipping | 0.8507 | Most balanced |
| Stayingpower | 0.7920 | |
| Colour | 0.7791 | |
| Texture | 0.7726 | |
| Smell | 0.7381 | Improved via augmentation |
| Packing | 0.5989 | Significant improvement |
| Price | 0.4944 | Stable despite data scarcity |

---

## 7. Explainable AI (XAI) Integration

All methods implemented in `outputs/cosmetic_sentiment_v1/evaluation/inference.py`:

| Method | CLI flag | Key benefit |
|--------|----------|------------|
| Attention heatmap | `--explain attention` | Fast, always available |
| LIME | `--explain lime` | Word-level contribution |
| SHAP | `--explain shap` | Shapley values, bar chart |
| **Integrated Gradients** | `--explain ig` | Satisfies completeness axiom; most theoretically rigorous |
| **MSR Delta** | `--explain msr` | Proves mixed sentiment separation between aspects |

**Example:**
```bash
python outputs/cosmetic_sentiment_v1/evaluation/inference.py \
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt \
    --text "Great colour but the smell is awful" \
    --aspect colour --explain all --save-path results/xai.png
```

---

## 8. Experiments

**19 experiments** implemented in `src/experiments/`:

```bash
python src/experiments/experiment_runner.py --list    # see all
python src/experiments/experiment_runner.py --group ablations
python src/experiments/experiment_runner.py --group baselines
python src/experiments/results_analyzer.py            # generate tables
```

| Group | IDs | Description |
|-------|-----|-------------|
| Ablations | A1–A6 | GCN, attention, loss, augmentation, classifier, preprocessing |
| Baselines | B1–B4 | PlainRoBERTa, RoBERTa+CE, BERT-base, TF-IDF+SVM |

---

*ClearView FYP — Tharushi Amasha, 2025*
