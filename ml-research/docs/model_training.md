# Model Training — ClearView

Last updated: 2026-03-04

## Architecture

```text
Input Review
    ↓
RoBERTa-base (125M params, 768-dim)
    ↓
Aspect-Aware MHA Attention  [8 heads, aspect embedding as query]
    ↓
Aspect-Oriented Dependency GCN  [2 layers, aspect-gated messages]
    ↓
7 Aspect-Specific Classifiers  [768→384→3]
    ↓
Sentiment: Negative / Neutral / Positive
```

Key model classes in `src/models/model.py`:

- `AspectAwareRoBERTa` — RoBERTa + MHA attention + per-aspect classifiers
- `AspectOrientedDepGCN` — 2-layer GCN with aspect-gated message passing
- `MultiAspectSentimentModel` — wires everything together, reads ablation flags from config

Ablation flags (set in `configs/config.yaml`):

| Flag | Default | Controls |
| --- | --- | --- |
| `use_dependency_gcn` | `true` | GCN on/off (Ablation A1) |
| `use_aspect_attention` | `true` | MHA vs CLS pooling (Ablation A2) |
| `use_shared_classifier` | `false` | 7 heads vs 1 shared (Ablation A5) |

---

## Mixed Sentiment Resolution

For a review like "The shipping was slow but the packing was elegant":

1. spaCy parses the dependency tree — "slow" links to "shipping", "elegant" to "packing"
2. The GCN's aspect-oriented gate suppresses cross-aspect signals based on which aspect is being queried
3. MSR Delta XAI (`explain_msr_delta` in `inference.py`) proves this works by measuring per-token confidence drop under token masking — tokens relevant to one aspect should barely affect another aspect's prediction

---

## Class Imbalance Strategy

**LLM Augmentation** — synthetic reviews for the worst minority classes:

| Aspect | Before (neg:pos) | After |
| --- | --- | --- |
| Price | 174:1 | ~11:1 |
| Packing | 185:1 | ~12:1 |
| Smell | 17:1 | ~6:1 |

Script: `data/data_layer/create_train_aug.py` → `data/splits/train_augmented.csv` (10,050 samples)

**Hybrid Loss** (`src/models/losses.py`):

| Loss | Purpose | Weight |
| --- | --- | --- |
| Focal Loss (γ per aspect) | Focus on hard/minority examples | 1.0 |
| Class-Balanced Loss (β per aspect) | Reweight by effective sample count | 0.5 |
| Dice Loss | Directly optimize F1-score | 0.3 |

Per-aspect parameters (auto-set by `AspectSpecificLossManager`):

| Aspect | Focal γ | CB β |
| --- | --- | --- |
| price, packing | 3.0 | 0.9999 |
| smell | 2.5 | 0.999 |
| others | 2.0 | 0.999 |

**Stratified Split** — two-phase split in `data/data_layer/preprocess_and_split.py`. Reserves rare-class rows first, splits them to val/test proportionally, then does standard stratified split on the rest. Ensures rare aspects (price-neg, packing-neu) actually appear in val/test.

---

## Training Config

| Parameter | Value |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU |
| Batch Size | 16 |
| Learning Rate | 2.0e-5 (AdamW) |
| Warmup Steps | 500 |
| Scheduler | Linear warmup + linear decay |
| Epochs | 30 (early stopping, patience=5) |
| Mixed Precision | Enabled (torch AMP) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping Metric | Validation Macro-F1 |

```bash
python src/models/train.py --config configs/config.yaml

# Resume from checkpoint
python src/models/train.py --config configs/config.yaml \
    --resume outputs/cosmetic_sentiment_v1/best_model.pt
```

---

## Results (Test Set)

| Metric | Score |
| --- | --- |
| Overall Accuracy | 92.14% |
| Overall Macro-F1 | 0.7981 |
| Weighted F1 | 0.9242 |
| MCC | 0.7842 |

Per-aspect Macro-F1:

| Aspect | Macro-F1 |
| --- | --- |
| Shipping | 0.8507 |
| Stayingpower | 0.7920 |
| Colour | 0.7791 |
| Texture | 0.7726 |
| Smell | 0.7381 |
| Packing | 0.5989 |
| Price | 0.4944 |

---

## XAI Methods

All implemented in `outputs/cosmetic_sentiment_v1/evaluation/inference.py`:

| Method | Flag | Notes |
| --- | --- | --- |
| Attention heatmap | `--explain attention` | Fast, always available |
| LIME | `--explain lime` | Word-level contribution |
| SHAP | `--explain shap` | Shapley values |
| Integrated Gradients | `--explain ig` | Satisfies completeness axiom; most rigorous |
| MSR Delta | `--explain msr` | Proves mixed sentiment separation |

```bash
python outputs/cosmetic_sentiment_v1/evaluation/inference.py \
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt \
    --text "Great colour but the smell is awful" \
    --aspect colour --explain all --save-path results/xai.png
```

---

## Ablation Experiments

```bash
python src/experiments/experiment_runner.py --list
python src/experiments/experiment_runner.py --group ablations
python src/experiments/experiment_runner.py --group baselines
python src/experiments/results_analyzer.py   # generate tables + charts
```

| Group | IDs | What it tests |
| --- | --- | --- |
| Ablations | A1–A6 | GCN, attention, loss, augmentation, classifier, preprocessing |
| Baselines | B1–B4 | PlainRoBERTa, RoBERTa+CE, BERT-base, TF-IDF+SVM |

---

ClearView FYP — Tharushi Amasha, 2025
