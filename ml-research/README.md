# ClearView — Multi-Aspect Mixed Sentiment Analysis with Explainability

> **Final Year Project** | BEng Software Engineering
> *Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain*

A research implementation combining **RoBERTa**, **Aspect-Aware Attention**, **Dependency GCN**, and advanced class-imbalance handling (Hybrid Loss + LLM-based synthetic augmentation) to analyze sentiment across 7 aspects of cosmetic product reviews — with full multi-level explainability.

---

## 🎯 Research Objectives

1. **Multi-Aspect Sentiment Analysis** — Classify sentiment (Positive / Neutral / Negative) for 7 aspects: *stayingpower, texture, smell, price, colour, shipping, packing*
2. **Mixed Sentiment Resolution** — Correctly separate conflicting sentiments in the same review (e.g. *"Love the colour but hate the smell"*)
3. **Severe Class Imbalance Handling** — Hybrid loss (primarily Focal + Class-Balanced) combined with LLM synthetic augmentation
4. **Multi-Level Explainability** — Attention visualization, LIME, SHAP, and **Integrated Gradients**

---

## 📊 Dataset

| Split | Source | Samples |
|-------|--------|---------|
| Train (augmented) | Original + LLM synthetic | 10,050 |
| Validation | Original stratified | ~810 |
| Test | Original stratified | ~810 |

**Augmentation impact** (post-synthetic integration):

| Aspect | Before (neg ratio) | After (neg ratio) |
|--------|--------------------|-------------------|
| Price | 174:1 | ~11:1 |
| Packing | 185:1 | ~12:1 |
| Smell | 17:1 | ~6:1 |

Data pipeline: `data/data_layer/preprocess_and_split.py` → `data/data_layer/create_train_aug.py`

---

## 🏗️ Architecture

```
Input Review
    │
    ▼
RoBERTa-base Encoder  (768-dim contextual embeddings)
    │
    ▼
Aspect-Aware Attention  (learnable aspect embeddings × 8-head MHA)
    │              ↑ ablation: use_aspect_attention flag
    ▼
Aspect-Oriented Dependency GCN  (2-layer, aspect-gated message passing)
    │              ↑ ablation: use_dependency_gcn flag
    ▼
7 Aspect-Specific Classifiers  (768→384→3)
    │              ↑ ablation: use_shared_classifier flag
    ▼
Sentiment: Negative / Neutral / Positive
```

**Total parameters: ~132M** (RoBERTa-base + GCN + heads)

---

## 📁 Project Structure

```
ml-research/
├── configs/
│   └── config.yaml                 # All hyperparameters, loss config, paths
│
├── data/
│   ├── raw/                        # Original annotated CSV
│   ├── augmented/                  # LLM-generated synthetic samples
│   ├── splits/                     # train_augmented.csv, val.csv, test.csv
│   └── data_layer/
│       ├── preprocess_and_split.py # Cleaning + stratified splitting
│       └── create_train_aug.py     # Merge synthetic data, generate augmentation_impact.md
│
├── src/
│   ├── models/
│   │   ├── model.py                # AspectAwareRoBERTa + DepGCN + MultiAspectSentimentModel
│   │   ├── losses.py               # FocalLoss + ClassBalancedLoss + DiceLoss + HybridLoss
│   │   └── train.py                # Trainer with early stopping, mixed precision, MixedSentimentEvaluator
│   └── experiments/
│       ├── baseline_models.py      # PlainRoBERTa, BERT-base, TF-IDF+SVM baselines
│       ├── ablation_configs.py     # 6 ablation config generators
│       ├── experiment_runner.py    # Unified CLI for all experiments
│       └── results_analyzer.py    # Generates Markdown + LaTeX + bar charts
│
├── utils/
│   ├── data_utils.py               # CosmeticReviewDataset, dependency parsing, DataLoaders
│   └── metrics.py                  # AspectSentimentEvaluator, MixedSentimentEvaluator, ErrorAnalyzer
│
├── outputs/
│   └── cosmetic_sentiment_v1/
│       └── evaluation/
│           └── inference.py        # SentimentPredictor with LIME, SHAP, IG, MSR Delta
│
├── inference_bridge/
│   ├── trained_model_adapter.py      # Bridge: website ↔ SentimentPredictor
│   └── trained_model_xai.py         # XAI bridge (IG, LIME, SHAP, MSR Delta)
│
└── tests/
    ├── comprehensive_test.py        # Full prediction + XAI tests (needs checkpoint)
    ├── test_integration.py          # Website adapter integration test
    └── test_model_components.py    # Unit tests — no checkpoint required
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Prepare Data

```bash
# Step 1: Clean and split original data
python data/data_layer/preprocess_and_split.py

# Step 2: Merge synthetic augmentation into training set
python data/data_layer/create_train_aug.py
```

### 3. Verify Architecture (no checkpoint needed)

```bash
python tests/test_model_components.py
```

### 4. Train

```bash
python src/models/train.py --config configs/config.yaml
```

### 5. Run Inference

```bash
# Single prediction
python outputs/cosmetic_sentiment_v1/evaluation/inference.py \
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt \
    --text "I love the colour but the smell is terrible" \
    --aspect colour

# All XAI methods
python outputs/cosmetic_sentiment_v1/evaluation/inference.py \
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt \
    --text "Great texture but overpriced" \
    --aspect texture --explain all --save-path results/xai_output.png
```

### 6. Run Ablation / Baseline Experiments

```bash
# List all 22 experiments
python src/experiments/experiment_runner.py --list

# Run all baseline comparisons
python src/experiments/experiment_runner.py --group baselines

# Run all ablation studies
python src/experiments/experiment_runner.py --group ablations

# Analyze results → generates Markdown + LaTeX + charts
python src/experiments/results_analyzer.py
```

---

## 📈 Performance Results (Trained Model)

| Metric | Score |
|--------|-------|
| Overall Accuracy | **92.47%** |
| Overall Macro-F1 | **0.7944** |
| Weighted F1 | **0.9236** |
| MCC | **0.7900** |

**Per-Aspect Macro-F1:**

| Aspect | Macro-F1 |
|--------|----------|
| Texture | 0.8088 |
| Shipping | 0.7975 |
| Stayingpower | 0.7933 |
| Colour | 0.7647 |
| Smell | 0.7311 |
| Packing | 0.5997 |
| Price | 0.3275 |

---

## 🔬 Explainability Methods

All four methods are available in `inference.py` via `SentimentPredictor`:

| Method | Flag | Description |
|--------|------|-------------|
| Attention Visualization | `--explain attention` | MHA weights over tokens |
| LIME | `--explain lime` | Local perturbation-based word contributions |
| SHAP | `--explain shap` | Shapley value attributions |
| Integrated Gradients | `--explain ig` | Meets completeness axiom; most rigorous for transformers |

---

## 🧪 Ablation Studies (22 Experiments)

| ID | Study | Variants |
|----|-------|---------|
| A1 | Dependency GCN | With / Without GCN |
| A2 | Aspect Attention | MHA attention / CLS pooling |
| A3 | Loss Function | Hybrid / Focal / CB / Dice / CE |
| A4 | Data Augmentation | With / Without LLM synthetic data |
| A5 | Classifier Head | 7 aspect-specific / 1 shared head |
| A6 | Mixed Sentiment Resolution | MSR Eval: Full model + GCN / No GCN |
| A7 | Hybrid Loss Weights | Focal 1.0 + CB 0.5 / Focal 1.0 + CB 1.0 |
| B1-B5 | Baselines | PlainRoBERTa / RoBERTa+CE / BERT-base / TF-IDF+SVM / Flat ABSA RoBERTa |

---

## ⚙️ Key Configuration (configs/config.yaml)

```yaml
model:
  roberta_model: "roberta-base"
  hidden_dim: 768
  num_aspects: 7
  num_classes: 3
  gcn_layers: 2
  dropout: 0.1
  use_dependency_gcn: true
  use_aspect_attention: true      # ablation A2
  use_shared_classifier: false    # ablation A5

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 30
  early_stopping_patience: 5
  early_stopping_metric: macro_f1
  loss_weights:
    focal: 1.0
    cb: 0.5
  focal_gamma:
    default: 2.0
    price: 3.0
    packing: 3.0
    smell: 2.5
```

---

## � Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `batch_size` to 8 or 4, enable `mixed_precision: true` |
| Low minority class recall | Increase `focal_gamma` for that aspect |
| Import errors | Run from `ml-research/` directory; check `sys.path` |
| Checkpoint not found | Run `train.py` first |
| captum not installed | `pip install captum` (needed for Integrated Gradients) |

---

## 📄 Citation

```bibtex
@article{clearview2025,
  title={Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution
         with Explainability in the Cosmetic Domain},
  author={Tharushi Amasha},
  year={2025}
}
```
