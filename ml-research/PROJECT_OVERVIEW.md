# Project Files Overview — ClearView

> *Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain*

---

## 📁 Actual File Structure

```
ml-research/
│
├── 📄 README.md                      # Project overview & quick start
├── 📄 model_training.md              # Training config & results
├── 📄 PROJECT_OVERVIEW.md            # This file — file-by-file guide
├── 📄 METHODOLOGY-STUDY2.md          # Full academic methodology writeup
├── 📄 requirements.txt               # Python dependencies
│
├── 📂 configs/
│   └── config.yaml                   # All hyperparameters, paths, loss config
│
├── 📂 data/
│   ├── raw/                          # Original annotated CSV files
│   ├── augmented/                    # LLM-generated synthetic samples
│   ├── splits/
│   │   ├── train_augmented.csv       # 10,050 samples (original + synthetic)
│   │   ├── val.csv                   # ~810 stratified samples
│   │   └── test.csv                  # ~810 stratified samples
│   └── data_layer/
│       ├── _common.py                # Shared constants (ASPECTS, LABEL_MAP, etc.)
│       ├── preprocess_and_split.py   # Text cleaning + two-phase stratified split
│       └── create_train_aug.py       # Merge synthetic data + generate augmentation_impact.md
│
├── 📂 src/
│   ├── __init__.py
│   ├── 📂 models/
│   │   ├── model.py                  # Core model architecture
│   │   ├── losses.py                 # Loss functions for imbalance
│   │   └── train.py                  # Training script
│   └── 📂 experiments/
│       ├── __init__.py
│       ├── baseline_models.py        # 4 baseline model implementations
│       ├── ablation_configs.py       # 6 ablation config generators + print_experiment_plan
│       ├── experiment_runner.py      # Unified CLI runner for all 19 experiments
│       └── results_analyzer.py      # Generates Markdown + LaTeX + bar charts
│
├── 📂 utils/
│   ├── data_utils.py                 # CosmeticReviewDataset, DataLoaders, DependencyParser
│   └── metrics.py                    # AspectSentimentEvaluator, MixedSentimentEvaluator, ErrorAnalyzer
│
├── 📂 outputs/
│   └── cosmetic_sentiment_v1/
│       └── evaluation/
│           └── inference.py          # SentimentPredictor (predict, LIME, SHAP, IG, MSR Delta)
│
├── 📂 inference_bridge/
│   ├── trained_model_adapter.py      # Website ↔ SentimentPredictor bridge
│   └── trained_model_xai.py         # XAI methods (IG, LIME, SHAP, MSR Delta) for website
│
├── 📂 tests/
│   ├── comprehensive_test.py         # Full prediction + optional XAI tests
│   ├── test_integration.py           # Website adapter integration test
│   └── test_model_components.py     # Architecture unit tests — no checkpoint needed
│
└── 📂 results/
    └── experiments/
        ├── all_results.json           # Incremental experiment results
        └── analysis/                  # Charts + Markdown + LaTeX tables
```

---

## 🎯 Key Files Explained

---

### `configs/config.yaml`
**Purpose**: Single source of truth for all settings

**Key sections**:
- `model`: roberta_model, hidden_dim, gcn_layers, dropout, ablation flags
- `training`: batch_size, lr, epochs, loss_weights, focal_gamma per aspect, early_stopping
- `data`: paths to split CSVs, max_seq_length, spacy_model
- `aspects`: list of 7 aspect names
- `training.class_counts`: actual counts from `train_augmented.csv` (used by losses)

**When to edit**:
- Change batch size for GPU memory
- Enable/disable GCN or aspect attention for ablation
- Tune focal_gamma / class_balanced_beta for a specific aspect

---

### `src/models/model.py`
**Purpose**: Core model architecture

**Classes**:
- `AspectAwareRoBERTa` — RoBERTa-base + 8-head MHA with learnable aspect queries + per-aspect classifiers
  - Flag `use_aspect_attention=False` → CLS pooling (Ablation A2)
  - Flag `use_shared_classifier=True` → single shared head (Ablation A5)
- `AspectOrientedDepGCN` — 2-layer GCN with aspect-gated message passing on dependency edges
- `MultiAspectSentimentModel` — integrates both above; reads ablation flags from config
- `create_model(config)` — factory function

**Ablation flags supported**:
```python
model:
  use_dependency_gcn: true/false    # A1
  use_aspect_attention: true/false  # A2
  use_shared_classifier: true/false # A5
```

---

### `src/models/losses.py`
**Purpose**: Handle severe class imbalance via specialized loss functions

**Classes**:
- `FocalLoss` — focuses learning on hard minority examples; γ controls intensity
- `ClassBalancedLoss` — reweights by effective sample count; β controls tightness
- `DiceLoss` — directly optimizes F1-score (Dice coefficient)
- `HybridLoss` — combines all three; constructor: `HybridLoss(samples_per_class, focal_gamma, cb_beta, weights)`
- `AspectSpecificLossManager` — auto-configures one HybridLoss per aspect based on class counts in config

**Parameters automatically set per aspect** (from `config.yaml: training.class_counts`):
- price, packing → γ=3.0, β=0.9999 (extreme imbalance)
- smell → γ=2.5, β=0.999
- others → γ=2.0, β=0.999

---

### `src/models/train.py`
**Purpose**: Full training pipeline

**Key features**:
- `Trainer` class: mixed precision (AMP), gradient clipping, linear LR schedule
- Early stopping on **validation Macro-F1 only at epoch end** (mid-epoch evaluations only log, never update patience — bug-fixed)
- `MixedSentimentEvaluator` integrated in test phase using `review_ids` from dataloader
- Checkpoint saves `model_state_dict`, `config`, `epoch`, `best_val_metric`

**Run**:
```bash
python src/models/train.py --config configs/config.yaml
python src/models/train.py --config configs/config.yaml --resume outputs/.../best_model.pt
```

---

### `utils/data_utils.py`
**Purpose**: Data loading and dependency parsing

**Key classes**:
- `CosmeticReviewDataset` — reads train/val/test CSV, tokenizes with RoBERTa, optionally runs spaCy dependency parsing, returns edge_index for GCN
- `collate_fn_with_dependencies` — batches samples + pads edge_index tensors
- `create_dataloaders` — factory for train/val/test DataLoaders
- `compute_class_weights` — returns `torch.Tensor` of inverse-frequency weights

---

### `utils/metrics.py`
**Purpose**: Comprehensive evaluation

**Key classes**:
- `AspectSentimentEvaluator` — per-aspect accuracy, Macro-F1, Weighted-F1, MCC, confusion matrices, LaTeX table generator
- `MixedSentimentEvaluator` — identifies reviews where aspects have conflicting sentiments and measures resolution accuracy
- `ErrorAnalyzer` — classifies errors into confusion patterns (pos↔neg, neutral confusion, etc.)

---

### `data/data_layer/preprocess_and_split.py`
**Purpose**: Text cleaning and stratified train/val/test split

**Key functions**:
- `clean_text()` — HTML removal, URL removal, garbled token detection, Unicode normalization
- `perform_stratified_split()` — **two-phase split**:
  1. Reserve all rare-class rows, split proportionally to val/test
  2. Standard stratified split on remaining rows
- `analyze_class_distribution()` — prints and logs imbalance ratios per aspect
- `identify_imbalanced_classes()` — flags aspects above `threshold` imbalance ratio

**Output**: `data/splits/train.csv`, `val.csv`, `test.csv`

---

### `data/data_layer/create_train_aug.py`
**Purpose**: Merge LLM synthetic data into training set

**Process**:
1. Loads `data/splits/train.csv` (original training split)
2. Loads `data/augmented/*.csv` (LLM-generated synthetic reviews)
3. Keeps only columns common to both
4. Concatenates → saves `data/splits/train_augmented.csv`
5. Generates `augmentation_impact.md` with before/after class distribution analysis

---

### `outputs/cosmetic_sentiment_v1/evaluation/inference.py`
**Purpose**: Use trained model for prediction and explanation

**Class `SentimentPredictor`**:

| Method | Description |
|--------|-------------|
| `predict(text, aspect)` | Returns sentiment, confidence, probabilities, optional attention |
| `predict_all_aspects(text)` | Runs all 7 aspects |
| `visualize_attention(text, aspect)` | Plots attention heatmap |
| `explain_with_lime(text, aspect)` | LIME word contributions |
| `visualize_lime(text, aspect)` | LIME bar chart |
| `explain_with_shap(text, aspect)` | SHAP token attributions |
| `explain_with_integrated_gradients(text, aspect)` | Captum IG — completeness axiom |
| `explain_msr_delta(text, focus_aspect)` | MSR: per-token confidence delta + cross-aspect summary |

**CLI usage**:
```bash
python inference.py --checkpoint best_model.pt \
    --text "Love the colour, smell is awful" --aspect colour \
    --explain all --save-path output.png
```

---

### `src/experiments/` package
**Purpose**: Structured ablation studies and baseline comparisons

| File | Contents |
|------|----------|
| `baseline_models.py` | `PlainRoBERTa`, `CrossEntropyLossWrapper`, `BERTBaseline`, `TFIDFSVMBaseline` |
| `ablation_configs.py` | `get_all_ablation_specs()`, `get_all_baseline_specs()`, `print_experiment_plan()` |
| `experiment_runner.py` | CLI: `--list`, `--experiment <id>`, `--group <baselines|ablations|all>` |
| `results_analyzer.py` | Generates `experiment_report.md`, LaTeX `.tex`, bar chart PNGs |

---

### `tests/` directory
**Purpose**: Verify model and integration correctness

| File | Requires checkpoint? | Tests |
|------|---------------------|-------|
| `test_model_components.py` | ❌ No | Architecture, loss functions, ablation flags, parameter count |
| `comprehensive_test.py` | ✅ Yes | Predictions on 10 reviews, mixed sentiment, attention XAI, optional IG/MSR |
| `test_integration.py` | ✅ Yes | Website adapter pipeline, format correctness, timing |

**Run before training:**
```bash
python tests/test_model_components.py
```

---

## 🔄 Typical Workflow

```
1. python data/data_layer/preprocess_and_split.py      # Clean + split
2. python data/data_layer/create_train_aug.py           # Add synthetic data
3. python tests/test_model_components.py               # Verify architecture
4. python src/models/train.py                          # Train main model
5. python tests/comprehensive_test.py --xai            # Test on real reviews
6. python src/experiments/experiment_runner.py --group baselines   # Baselines
7. python src/experiments/experiment_runner.py --group ablations   # Ablations
8. python src/experiments/results_analyzer.py          # Generate tables
```

---

## ✅ Pre-Submission Checklist

- [ ] `test_model_components.py` passes all checks
- [ ] Main model trained and `best_model.pt` saved
- [ ] All 19 experiment results in `results/experiments/all_results.json`
- [ ] `results_analyzer.py` run → LaTeX tables ready for thesis
- [ ] XAI charts generated for at least 3 representative reviews (one strongly pos, one strongly neg, one mixed)
- [ ] `augmentation_impact.md` generated and reviewed
- [ ] Confusion matrices saved for all 7 aspects
- [ ] MixedSentimentEvaluator results included in test report
