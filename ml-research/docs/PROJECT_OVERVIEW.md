# Project Files Overview — ClearView

Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain

---

## File structure

```text
ml-research/
│
├── README.md                          # Setup and notebook run order
├── requirements.txt                   # Python dependencies
│
├── configs/
│   └── config.yaml                    # All hyperparameters, paths, loss config
│
├── data/
│   ├── raw/                           # Original annotated CSV files
│   ├── augmented/                     # LLM-generated synthetic samples
│   └── splits/
│       ├── train_augmented.csv        # 10,050 samples (original + synthetic)
│       ├── val.csv                    # ~810 stratified samples
│       └── test.csv                   # ~810 stratified samples
│
├── docs/                              # Methodology notes, analysis reports
│
├── notebooks/                         # All 21 notebooks — run in order 01 → 21
│
├── src/
│   ├── models/
│   │   ├── model.py                   # Full model architecture
│   │   └── losses.py                  # Loss functions for imbalance
│   ├── experiments/
│   │   ├── ablation_configs.py        # Ablation config generators
│   │   └── baseline_models.py         # Baseline model classes
│   └── utils/
│       ├── data_utils.py              # Dataset, DataLoader, DependencyParser
│       └── metrics.py                 # Evaluator classes
│
├── inference_bridge/                  # Website backend imports these at runtime
│   ├── trained_model_adapter.py       # Prediction API for the website
│   └── trained_model_xai.py          # XAI API for the website
│
└── outputs/
    ├── cosmetic_sentiment_v1/
    │   ├── best_model.pt              # Trained checkpoint (~505 MB)
    │   └── evaluation/
    │       ├── inference.py           # SentimentPredictor
    │       ├── metrics.json           # Test-set results
    │       ├── predictions.csv        # Per-sample predictions
    │       └── all_confusion_matrices.png
    └── experiments/
        └── analysis/                  # Charts + LaTeX tables + experiment report
```

---

## Key files

### `configs/config.yaml`

Single source of truth for all settings. Sections: `model` (architecture flags), `training` (batch size, lr, loss weights, focal_gamma per aspect), `data` (paths, max_seq_length), `aspects` (7 aspect names), `training.class_counts` (used by loss functions).

Edit this to change batch size for memory, enable/disable GCN or aspect attention for ablation, or tune focal_gamma for a specific aspect.

### `src/models/model.py`

Three classes:

- `AspectAwareRoBERTa` — RoBERTa-base + 8-head MHA with learnable aspect queries + per-aspect classifiers. Set `use_aspect_attention=False` for CLS pooling (A2), `use_shared_classifier=True` for single head (A5).
- `AspectOrientedDepGCN` — 2-layer GCN with aspect-gated message passing on dependency edges
- `MultiAspectSentimentModel` — puts everything together, reads flags from config
- `create_model(config)` — factory function

### `src/models/losses.py`

- `FocalLoss` — focuses on hard minority examples, γ controls intensity
- `ClassBalancedLoss` — reweights by effective sample count, β controls tightness
- `HybridLoss` — combines all three
- `AspectSpecificLossManager` — auto-configures one HybridLoss per aspect from class counts in config. price and packing get γ=3.0, β=0.9999; smell gets γ=2.5; others γ=2.0.

### `src/utils/data_utils.py`

- `CosmeticReviewDataset` — reads CSV, tokenizes, optionally runs spaCy dependency parsing, returns edge_index for GCN
- `collate_fn_with_dependencies` — batches samples + pads edge_index
- `create_dataloaders` — train/val/test DataLoader factory
- `compute_class_weights` — inverse-frequency class weights tensor

### `src/utils/metrics.py`

- `AspectSentimentEvaluator` — per-aspect accuracy, Macro-F1, Weighted-F1, MCC, confusion matrices, LaTeX table output
- `MixedSentimentEvaluator` — groups by review_id, identifies conflicting-aspect reviews, measures resolution accuracy
- `ErrorAnalyzer` — classifies error patterns (pos↔neg confusion, neutral confusion, etc.)

### `inference_bridge/inference.py`

`SentimentPredictor` class — wraps the trained model for prediction and all XAI methods:

| Method | Description |
| --- | --- |
| `predict(text, aspect)` | Sentiment + confidence + probabilities |
| `predict_all_aspects(text)` | All 7 aspects at once |
| `explain_with_lime(text, aspect)` | Word contribution scores |
| `explain_with_shap(text, aspect)` | SHAP token attributions |
| `explain_with_integrated_gradients(text, aspect)` | Captum IG — completeness axiom |

### `inference_bridge/`

The website backend imports `trained_model_adapter.py` and `trained_model_xai.py` from here. Do not move or rename these — the import paths are hardcoded in `website/backend/model_cache.py`. They wrap `SentimentPredictor` and expose a clean JSON interface.

### `src/experiments/`

| File | Contents |
| --- | --- |
| `baseline_models.py` | PlainRoBERTa, CrossEntropyLossWrapper, BERTBaseline, TFIDFSVMBaseline |
| `ablation_configs.py` | Config generators for all ablation variants |

Experiments are run interactively via `notebooks/12_experiment_runner.ipynb` and results are analysed via `notebooks/13_results_analyzer.ipynb`.

---

## Typical workflow

```text
1. notebooks/02_preprocess_and_split   → clean + split data
2. notebooks/03_create_train_aug       → add synthetic data
3. notebooks/18_test_model_components  → verify architecture (no checkpoint needed)
4. notebooks/09_train                  → train main model
5. notebooks/14_inference              → test predictions + XAI
6. notebooks/12_experiment_runner      → run baselines + ablations
7. notebooks/13_results_analyzer       → generate tables + charts
```
