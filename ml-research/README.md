# ClearView — Multi-Aspect Mixed Sentiment Analysis with Explainability

> **Final Year Project** | BEng Software Engineering
> *Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain*

A research implementation combining **RoBERTa**, **Aspect-Aware Attention**, **Dependency GCN**, and advanced class-imbalance handling (Hybrid Loss + LLM-based synthetic augmentation) to analyze sentiment across 7 aspects of cosmetic product reviews — with full multi-level explainability.

---

## Project Structure

```text
ml-research/
│
├── README.md                          # This file
├── requirements.txt                   # All Python dependencies
├── setup.sh                           # One-command environment setup (Linux/Mac)
├── configs/
│   └── config.yaml                    # All hyperparameters, loss config, aspect names
│
├── data/
│   ├── raw/                           # Original Vietnamese + translated English CSVs
│   ├── augmented/                     # LLM-generated synthetic samples (minority classes)
│   └── splits/                        # train_augmented.csv, val.csv, test.csv
│
├── docs/                              # Methodology notes, ablation analysis reports
│
├── notebooks/                         # All 21 Jupyter notebooks — run in order 01 → 21
│   ├── 01_google_translator_local     # Translate Vietnamese reviews → English
│   ├── 02_preprocess_and_split        # Clean data + stratified train/val/test split
│   ├── 03_create_train_aug            # Merge LLM synthetic data into training set
│   ├── 04_convert_synthetic_robust    # Validate + normalise synthetic samples
│   ├── 05_data_utils                  # Dataset class, DataLoaders, dependency parsing
│   ├── 06_metrics                     # Evaluators: AspectSentiment, MixedSentiment, Error
│   ├── 07_losses                      # FocalLoss, HybridLoss, AspectSpecificLossManager
│   ├── 08_model                       # Full model architecture: RoBERTa + GCN + heads
│   ├── 09_train                       # Training loop, early stopping, mixed precision
│   ├── 10_ablation_configs            # Ablation study configuration generators
│   ├── 11_baseline_models             # PlainRoBERTa, BERT-base, TF-IDF+SVM baselines
│   ├── 12_experiment_runner           # Run all 22 ablation + baseline experiments
│   ├── 13_results_analyzer            # Generate Markdown + LaTeX + bar charts
│   ├── 14_inference                   # SentimentPredictor: inference + XAI methods
│   ├── 15_evaluate                    # Full test-set evaluation + confusion matrices
│   ├── 16_trained_model_adapter       # Website bridge: TrainedModelAdapter
│   ├── 17_trained_model_xai           # Website XAI bridge: IG, LIME, SHAP
│   ├── 18_test_model_components       # Unit tests — no checkpoint required
│   ├── 19_test_integration            # Integration test: adapter output format
│   ├── 20_comprehensive_test          # Full diagnostic: predictions + XAI
│   └── 21_test_api                    # Live HTTP tests against running backend
│
├── src/
│   ├── models/
│   │   ├── model.py                   # AspectAwareRoBERTa + DepGCN + MultiAspectSentimentModel
│   │   └── losses.py                  # All loss functions
│   ├── experiments/
│   │   ├── ablation_configs.py        # Ablation config generators (imported by notebooks)
│   │   └── baseline_models.py         # Baseline model classes (imported by notebooks)
│   └── utils/
│       ├── data_utils.py              # Dataset, DataLoader, DependencyParser
│       └── metrics.py                 # All evaluator classes
│
├── inference_bridge/                  # Production bridge — website backend imports these at runtime
│   ├── trained_model_adapter.py       # TrainedModelAdapter: prediction API for the website
│   └── trained_model_xai.py          # TrainedModelXAI: XAI API for the website
│
└── outputs/
    ├── cosmetic_sentiment_v1/
    │   ├── best_model.pt              # Trained model checkpoint (~505 MB)
    │   └── evaluation/
    │       ├── inference.py           # SentimentPredictor (imported by inference_bridge)
    │       ├── metrics.json           # Test-set evaluation results
    │       ├── predictions.csv        # Per-sample predictions
    │       ├── error_analysis.csv     # Misclassification breakdown
    │       └── all_confusion_matrices.png
    └── experiments/
        ├── A1_*  A2_*  ...            # Ablation study checkpoints
        ├── B1_*  B2_*  ...            # Baseline model checkpoints
        └── analysis/                  # Charts + LaTeX tables + experiment report
```

---

## About inference_bridge

**Yes — `inference_bridge/` is required and must not be moved or renamed.**

The website backend (`website/backend/model_cache.py`) imports these files directly at runtime:

```python
from inference_bridge.trained_model_adapter import TrainedModelAdapter
from inference_bridge.trained_model_xai import TrainedModelXAI
```

They are the glue between the trained ML model and the web API. They wrap `SentimentPredictor` (defined in `outputs/.../inference.py`) and expose a clean JSON interface the frontend calls. If you move or delete them, the website stops working.

The `.py` files in `src/` are also required — they are imported as Python modules by the notebooks and by `inference.py`. Jupyter notebooks cannot be `import`-ed, so these library files must remain as `.py`.

---

## Setup from Scratch

### Prerequisites

- Python **3.10** or **3.11**
- CUDA-capable GPU recommended (CPU works but training will be very slow)
- At least **8 GB VRAM** for training (16 GB recommended)
- At least **4 GB VRAM** for inference only

---

### Step 1 — Clone and Enter the Project

```bash
git clone <repo-url>
cd Clearview/ml-research
```

---

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note on torch-geometric:** If the above fails, install it manually with the wheel matching your PyTorch + CUDA version:
>
> ```bash
> pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
> ```
>
> Replace `torch-2.5.0+cu121` with your actual version. Check with:
> `python -c "import torch; print(torch.__version__)"`

---

### Step 4 — Launch Jupyter

```bash
jupyter lab notebooks/
```

Or open `ml-research/` in **VS Code** and run notebooks with the Jupyter extension (select your venv as the kernel).

> Each notebook resolves all paths automatically via `ML_RESEARCH = os.path.dirname(os.path.abspath(''))`. They work correctly regardless of where Jupyter is launched from.

---

## Notebook Execution Order

Run notebooks in numerical order. The phases below show what is required vs. optional.

---

### Phase 1 — Data Pipeline (01–04)

> **Skip entirely** if `data/splits/` already has `train_augmented.csv`, `val.csv`, and `test.csv` — they are pre-built in the repo.

| Notebook                      | Skip if...                    | Output                          |
| ----------------------------- | ----------------------------- | ------------------------------- |
| 01 google_translator_local    | English data already exists   | `data/raw/full_data_en.csv`     |
| 02 preprocess_and_split       | Splits already exist          | `data/splits/train/val/test`    |
| 03 create_train_aug           | Augmented split already exists| `data/splits/train_augmented`   |
| 04 convert_synthetic_robust   | Augmented split already exists| Updated `train_augmented.csv`   |

---

### Phase 2 — Understand the Code (05–08)

Read these to understand the architecture. Run the Quick Test cell at the bottom of notebook 08 to verify the model forward pass works.

| Notebook              | What it explains                                      |
| --------------------- | ----------------------------------------------------- |
| 05 data_utils         | How reviews are tokenised, batched, and parsed        |
| 06 metrics            | How accuracy, Macro-F1, and MCC are computed          |
| 07 losses             | FocalLoss, HybridLoss, aspect-specific loss management|
| 08 model              | Full model: RoBERTa + GCN + aspect-specific heads     |

---

### Phase 3 — Train (09)

Open `09_train` and run all cells. Edit `configs/config.yaml` first if needed:

```yaml
training:
  batch_size: 16    # Reduce to 8 if CUDA OOM
  num_epochs: 30
  learning_rate: 2.0e-5

hardware:
  mixed_precision: true
```

The best checkpoint saves to `outputs/cosmetic_sentiment_v1/best_model.pt`.
Training on an RTX 3090 takes approximately **4–6 hours**.

---

### Phase 4 — Ablation and Baseline Experiments (10–13, Optional)

Only needed if you want to reproduce the ablation study results.

| Notebook | Depends on | Output |
| --- | --- | --- |
| 10 ablation_configs | Nothing | Config reference only |
| 11 baseline_models | Nothing | Class reference only |
| 12 experiment_runner | `data/splits/` + GPU | `outputs/experiments/*.pt` |
| 13 results_analyzer | `outputs/experiments/all_results.json` | `outputs/experiments/analysis/` |

---

### Phase 5 — Inference and Evaluation (14–15)

| Notebook      | Depends on                                         |
| ------------- | -------------------------------------------------- |
| 14 inference  | `outputs/cosmetic_sentiment_v1/best_model.pt`      |
| 15 evaluate   | Same checkpoint + `data/splits/test.csv`           |

Run `14_inference` to test predictions and XAI methods interactively. Run `15_evaluate` for the full test-set metrics and confusion matrices.

Example from notebook 14:

```python
predictor = SentimentPredictor('outputs/cosmetic_sentiment_v1/best_model.pt')

result = predictor.predict(
    "I love the colour but the smell is absolutely awful",
    aspect='colour'
)
# {'sentiment': 'positive', 'confidence': 0.94, 'probabilities': {...}}
```

---

### Phase 6 — Website Bridge (16–17, Optional)

These notebooks explain and demonstrate the production bridge code.
The actual bridge files (`inference_bridge/*.py`) are used by the website backend automatically — you do not need to run these notebooks for the website to work.

| Notebook                    | What it shows                                   |
| --------------------------- | ----------------------------------------------- |
| 16 trained_model_adapter    | How the website calls the model for predictions |
| 17 trained_model_xai        | How the website calls IG, LIME, SHAP |

---

### Phase 7 — Tests (18–21)

Run these in order to verify everything works end-to-end.

| Notebook                  | Needs checkpoint | Needs backend running |
| ------------------------- | ---------------- | --------------------- |
| 18 test_model_components  | No               | No                    |
| 19 test_integration       | Yes              | No                    |
| 20 comprehensive_test     | Yes              | No                    |
| 21 test_api               | Yes              | Yes                   |

For notebook 21, start the backend first:

```bash
# From Clearview/website/
python backend/backend_server.py
```

Then run `21_test_api`.

---

## Model Architecture

```text
Input Review Text
      │
      ▼
RoBERTa-base Encoder            (768-dim contextual embeddings)
      │
      ▼
Aspect-Aware Attention          (learnable aspect embeddings × 8-head MHA)
      │                          ablation A2: use_aspect_attention
      ▼
Aspect-Oriented Dependency GCN  (2-layer, aspect-gated message passing)
      │                          ablation A1: use_dependency_gcn
      ▼
7 Aspect-Specific Classifiers   (768 → 384 → 3, one per aspect)
      │                          ablation A5: use_shared_classifier
      ▼
Sentiment: Negative / Neutral / Positive
```

**Total parameters: ~132M** (RoBERTa-base 125M + GCN + classification heads)

---

## Performance Results

| Metric           | Score      |
| ---------------- | ---------- |
| Overall Accuracy | **92.47%** |
| Overall Macro-F1 | **0.7944** |
| Weighted F1      | **0.9236** |
| MCC              | **0.7900** |

**Per-Aspect Macro-F1:**

| Aspect       | Macro-F1 |
| ------------ | -------- |
| Texture      | 0.8088   |
| Shipping     | 0.7975   |
| Stayingpower | 0.7933   |
| Colour       | 0.7647   |
| Smell        | 0.7311   |
| Packing      | 0.5997   |
| Price        | 0.3275   |

---

## Explainability Methods

| Method                  | Description                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------- |
| Attention Visualization | MHA weights over tokens — fast, built-in                                            |
| LIME                    | Local perturbation-based word contributions                                         |
| SHAP                    | Shapley value attributions                                                          |
| Integrated Gradients    | Meets completeness axiom; most rigorous for transformers. Requires `captum`         |


All methods are demonstrated in `notebooks/14_inference` and `notebooks/17_trained_model_xai`.

---

## Ablation Studies

| ID | Study | Variants |
| --- | --- | --- |
| A1 | Dependency GCN | Full model vs. No GCN |
| A2 | Aspect Attention | MHA attention vs. CLS pooling |
| A3 | Loss Function | Hybrid / Focal only / CB only / CE |
| A4 | Data Augmentation | With LLM synthetic / Without |
| A5 | Classifier Head | 7 aspect-specific heads / 1 shared head |
| A6 | Mixed Sentiment Resolution | Full model + GCN vs. No GCN |
| A7 | Hybrid Loss CB Weight | Focal×1.0 + CB×0.5 vs. CB×1.0 |
| B1–B5 | Baselines | PlainRoBERTa / DistilBERTBaseline / BERT-base / TF-IDF+SVM |

---

## Troubleshooting

| Problem | Solution |
| --- | --- |
| CUDA out of memory | Reduce `batch_size` to `8` or `4` in `configs/config.yaml` |
| torch_geometric install fails | Install the CUDA-specific wheel manually (see Step 3) |
| en_core_web_sm not found | Run `python -m spacy download en_core_web_sm` |
| ModuleNotFoundError in notebook | Ensure the `ML_RESEARCH = ...` cell ran first |
| Checkpoint not found | Run `09_train` first, or place checkpoint at `outputs/.../best_model.pt` |
| captum not installed | `pip install captum` — only needed for Integrated Gradients |
| Low recall for Price or Packing | Increase `focal_gamma` for that aspect in `configs/config.yaml` |
| 21_test_api connection refused | Start the backend: `python website/backend/backend_server.py` |

---

## Key Files Reference

| File | Role |
| --- | --- |
| `configs/config.yaml` | Single source of truth for all hyperparameters |
| `src/models/model.py` | Full model class — edit to change architecture |
| `src/models/losses.py` | Loss functions — edit to change loss strategy |
| `src/utils/data_utils.py` | Dataset + DataLoader — edit to change tokenisation |
| `inference_bridge/inference.py` | SentimentPredictor — used by website at runtime |
| `inference_bridge/trained_model_adapter.py` | Website prediction bridge — do not move |
| `inference_bridge/trained_model_xai.py` | Website XAI bridge — do not move |
