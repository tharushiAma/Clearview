# Methodology: Multi-Aspect Mixed Sentiment Resolution with Class Imbalance Handling and Explainability

> *ClearView Final Year Project — Academic Methodology Reference*

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Problem Statement](#2-problem-statement)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Class Imbalance Handling](#5-class-imbalance-handling)
6. [Explainability Integration](#6-explainability-integration)
7. [Training Strategy](#7-training-strategy)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Ablation Studies & Baselines](#9-ablation-studies--baselines)
10. [How to Run](#10-how-to-run)
11. [References](#11-references)

---

## 1. Research Overview

This project implements a **Multi-Aspect Mixed Sentiment Resolution (MAMSR)** system for the cosmetic domain. It is specifically designed to handle:

- **Multi-aspect sentiment classification** across 7 aspects: *stayingpower, texture, smell, price, colour, shipping, packing*
- **Severe class imbalance** with pre-augmentation ratios up to 185:1 (positive:negative)
- **Mixed sentiment resolution** — correctly separating conflicting sentiments for different aspects within the same review
- **Multi-level explainability** through attention visualization, LIME, SHAP, Integrated Gradients, and MSR Delta analysis

The system combines RoBERTa-base (contextualized embeddings) with Aspect-Aware Multi-Head Attention, an Aspect-Oriented Dependency GCN, and a Hybrid Loss function (Focal + Class-Balanced + Dice), with LLM-based synthetic data augmentation for minority class enrichment.

---

## 2. Problem Statement

### Challenges Addressed

1. **Multi-Aspect Sentiment Analysis**
   - Reviews contain sentiments about multiple aspects simultaneously
   - Each aspect may have a different polarity
   - Example: *"Beautiful colour but terrible smell"* → colour: positive, smell: negative

2. **Extreme Class Imbalance** (before augmentation)

   | Aspect | Negative | Neutral | Positive | Pos:Neg Ratio |
   |--------|----------|---------|----------|---------------|
   | Price | 17 | 15 | 2,244 | 132:1 |
   | Packing | 29 | 6 | 2,070 | 71:1 |
   | Smell | 51 | 17 | 872 | 17:1 |
   | Colour | 131 | 113 | 1,506 | 11:1 |

   Traditional cross-entropy loss leads to high overall accuracy but near-zero minority class recall.

3. **Mixed Sentiment Resolution**
   - Sentences with contradictory sentiments for different aspects require separating which opinion words attach to which aspect
   - Syntactic dependency structure provides the key signal

4. **Explainability Requirements**
   - Model decisions must be interpretable for trust, debugging, and academic credibility
   - Multiple explanation methods provide complementary views

---

## 3. Data Pipeline

### 3.1 Text Cleaning (`data/data_layer/preprocess_and_split.py`)

Applied to all splits at preprocessing time and also replicated at inference time (`inference.py: clean_text_for_inference`) to ensure train-inference parity:

1. Unicode normalization (NFC)
2. HTML entity and tag removal
3. URL and email removal
4. Repeated punctuation normalization (`!!!` → `!`)
5. Garbled token detection (high consonant ratio or character repetition)
6. Invisible/zero-width character removal

### 3.2 Stratified Split — Two-Phase Rare-Class Guarantee

Standard `train_test_split` with stratification over a composite label fails for aspects with very few minority samples (e.g. price-negative: 17 samples total). The solution is a two-phase split:

**Phase 1**: Identify rare-class rows (below `min_val_test_samples` threshold). Split them proportionally into val/test first.

**Phase 2**: Perform standard stratified split on the remaining majority-class rows.

This guarantees a minimum number of minority-class samples in val/test for reliable evaluation.

**Output**: `data/splits/train.csv`, `val.csv`, `test.csv`

### 3.3 Synthetic Augmentation (`data/data_layer/create_train_aug.py`)

LLM-generated reviews target the most imbalanced minority classes:

| Aspect | Before ratio | After ratio |
|--------|-------------|-------------|
| Price | 132:1 | ~11:1 |
| Packing | 71:1 | ~12:1 |
| Smell | 17:1 | ~6:1 |

**Output**: `data/splits/train_augmented.csv` (10,050 samples total) + `augmentation_impact.md` (before/after distribution report)

---

## 4. Model Architecture

### 4.1 Overview

```
Input Text
    │
    ▼
RoBERTa-base Encoder
(12 layers, 768-dim, 125M params)
    │  last_hidden_state: (batch, seq_len, 768)
    ▼
Aspect-Aware Attention Module
─ learnable aspect embeddings (7 × 768, xavier init)
─ aspect embedding is query, token states are K/V
─ 8-head Multi-Head Attention
→ aspect_representation: (batch, 768)
    │
    ▼
Aspect-Oriented Dependency GCN   [optional; use_dependency_gcn flag]
─ 2-layer GCN on spaCy dependency parse trees
─ Aspect-oriented gate: σ(W_g × aspect_emb)
─ Message passing: gated aggregation along dependency edges
─ Residual connections
→ gcn_output: (batch, 768)
    │
    ▼
Fusion: aspect_repr + gcn_output (residual if GCN enabled)
    │
    ▼
7 Aspect-Specific Classifiers   [or 1 shared; use_shared_classifier flag]
─ Linear(768 → 384) → ReLU → Dropout(0.1) → Linear(384 → 3)
    │
    ▼
Logits → Sentiment {Negative, Neutral, Positive}
```

**Total trainable parameters**: ~132M (with GCN), ~129M (without GCN)

### 4.2 RoBERTa Encoder

- Pre-trained `roberta-base` (HuggingFace)
- Full fine-tuning (all 12 layers updated)
- Input: BPE tokenized text, max 128 tokens
- Output: `last_hidden_state` of shape `(batch, seq_len, 768)`

**Why RoBERTa over BERT?**
- Trained longer on more data (160GB vs 16GB)
- No Next Sentence Prediction objective (less noise)
- Consistently outperforms BERT on sentiment tasks

### 4.3 Aspect-Aware Attention Module (`AspectAwareRoBERTa`)

**Key innovation**: Instead of using [CLS] token for classification, we use a learnable aspect embedding as the **query** in Multi-Head Attention. This forces the model to retrieve only the information relevant to the queried aspect.

```python
aspect_query = self.aspect_embeddings(aspect_id)   # (batch, 768)
aspect_query = aspect_query.unsqueeze(1)            # (batch, 1, 768)

attended, attn_weights = self.aspect_attention(
    query=aspect_query,
    key=hidden_states,
    value=hidden_states,
    key_padding_mask=~attention_mask.bool()
)
aspect_repr = attended.squeeze(1)  # (batch, 768)
```

**Benefits**:
- Learns aspect-specific token importance
- Outputs interpretable attention weights for attention-based XAI
- Reduces noise from irrelevant parts of the review

**Ablation A2**: Replacing this with simple [CLS] pooling (`use_aspect_attention=False`) measures this component's isolated contribution.

### 4.4 Aspect-Oriented Dependency GCN (`AspectOrientedDepGCN`)

**Purpose**: Capture syntactic relationships to resolve mixed sentiments.

**Dependency parsing**: spaCy `en_core_web_sm` extracts dependency edges at dataset creation time (`utils/data_utils.py: DependencyParser`). Edge indices are stored per sample and batched by `collate_fn_with_dependencies`.

**GCN message passing** (2 layers):
```
gate = σ(W_gate × aspect_embedding)       # aspect-specific filter
for each edge (src → dst):
    messages[dst] += gate ⊙ H[src]        # gated message
H_new = ReLU(messages) + H               # residual update
```

The gate controls how much each other token's representation is allowed to flow based on its relevance to the target aspect.

**Why GCN?**
- Long-range syntactic signals that attention can miss
- "cheap price but awful smell" — "awful" is syntactically linked to "smell" not "price"
- Aspect-oriented gate prevents cross-aspect interference

**Ablation A1**: Training without GCN (`use_dependency_gcn=False`) isolates this component's contribution.

### 4.5 Classifier Heads

Default: 7 separate per-aspect heads, each `Linear(768→384) → ReLU → Dropout → Linear(384→3)`.

**Rationale**: Different aspects may have different linguistic patterns. A shared head would force the same decision boundary for "smells amazing" (smell) and "delivers fast" (shipping).

**Ablation A5**: Using a single shared head (`use_shared_classifier=True`) tests this assumption.

---

## 5. Class Imbalance Handling

### 5.1 Three-Pronged Strategy

```
Synthetic Augmentation → reduces ratio before training
         +
Hybrid Loss Function → corrects remaining imbalance during training
         +
Two-Phase Stratified Split → ensures minority classes in val/test
```

### 5.2 Focal Loss

**Paper**: Lin et al., ICCV 2017

**Formula**: `FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)`

- `(1 − p_t)^γ`: down-weights easy examples already predicted correctly
- `α_t`: per-class weight (set from class counts by `AspectSpecificLossManager`)
- `γ`: focusing parameter — higher γ = more emphasis on hard examples

**Per-aspect γ configuration**:
```yaml
focal_gamma:
  default: 2.0
  smell: 2.5
  price: 3.0
  packing: 3.0
```

### 5.3 Class-Balanced Loss

**Paper**: Cui et al., CVPR 2019

**Formula**: `w_c = (1 − β) / (1 − β^{n_c})`

- `n_c`: sample count for class c  
- `β`: smoothing parameter (0.999 → moderate; 0.9999 → extreme)

The effective number of samples diminishes marginally with additional samples, making this more principled than inverse-frequency weighting.

**Per-aspect β**:
```yaml
class_balanced_beta:
  default: 0.999
  price: 0.9999
  packing: 0.9999
```

### 5.4 Dice Loss

**Paper**: Li et al., ACL 2020

**Formula**: `DL = 1 − (2 · |P ∩ T| + ε) / (|P| + |T| + ε)`

Directly optimizes the Dice coefficient (equivalent to F1-score). Complements cross-entropy-based losses by targeting the evaluation metric directly.

### 5.5 Hybrid Loss

```python
total_loss = (1.0 × focal_loss) + (0.5 × cb_loss) + (0.3 × dice_loss)
```

Each aspect gets its own `HybridLoss` instance auto-configured from `config.yaml: training.class_counts` by `AspectSpecificLossManager`.

**Ablation A3** tests: Hybrid vs. Focal-only vs. CB-only vs. Dice-only vs. plain CE.

---

## 6. Explainability Integration

### 6.1 Motivation

Mixed sentiment resolution is a novel claim. Explainability provides:
1. **Evidence** that the model separates aspects correctly
2. **Trust** for end users of the website
3. **Interpretability** required for academic credibility

### 6.2 Attention Visualization

**Method**: Extract MHA attention weights from `AspectAwareRoBERTa.aspect_attention`.

**Key property**: The query is the aspect embedding, so attention weights directly show which tokens the model uses to form its aspect-specific sentiment.

```python
result = predictor.predict(text, aspect, return_attention=True)
tokens  = result['attention']['tokens']
weights = result['attention']['weights']  # (seq_len,)
```

**Limitation**: Attention ≠ feature importance. Use IG for rigorous attribution.

### 6.3 LIME

**Paper**: Ribeiro et al., KDD 2016

**Method**: Randomly mask words and observe prediction change → fit local linear model → report feature importances.

**Implementation** (`inference.py: explain_with_lime`):
- Perturbs text by removing words
- Calls `SentimentPredictor.predict()` on each perturb (same code path as normal prediction)
- LIME `LimeTextExplainer` fits a linear model locally

### 6.4 SHAP

**Method**: Shapley values from cooperative game theory. Each token's contribution is its average marginal contribution across all possible coalitions.

**Implementation** (`inference.py: explain_with_shap`):
- Uses `shap.Explainer` with Partition algorithm
- Baseline: randomly masked versions of the input
- Reports per-token SHAP values for the predicted class

### 6.5 Integrated Gradients *(Most Theoretically Rigorous)*

**Paper**: Sundararajan et al., ICML 2017

**Formula**: `IG(x) = (x − x') · ∫₀¹ ∂F(x' + α(x − x'))/∂x dα`

- `x'`: baseline (all-PAD embedding)
- Satisfies **completeness axiom**: attribution scores sum exactly to the output change
- Implemented via **Captum's `LayerIntegratedGradients`** on RoBERTa's embedding layer
- n_steps=50 interpolation steps by default

**Why IG > attention weights for a research paper**:
- Theoretically grounded with formal axioms (completeness, sensitivity, implementation invariance)
- Attributions are verified by convergence delta (should be near 0)
- Works on the actual forward pass, not a proxy

**Implementation** (`inference.py: explain_with_integrated_gradients`):
- Hooks into `roberta.embeddings` layer
- Forward function routes through aspect attention + classifier
- Sums per-token attributions across embedding dimensions
- Normalizes to [-1, 1] for visualization

### 6.6 MSR Delta *(Unique Research Contribution)*

**Purpose**: Directly prove that the model performs Mixed Sentiment Resolution — i.e., it separates aspect-specific signals from a review containing conflicting sentiments.

**Method**:
1. Predict baseline confidence for `focus_aspect` on the full review
2. For each token, replace it with `[MASK]` and re-predict
3. `delta[i] = baseline_conf − masked_conf`
   - Large positive delta → token actively supports the `focus_aspect` prediction
   - Near-zero delta → token is irrelevant to `focus_aspect` (even if it expresses opinion about another aspect)

**Research significance**: For a review like *"Great colour but the smell is awful"*:
- When `focus_aspect = colour`: tokens "great" and "colour" should have high +delta; "awful" and "smell" should have ~0 delta
- This proves the model is NOT conflating aspects — it is resolving them separately

**Cross-aspect summary** is printed automatically: shows the model's prediction for all OTHER aspects simultaneously, demonstrating it understands the multi-aspect nature of the review.

---

## 7. Training Strategy

### 7.1 Optimizer and Schedule

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

- Low LR (2e-5) required for fine-tuning pre-trained RoBERTa without catastrophic forgetting
- Warmup prevents early divergence of the LM head
- Weight decay provides L2 regularization

### 7.2 Mixed Precision (AMP)

```python
with torch.cuda.amp.autocast():
    logits = model(input_ids, attention_mask, aspect_id, edge_index=edge_idx)
    loss, _ = loss_manager.compute_loss(logits, labels, aspect_ids, aspect_names)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

- ~2× speedup on RTX 4060
- Gradient clipping at 1.0 prevents exploding gradients

### 7.3 Early Stopping (Bug-Fixed)

**Important fix**: Early stopping is evaluated only at **epoch end**. Mid-epoch evaluations log metrics only and do NOT update `patience_counter` or `best_val_metric`. This prevents premature stopping due to noisy mid-epoch checkpoints.

- **Metric**: Validation Macro-F1 (imbalance-robust)
- **Patience**: 5 epochs
- Best model saved as `outputs/cosmetic_sentiment_v1/best_model.pt`

### 7.4 MixedSentimentEvaluator

Integrated into the test phase (`train.py`). After standard per-aspect metrics, the evaluator:
1. Groups test predictions by `review_id`
2. Identifies reviews where ≥2 aspects have conflicting predicted sentiments
3. Measures whether the conflicts were resolved correctly vs. ground truth
4. Reports a Mixed Sentiment Resolution Accuracy score

### 7.5 Training Configuration Summary

| Parameter | Value |
|-----------|-------|
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

## 8. Evaluation Metrics

### 8.1 Primary Metric: Macro-F1

Treats all classes equally regardless of support. Critical for imbalanced datasets where accuracy is misleading.

`Macro-F1 = mean(F1_negative, F1_neutral, F1_positive)`

### 8.2 Complementary Metrics

| Metric | Purpose |
|--------|---------|
| Per-class F1 | Shows performance on each minority class separately |
| Weighted F1 | F1 weighted by class support; shows aggregate performance |
| MCC | Matthews Correlation Coefficient: balanced even for extreme imbalance; range [-1, 1] |
| Accuracy | Reported but secondary given class imbalance |
| Confusion Matrix | Visual breakdown of prediction errors per aspect |

### 8.3 Mixed Sentiment Resolution Accuracy

Computed by `MixedSentimentEvaluator` — percentage of "conflicting reviews" where the model correctly identifies the different sentiments for each aspect.

### 8.4 Implementation

```python
from utils.metrics import AspectSentimentEvaluator, MixedSentimentEvaluator

evaluator = AspectSentimentEvaluator()
evaluator.add_batch(y_pred, y_true, aspect_names)
results = evaluator.compute()  # {accuracy, macro_f1, weighted_f1, mcc, per_class, confusion_matrix}
```

---

## 9. Ablation Studies & Baselines

### 9.1 Ablation Studies (6 studies, 15 variants)

| ID | Component | Conditions |
|----|-----------|-----------|
| A1 | Dependency GCN | Full model vs. No GCN |
| A2 | Aspect Attention | MHA attention vs. [CLS] pooling |
| A3 | Loss Function | Hybrid / Focal-only / CB-only / Dice-only / plain CE |
| A4 | Data Augmentation | With synthetic data vs. original only |
| A5 | Classifier Head | 7 aspect-specific heads vs. 1 shared head |
| A6 | Text Preprocessing | With cleaning pipeline vs. raw text |

### 9.2 Baseline Comparisons (4 models)

| ID | Model | Description |
|----|-------|-------------|
| B1 | PlainRoBERTa | RoBERTa + [CLS] head, no aspect awareness, CE loss |
| B2 | RoBERTa+CE | Full architecture but CrossEntropy only (no hybrid loss) |
| B3 | BERTBaseline | BERT-base-uncased + [CLS] head, CE loss |
| B4 | TF-IDF+SVM | Classical: TF-IDF features + LinearSVC per aspect |

### 9.3 Running Experiments

```bash
# See all 19 experiments
python src/experiments/experiment_runner.py --list

# Run all baselines
python src/experiments/experiment_runner.py --group baselines

# Run one ablation
python src/experiments/experiment_runner.py --experiment A3_focal_only

# Analyze results → Markdown + LaTeX + charts
python src/experiments/results_analyzer.py
```

---

## 10. How to Run

### Environment Setup

```powershell
cd "c:\Users\TharushiAmasha\OneDrive - inivosglobal.com\FYP\Clearview\ml-research"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Verify GPU

```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Data Preparation

```powershell
python data/data_layer/preprocess_and_split.py   # clean + split
python data/data_layer/create_train_aug.py        # add synthetic data
```

### Test Architecture (no checkpoint)

```powershell
python tests/test_model_components.py
```

### Train

```powershell
python src/models/train.py --config configs/config.yaml

# Resume
python src/models/train.py --config configs/config.yaml `
    --resume outputs/cosmetic_sentiment_v1/best_model.pt
```

### Inference and XAI

```powershell
# Interactive mode
python outputs/cosmetic_sentiment_v1/evaluation/inference.py `
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt

# All XAI methods
python outputs/cosmetic_sentiment_v1/evaluation/inference.py `
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt `
    --text "Love the colour but smell is awful" `
    --aspect colour --explain all --save-path results/xai.png

# MSR Delta (demonstrates mixed sentiment resolution)
python outputs/cosmetic_sentiment_v1/evaluation/inference.py `
    --checkpoint outputs/cosmetic_sentiment_v1/best_model.pt `
    --text "Love the colour but smell is awful" `
    --aspect colour --explain msr
```

### Monitoring Training

The console shows per-batch loss and per-epoch metrics. Results saved to `outputs/cosmetic_sentiment_v1/`:
- `best_model.pt` — best checkpoint by val Macro-F1
- `training_log.json` — per-epoch metrics
- `evaluation/` — test set metrics, confusion matrices, LaTeX tables

---

## 11. References

1. **RoBERTa**: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019
2. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
3. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
4. **Dice Loss for NLP**: Li et al., "Dice Loss for Data-imbalanced NLP Tasks", ACL 2020
5. **LIME**: Ribeiro et al., "Why Should I Trust You? Explaining the Predictions of Any Classifier", KDD 2016
6. **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
7. **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
8. **Captum**: Kokhlikyan et al., "Captum: A unified and generic model interpretability library for PyTorch", 2020

---

*ClearView FYP — Tharushi Amasha — 2025*
