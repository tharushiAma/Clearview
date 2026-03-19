# Experiment Results Report

Generated from 22 experiments

---

# Baseline Comparisons

## Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Flat ABSA RoBERTa — aspect attention, shared head, CE loss (no GCN/hybrid loss) | 0.9214 | **0.7948** | 0.9227 | 0.7846 | 0.51 min |
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.7754 | **0.5731** | 0.7827 | 0.4235 | 0.52 min |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 0.62 min |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.7566 | **0.5697** | 0.7713 | 0.4398 | 0.54 min |
| Classical TF-IDF + LinearSVC — no deep learning | 0.8997 | **0.6971** | 0.8880 | 0.7023 | 0.05 min |


## Baseline Comparison — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
|---|---|---|---|---|---|---|---|---|
| Flat ABSA RoBERTa — aspect attention, shared head, CE loss (no GCN/hybrid loss) | 0.8304 | 0.8089 | 0.7716 | 0.3267 | 0.7335 | 0.7923 | 0.6139 | 0.6968 |
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.6086 | 0.5626 | 0.5212 | 0.3378 | 0.5296 | 0.6300 | 0.4627 | 0.5218 |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.8338 | 0.7988 | 0.7295 | 0.4223 | 0.7689 | 0.7819 | 0.7077 | 0.7204 |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.5759 | 0.5882 | 0.5164 | 0.3131 | 0.5213 | 0.6432 | 0.4218 | 0.5114 |
| Classical TF-IDF + LinearSVC — no deep learning | 0.7185 | 0.7131 | 0.6531 | 0.3751 | 0.5901 | 0.7103 | 0.5430 | 0.6147 |


## Baseline Comparison — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| Flat ABSA RoBERTa — aspect attention, shared head, CE loss (no GCN/hybrid loss) | 0.0000 | 0.0000 | 0.0000 | 0.4444 |
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.0580 | 0.0588 | 0.0000 | 0.1579 |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.2857 | 0.0000 | 0.2857 | 0.3636 |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.0762 | 0.0000 | 0.0000 | 0.2000 |
| Classical TF-IDF + LinearSVC — no deep learning | 0.0000 | 0.1429 | 0.0000 | 0.2353 |

---
# Ablation Studies


### Ablation 1: GCN Component

## Ablation 1: GCN Component

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Full model (with Dependency GCN) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 0.61 min |
| No GCN — aspect attention only | 0.8802 | **0.6863** | 0.8794 | 0.6701 | 0.52 min |


## Ablation 1: GCN Component — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| Full model (with Dependency GCN) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| No GCN — aspect attention only | 0.2222 | 0.1333 | 0.0000 | 0.3333 |

### Ablation 2: Aspect Attention

## Ablation 2: Aspect Attention

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Aspect-guided MHA attention | 0.9234 | **0.7904** | 0.9229 | 0.7847 | 0.61 min |
| CLS token pooling (no attention) | 0.7317 | **0.5378** | 0.7507 | 0.4376 | 0.6 min |


## Ablation 2: Aspect Attention — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| Aspect-guided MHA attention | 0.5714 | 0.2500 | 0.0000 | 0.3636 |
| CLS token pooling (no attention) | 0.0508 | 0.1250 | 0.0000 | 0.3448 |

### Ablation 3: Loss Function

## Ablation 3: Loss Function

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Hybrid Loss (Focal + CB + Dice) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 0.61 min |
| Focal Loss only | 0.9178 | **0.7725** | 0.9166 | 0.7731 | 0.6 min |
| Class-Balanced Loss only | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 0.6 min |
| Dice Loss only | 0.7823 | **0.2926** | 0.6868 | 0.0000 | 44.51 min |
| Cross-Entropy Loss (no imbalance handling) | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 93.48 min |


## Ablation 3: Loss Function — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| Hybrid Loss (Focal + CB + Dice) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Focal Loss only | 0.2000 | 0.2353 | 0.0000 | 0.4211 |
| Class-Balanced Loss only | 0.2857 | 0.0000 | 0.2857 | 0.3636 |
| Dice Loss only | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Cross-Entropy Loss (no imbalance handling) | 0.2857 | 0.0000 | 0.2857 | 0.3636 |

### Ablation 4: Data Augmentation

## Ablation 4: Data Augmentation

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| With LLM synthetic augmentation (10,050 samples) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 103.01 min |
| Without augmentation (9,240 samples) | 0.9245 | **0.7872** | 0.9225 | 0.7874 | 68.82 min |


## Ablation 4: Data Augmentation — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| With LLM synthetic augmentation (10,050 samples) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Without augmentation (9,240 samples) | 0.0000 | 0.0000 | 0.0000 | 0.3810 |

### Ablation 5: Classifier Head

## Ablation 5: Classifier Head

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| 7 aspect-specific classifier heads | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 103.0 min |
| Single shared classifier head | 0.9205 | **0.7797** | 0.9194 | 0.7778 | 88.87 min |


## Ablation 5: Classifier Head — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| 7 aspect-specific classifier heads | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Single shared classifier head | 0.2000 | 0.1905 | 0.0000 | 0.4000 |

### Ablation 6: Text Preprocessing

## Ablation 6: Text Preprocessing

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| MSR Eval: Full model + GCN (mixed sent resolution) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 102.07 min |
| MSR Eval: No GCN (attention only, no dep parsing) | 0.9212 | **0.7877** | 0.9212 | 0.7799 | 109.12 min |


## Ablation 6: Text Preprocessing — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
|---|---|---|---|---|
| MSR Eval: Full model + GCN (mixed sent resolution) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| MSR Eval: No GCN (attention only, no dep parsing) | 0.3333 | 0.1429 | 0.0000 | 0.4615 |


## Ablation 6: Text Preprocessing — Mixed Sentiment Resolution Metrics

| Experiment | MSR Review Acc (%) | MSR Aspect Acc (%) | Mixed Detection Rate (%) | Mixed Review Count |
|---|---|---|---|---|
| MSR Eval: Full model + GCN (mixed sent resolution) | **66.6%** | 87.2% | 43.4% | 628 |
| MSR Eval: No GCN (attention only, no dep parsing) | **66.6%** | 87.3% | 43.4% | 628 |