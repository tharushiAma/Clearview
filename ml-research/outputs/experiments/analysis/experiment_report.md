# Experiment Results Report

Generated from 22 experiments

---

# Baseline Comparisons

## Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.7754 | **0.5731** | 0.7827 | 0.4235 | 0.52 min |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 0.62 min |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.7566 | **0.5697** | 0.7713 | 0.4398 | 0.54 min |
| Classical TF-IDF + LinearSVC — no deep learning | 0.8997 | **0.6971** | 0.8880 | 0.7023 | 0.05 min |


## Baseline Comparison — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.6086 | 0.5626 | 0.5212 | 0.3378 | 0.5296 | 0.6300 | 0.4627 | 0.5218 |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.8338 | 0.7988 | 0.7295 | 0.4223 | 0.7689 | 0.7819 | 0.7077 | 0.7204 |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.5759 | 0.5882 | 0.5164 | 0.3131 | 0.5213 | 0.6432 | 0.4218 | 0.5114 |
| Classical TF-IDF + LinearSVC — no deep learning | 0.7185 | 0.7131 | 0.6531 | 0.3751 | 0.5901 | 0.7103 | 0.5430 | 0.6147 |


## Baseline Comparison — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| Plain RoBERTa — [CLS] head, no aspect awareness, CE loss | 0.0580 | 0.0588 | 0.0000 | 0.1579 |
| RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss) | 0.2857 | 0.0000 | 0.2857 | 0.3636 |
| BERT-base-uncased — [CLS] head, aspect-unaware, CE loss | 0.0762 | 0.0000 | 0.0000 | 0.2000 |
| Classical TF-IDF + LinearSVC — no deep learning | 0.0000 | 0.1429 | 0.0000 | 0.2353 |

---
# Ablation Studies


## A1: Dependency GCN — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Full model (with Dependency GCN) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 0.61 min |
| No GCN — aspect attention only | 0.8802 | **0.6863** | 0.8794 | 0.6701 | 0.52 min |


## A1: Dependency GCN — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full model (with Dependency GCN) | 0.7989 | 0.8104 | 0.7561 | 0.4062 | 0.7234 | 0.8011 | 0.7426 | 0.7198 |
| No GCN — aspect attention only | 0.6661 | 0.6962 | 0.7075 | 0.4449 | 0.6421 | 0.6453 | 0.5434 | 0.6208 |


## A1: Dependency GCN — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| Full model (with Dependency GCN) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| No GCN — aspect attention only | 0.2222 | 0.1333 | 0.0000 | 0.3333 |


## A2: Aspect Attention — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Aspect-guided MHA attention | 0.9234 | **0.7904** | 0.9229 | 0.7847 | 0.61 min |
| CLS token pooling (no attention) | 0.7317 | **0.5378** | 0.7507 | 0.4376 | 0.60 min |


## A2: Aspect Attention — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aspect-guided MHA attention | 0.8202 | 0.8068 | 0.7399 | 0.6019 | 0.7333 | 0.7922 | 0.6026 | 0.7281 |
| CLS token pooling (no attention) | 0.5449 | 0.5238 | 0.5521 | 0.3391 | 0.4793 | 0.6294 | 0.3805 | 0.4927 |


## A2: Aspect Attention — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| Aspect-guided MHA attention | 0.5714 | 0.2500 | 0.0000 | 0.3636 |
| CLS token pooling (no attention) | 0.0508 | 0.1250 | 0.0000 | 0.3448 |


## A3: Loss Function — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Hybrid Loss (Focal + CB + Dice) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 0.61 min |
| Focal Loss only | 0.9178 | **0.7725** | 0.9166 | 0.7731 | 0.60 min |
| Class-Balanced Loss only | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 0.60 min |
| Dice Loss only | 0.7823 | **0.2926** | 0.6868 | 0.0000 | 44.51 min |
| Cross-Entropy Loss (no imbalance handling) | 0.9250 | **0.7911** | 0.9246 | 0.7939 | 93.48 min |


## A3: Loss Function — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid Loss (Focal + CB + Dice) | 0.7989 | 0.8104 | 0.7561 | 0.4062 | 0.7234 | 0.8011 | 0.7426 | 0.7198 |
| Focal Loss only | 0.8106 | 0.7653 | 0.7516 | 0.4718 | 0.7381 | 0.7707 | 0.5985 | 0.7009 |
| Class-Balanced Loss only | 0.8338 | 0.7988 | 0.7295 | 0.4223 | 0.7689 | 0.7819 | 0.7077 | 0.7204 |
| Dice Loss only | 0.2375 | 0.2827 | 0.3006 | 0.3278 | 0.3050 | 0.2580 | 0.3243 | 0.2909 |
| Cross-Entropy Loss (no imbalance handling) | 0.8338 | 0.7988 | 0.7295 | 0.4223 | 0.7689 | 0.7819 | 0.7077 | 0.7204 |


## A3: Loss Function — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| Hybrid Loss (Focal + CB + Dice) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Focal Loss only | 0.2000 | 0.2353 | 0.0000 | 0.4211 |
| Class-Balanced Loss only | 0.2857 | 0.0000 | 0.2857 | 0.3636 |
| Dice Loss only | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Cross-Entropy Loss (no imbalance handling) | 0.2857 | 0.0000 | 0.2857 | 0.3636 |


## A4: Data Augmentation — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| With LLM synthetic augmentation (10,050 samples) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 103.01 min |
| Without augmentation (9,240 samples) | 0.9245 | **0.7872** | 0.9225 | 0.7874 | 68.82 min |


## A4: Data Augmentation — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| With LLM synthetic augmentation (10,050 samples) | 0.7989 | 0.8104 | 0.7561 | 0.4062 | 0.7234 | 0.8011 | 0.7426 | 0.7198 |
| Without augmentation (9,240 samples) | 0.8096 | 0.7902 | 0.7378 | 0.3282 | 0.7417 | 0.7822 | 0.5908 | 0.6829 |


## A4: Data Augmentation — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| With LLM synthetic augmentation (10,050 samples) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Without augmentation (9,240 samples) | 0.0000 | 0.0000 | 0.0000 | 0.3810 |


## A5: Classifier Head — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| 7 aspect-specific classifier heads | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 103.00 min |
| Single shared classifier head | 0.9205 | **0.7797** | 0.9194 | 0.7778 | 88.87 min |


## A5: Classifier Head — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 aspect-specific classifier heads | 0.7989 | 0.8104 | 0.7561 | 0.4062 | 0.7234 | 0.8011 | 0.7426 | 0.7198 |
| Single shared classifier head | 0.7824 | 0.7932 | 0.7458 | 0.4554 | 0.7489 | 0.8062 | 0.6040 | 0.7051 |


## A5: Classifier Head — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| 7 aspect-specific classifier heads | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| Single shared classifier head | 0.2000 | 0.1905 | 0.0000 | 0.4000 |


## A6: Mixed Sentiment Resolution — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| MSR Eval: Full model + GCN (mixed sent resolution) | 0.9236 | **0.7856** | 0.9221 | 0.7838 | 102.07 min |
| MSR Eval: No GCN (attention only, no dep parsing) | 0.9212 | **0.7877** | 0.9212 | 0.7799 | 109.12 min |


## A6: Mixed Sentiment Resolution — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSR Eval: Full model + GCN (mixed sent resolution) | 0.7989 | 0.8104 | 0.7561 | 0.4062 | 0.7234 | 0.8011 | 0.7426 | 0.7198 |
| MSR Eval: No GCN (attention only, no dep parsing) | 0.7951 | 0.7900 | 0.7696 | 0.4865 | 0.7361 | 0.8072 | 0.6033 | 0.7125 |


## A6: Mixed Sentiment Resolution — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| MSR Eval: Full model + GCN (mixed sent resolution) | 0.0000 | 0.2353 | 0.4000 | 0.4390 |
| MSR Eval: No GCN (attention only, no dep parsing) | 0.3333 | 0.1429 | 0.0000 | 0.4615 |


## A7: Hybrid Loss Weights — Overall Metrics

| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |
|-----------|----------|----------|-------------|-----|------|
| Hybrid Loss (Focal 1.0 + CB 0.5 + Dice 0.0) | 0.9247 | **0.7944** | 0.9236 | 0.7900 | 133.23 min |
| Hybrid Loss (Focal 1.0 + CB 1.0 + Dice 0.0) | 0.9203 | **0.7885** | 0.9211 | 0.7813 | 102.63 min |


## A7: Hybrid Loss Weights — Per-Aspect Macro-F1

| Experiment | stayingpower | texture | smell | price | colour | shipping | packing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid Loss (Focal 1.0 + CB 0.5 + Dice 0.0) | 0.7933 | 0.8088 | 0.7311 | 0.3275 | 0.7647 | 0.7975 | 0.5997 | 0.6889 |
| Hybrid Loss (Focal 1.0 + CB 1.0 + Dice 0.0) | 0.8122 | 0.7996 | 0.7653 | 0.4956 | 0.7442 | 0.7909 | 0.6269 | 0.7192 |


## A7: Hybrid Loss Weights — Rare Class F1

| Experiment | price-negative | price-neutral | packing-neutral | smell-neutral |
| --- | --- | --- | --- | --- |
| Hybrid Loss (Focal 1.0 + CB 0.5 + Dice 0.0) | 0.0000 | 0.0000 | 0.0000 | 0.3429 |
| Hybrid Loss (Focal 1.0 + CB 1.0 + Dice 0.0) | 0.2222 | 0.2857 | 0.0000 | 0.4286 |
