# Class Imbalance Analysis

**Project**: Class balanced aspect base mixed sentiment resolution with XAI  
**Date**: 2026-03-03 22:39:52

## Cleaning Pipeline Applied

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | Unicode NFC normalisation | Unify combining characters |
| 2 | HTML tag & entity removal | Strip `<br>`, `&amp;`, `&#39;` etc. |
| 3 | URL / e-mail removal | Free up token budget |
| 4 | Translation artifact normalisation | Fix vi→en MT filler & punctuation |
| 5 | Garbled / keyboard-spam removal | Drop incoherent tokens |
| 6 | Whitespace collapse | Clean token boundaries |

## Dataset Split Distribution

| Set | Samples | % |
|-----|---------|---|
| Train | 9258 | 70% |
| Validation | 1985 | 15% |
| Test | 1985 | 15% |

## Aspect-wise Class Distribution


### STAYINGPOWER

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 7299 (78.84%) | 1571 (79.14%) | 1586 (79.90%) |
| negative | 664 (7.17%) | 138 (6.95%) | 138 (6.95%) |
| neutral | 229 (2.47%) | 46 (2.32%) | 38 (1.91%) |
| positive | 1066 (11.51%) | 230 (11.59%) | 223 (11.23%) |

### TEXTURE

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 5830 (62.97%) | 1267 (63.83%) | 1257 (63.32%) |
| negative | 561 (6.06%) | 118 (5.94%) | 116 (5.84%) |
| neutral | 392 (4.23%) | 74 (3.73%) | 76 (3.83%) |
| positive | 2475 (26.73%) | 526 (26.50%) | 536 (27.00%) |

### SMELL

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 7221 (78.00%) | 1551 (78.14%) | 1553 (78.24%) |
| negative | 307 (3.32%) | 67 (3.38%) | 65 (3.27%) |
| neutral | 93 (1.00%) | 18 (0.91%) | 19 (0.96%) |
| positive | 1637 (17.68%) | 349 (17.58%) | 348 (17.53%) |

### PRICE

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 6965 (75.23%) | 1493 (75.21%) | 1500 (75.57%) |
| negative | 15 (0.16%) | 3 (0.15%) | 3 (0.15%) |
| neutral | 22 (0.24%) | 2 (0.10%) | 2 (0.10%) |
| positive | 2256 (24.37%) | 487 (24.53%) | 480 (24.18%) |

### COLOUR

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 4016 (43.38%) | 854 (43.02%) | 864 (43.53%) |
| negative | 458 (4.95%) | 103 (5.19%) | 101 (5.09%) |
| neutral | 387 (4.18%) | 78 (3.93%) | 77 (3.88%) |
| positive | 4397 (47.49%) | 950 (47.86%) | 943 (47.51%) |

### SHIPPING

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 5459 (58.97%) | 1174 (59.14%) | 1173 (59.09%) |
| negative | 1182 (12.77%) | 256 (12.90%) | 250 (12.59%) |
| neutral | 244 (2.64%) | 50 (2.52%) | 48 (2.42%) |
| positive | 2373 (25.63%) | 505 (25.44%) | 514 (25.89%) |

### PACKING

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
|-------|----------------|---------------|----------------|
| nan | 7139 (77.11%) | 1545 (77.83%) | 1538 (77.48%) |
| negative | 70 (0.76%) | 11 (0.55%) | 20 (1.01%) |
| neutral | 11 (0.12%) | 1 (0.05%) | 6 (0.30%) |
| positive | 2038 (22.01%) | 428 (21.56%) | 421 (21.21%) |

## Imbalanced Classes Identified

**Threshold**: < 10.0% in training set

### STAYINGPOWER

- **negative**: 664 samples (7.17%)
- **neutral**: 229 samples (2.47%)

### TEXTURE

- **negative**: 561 samples (6.06%)
- **neutral**: 392 samples (4.23%)

### SMELL

- **negative**: 307 samples (3.32%)
- **neutral**: 93 samples (1.00%)

### PRICE

- **neutral**: 22 samples (0.24%)
- **negative**: 15 samples (0.16%)

### COLOUR

- **negative**: 458 samples (4.95%)
- **neutral**: 387 samples (4.18%)

### SHIPPING

- **neutral**: 244 samples (2.64%)

### PACKING

- **negative**: 70 samples (0.76%)
- **neutral**: 11 samples (0.12%)

## Recommendations

1. Use class-balanced loss functions (Focal Loss or Weighted Cross-Entropy)
2. Consider oversampling rare classes (SMOTE on embeddings or text augmentation)
3. Apply label-smoothing during training to reduce over-confident predictions
4. Use ensemble methods / multi-exit architectures to boost minority class F1
5. Monitor per-class Precision / Recall / F1 — do NOT rely on macro accuracy alone

## Stratification Strategy

Dataset split with **stratified sampling** over multi-label aspect-sentiment keys to ensure:
- Proportional representation of all aspect-sentiment combinations across every split
- Rare classes maintain the same percentage in train / val / test
- Garbled / empty rows are excluded before splitting so they do not dilute any split
