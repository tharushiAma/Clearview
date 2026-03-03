# Data Augmentation — Class Imbalance Impact Report

**Project**: Class Imbalanced Multi-Aspect Mixed Sentiment Resolution with XAI  
**Generated**: 2026-03-03 23:05:41

## Dataset Overview

| Metric | Count |
|--------|-------|
| Original training samples | 9,258 |
| Synthetic samples added | 810 |
| **Combined (augmented) training samples** | **10,068** |

### Synthetic Data Sources

| File | Records |
|------|---------|
| LLM_Gen_Packing_Neg_Reviews.csv | 192 |
| LLM_Gen_Packing_Neu_Reviews.csv | 196 |
| LLM_Gen_Price_Neg_Reviews.csv | 176 |
| LLM_Gen_Price_Neu_Reviews.csv | 193 |
| LLM_Gen_Smell_Neu_Reviews.csv | 166 |

---

## Before vs After Augmentation — Per-Aspect Class Distribution

### STAYINGPOWER

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 1,066 | 54.4% | 1,083 | 52.7% | ↑ +17 | -1.7% |
| neutral | 229 | 11.7% | 249 | 12.1% | ↑ +20 | +0.4% |
| negative | 664 | 33.9% | 723 | 35.2% | ↑ +59 | +1.3% |
| **Total** | **1,959** | | **2,055** | | **+96** | |

### TEXTURE

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 2,475 | 72.2% | 2,571 | 70.8% | ↑ +96 | -1.4% |
| neutral | 392 | 11.4% | 427 | 11.8% | ↑ +35 | +0.3% |
| negative | 561 | 16.4% | 635 | 17.5% | ↑ +74 | +1.1% |
| **Total** | **3,428** | | **3,633** | | **+205** | |

### SMELL

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 1,637 | 80.4% | 1,671 | 73.3% | ↑ +34 | -7.1% |
| neutral | 93 | 4.6% | 272 | 11.9% | ↑ +179 | +7.4% |
| negative | 307 | 15.1% | 337 | 14.8% | ↑ +30 | -0.3% |
| **Total** | **2,037** | | **2,280** | | **+243** | |

### PRICE

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 2,256 | 98.4% | 2,284 | 83.9% | ↑ +28 | -14.5% |
| neutral | 22 | 1.0% | 231 | 8.5% | ↑ +209 | +7.5% |
| negative | 15 | 0.7% | 207 | 7.6% | ↑ +192 | +7.0% |
| **Total** | **2,293** | | **2,722** | | **+429** | |

### COLOUR

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 4,397 | 83.9% | 4,492 | 83.0% | ↑ +95 | -0.9% |
| neutral | 387 | 7.4% | 425 | 7.9% | ↑ +38 | +0.5% |
| negative | 458 | 8.7% | 494 | 9.1% | ↑ +36 | +0.4% |
| **Total** | **5,242** | | **5,411** | | **+169** | |

### SHIPPING

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 2,373 | 62.5% | 2,414 | 61.9% | ↑ +41 | -0.6% |
| neutral | 244 | 6.4% | 266 | 6.8% | ↑ +22 | +0.4% |
| negative | 1,182 | 31.1% | 1,222 | 31.3% | ↑ +40 | +0.2% |
| **Total** | **3,799** | | **3,902** | | **+103** | |

### PACKING

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
|-------|-------------|----------|-------------|---------|---------|-----|
| positive | 2,038 | 96.2% | 2,072 | 83.1% | ↑ +34 | -13.0% |
| neutral | 11 | 0.5% | 179 | 7.2% | ↑ +168 | +6.7% |
| negative | 70 | 3.3% | 241 | 9.7% | ↑ +171 | +6.4% |
| **Total** | **2,119** | | **2,492** | | **+373** | |

---

## Summary of Improvements

| Aspect | Class | Before Count | After Count | Samples Added | % Change |
|--------|-------|-------------|-------------|---------------|----------|
| stayingpower | positive | 1,066 | 1,083 | +17 | +1.6% |
| stayingpower | neutral | 229 | 249 | +20 | +8.7% |
| stayingpower | negative | 664 | 723 | +59 | +8.9% |
| texture | positive | 2,475 | 2,571 | +96 | +3.9% |
| texture | neutral | 392 | 427 | +35 | +8.9% |
| texture | negative | 561 | 635 | +74 | +13.2% |
| smell | positive | 1,637 | 1,671 | +34 | +2.1% |
| smell | neutral | 93 | 272 | +179 | +192.5% |
| smell | negative | 307 | 337 | +30 | +9.8% |
| price | positive | 2,256 | 2,284 | +28 | +1.2% |
| price | neutral | 22 | 231 | +209 | +950.0% |
| price | negative | 15 | 207 | +192 | +1280.0% |
| colour | positive | 4,397 | 4,492 | +95 | +2.2% |
| colour | neutral | 387 | 425 | +38 | +9.8% |
| colour | negative | 458 | 494 | +36 | +7.9% |
| shipping | positive | 2,373 | 2,414 | +41 | +1.7% |
| shipping | neutral | 244 | 266 | +22 | +9.0% |
| shipping | negative | 1,182 | 1,222 | +40 | +3.4% |
| packing | positive | 2,038 | 2,072 | +34 | +1.7% |
| packing | neutral | 11 | 179 | +168 | +1527.3% |
| packing | negative | 70 | 241 | +171 | +244.3% |

---

## Recommendations

> [!NOTE]
> Data augmentation addresses imbalance at the **data level**. The training pipeline
> also uses **Focal Loss** (model-level) to further down-weight easy/majority examples
> and up-weight hard/minority examples. Both techniques work together.

1. **Monitor per-class F1** during training — not just overall accuracy
2. If rare classes still underperform, consider generating more targeted synthetic data
3. The stratified split in `preprocess_and_split.py` ensures the val/test sets reflect the original (non-augmented) distribution for honest evaluation
