# Data Augmentation — Class Imbalance Impact Report

**Generated:** 2026-03-04 09:21:49

## Dataset overview

| | Count |
| --- | --- |
| Original training samples | 9,240 |
| Synthetic samples added | 810 |
| Combined (augmented) training samples | 10,050 |

### Synthetic files used

| File | Records |
| --- | --- |
| LLM_Gen_Packing_Neg_Reviews.csv | 192 |
| LLM_Gen_Packing_Neu_Reviews.csv | 196 |
| LLM_Gen_Price_Neg_Reviews.csv | 176 |
| LLM_Gen_Price_Neu_Reviews.csv | 193 |
| LLM_Gen_Smell_Neu_Reviews.csv | 166 |

---

## Before vs after — per-aspect class distribution

### STAYINGPOWER

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 1,059 | 54.3% | 1,076 | 52.6% | ↑ +17 | -1.7% |
| neutral | 224 | 11.5% | 244 | 11.9% | ↑ +20 | +0.4% |
| negative | 668 | 34.2% | 727 | 35.5% | ↑ +59 | +1.3% |
| **Total** | **1,951** | | **2,047** | | **+96** | |

### TEXTURE

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 2,467 | 72.2% | 2,563 | 70.8% | ↑ +96 | -1.4% |
| neutral | 385 | 11.3% | 420 | 11.6% | ↑ +35 | +0.3% |
| negative | 565 | 16.5% | 639 | 17.6% | ↑ +74 | +1.1% |
| **Total** | **3,417** | | **3,622** | | **+205** | |

### SMELL

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 1,634 | 80.3% | 1,668 | 73.3% | ↑ +34 | -7.1% |
| neutral | 95 | 4.7% | 274 | 12.0% | ↑ +179 | +7.4% |
| negative | 305 | 15.0% | 335 | 14.7% | ↑ +30 | -0.3% |
| **Total** | **2,034** | | **2,277** | | **+243** | |

### PRICE

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 2,267 | 99.2% | 2,295 | 84.6% | ↑ +28 | -14.7% |
| neutral | 9 | 0.4% | 218 | 8.0% | ↑ +209 | +7.6% |
| negative | 9 | 0.4% | 201 | 7.4% | ↑ +192 | +7.0% |
| **Total** | **2,285** | | **2,714** | | **+429** | |

### COLOUR

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 4,388 | 83.7% | 4,483 | 82.9% | ↑ +95 | -0.9% |
| neutral | 390 | 7.4% | 428 | 7.9% | ↑ +38 | +0.5% |
| negative | 463 | 8.8% | 499 | 9.2% | ↑ +36 | +0.4% |
| **Total** | **5,241** | | **5,410** | | **+169** | |

### SHIPPING

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 2,365 | 62.5% | 2,406 | 61.9% | ↑ +41 | -0.6% |
| neutral | 245 | 6.5% | 267 | 6.9% | ↑ +22 | +0.4% |
| negative | 1,176 | 31.1% | 1,216 | 31.3% | ↑ +40 | +0.2% |
| **Total** | **3,786** | | **3,889** | | **+103** | |

### PACKING

| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |
| --- | --- | --- | --- | --- | --- | --- |
| positive | 2,034 | 96.5% | 2,068 | 83.4% | ↑ +34 | -13.1% |
| neutral | 9 | 0.4% | 177 | 7.1% | ↑ +168 | +6.7% |
| negative | 65 | 3.1% | 236 | 9.5% | ↑ +171 | +6.4% |
| **Total** | **2,108** | | **2,481** | | **+373** | |

---

## Summary

| Aspect | Class | Before | After | Added | % Change |
| --- | --- | --- | --- | --- | --- |
| stayingpower | positive | 1,059 | 1,076 | +17 | +1.6% |
| stayingpower | neutral | 224 | 244 | +20 | +8.9% |
| stayingpower | negative | 668 | 727 | +59 | +8.8% |
| texture | positive | 2,467 | 2,563 | +96 | +3.9% |
| texture | neutral | 385 | 420 | +35 | +9.1% |
| texture | negative | 565 | 639 | +74 | +13.1% |
| smell | positive | 1,634 | 1,668 | +34 | +2.1% |
| smell | neutral | 95 | 274 | +179 | +188.4% |
| smell | negative | 305 | 335 | +30 | +9.8% |
| price | positive | 2,267 | 2,295 | +28 | +1.2% |
| price | neutral | 9 | 218 | +209 | +2322.2% |
| price | negative | 9 | 201 | +192 | +2133.3% |
| colour | positive | 4,388 | 4,483 | +95 | +2.2% |
| colour | neutral | 390 | 428 | +38 | +9.7% |
| colour | negative | 463 | 499 | +36 | +7.8% |
| shipping | positive | 2,365 | 2,406 | +41 | +1.7% |
| shipping | neutral | 245 | 267 | +22 | +9.0% |
| shipping | negative | 1,176 | 1,216 | +40 | +3.4% |
| packing | positive | 2,034 | 2,068 | +34 | +1.7% |
| packing | neutral | 9 | 177 | +168 | +1866.7% |
| packing | negative | 65 | 236 | +171 | +263.1% |
