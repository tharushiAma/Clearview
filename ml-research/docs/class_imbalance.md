# Class Imbalance Analysis

**Project:** Class-Balanced Aspect-Based Mixed Sentiment Resolution with XAI
**Date generated:** 2026-04-02
**Source:** `notebooks/02_preprocess_and_split.ipynb`

---

## Cleaning Pipeline Applied

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | Unicode NFC normalisation | Unify combining characters |
| 2 | HTML tag & entity removal | Strip `<br>`, `&amp;`, `&#39;` etc. |
| 3 | URL / e-mail removal | Remove non-sentiment noise |
| 4 | Translation artifact normalisation | Fix vi→en MT filler & punctuation |
| 5 | Whitespace collapse | Clean token boundaries |

---

## Dataset Split

**Input:** 13,241 rows loaded from `data/raw/full_data_en.csv`
**After cleaning:** 13,240 rows (1 empty-text row dropped)

| Set | Samples | % of total |
|-----|---------|------------|
| Train | 9,240 | 69.8% |
| Validation | 1,994 | 15.1% |
| Test | 1,994 | 15.1% |

### Rare-class reservation (Phase 1 split)

Classes whose total count was below the `rare_threshold` (= `MIN_EVAL_SAMPLES / TEST_SPLIT` = 53) were extracted and split 40/30/30 (train/val/test) to guarantee minimum eval-set representation.

| Aspect | Class | Total samples | Action |
|--------|-------|--------------|--------|
| price | neutral | 26 | Reserved |
| price | negative | 21 | Reserved |
| packing | neutral | 18 | Reserved |

65 rows reserved → split: Train 27 / Val 19 / Test 19

---

## Aspect-wise Class Distribution

### STAYINGPOWER

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 7,289 (78.89%) | 1,585 (79.49%) | 1,582 (79.34%) |
| negative | 668 (7.23%) | 134 (6.72%) | 138 (6.92%) |
| neutral | 224 (2.42%) | 43 (2.16%) | 46 (2.31%) |
| positive | 1,059 (11.46%) | 232 (11.63%) | 228 (11.43%) |

### TEXTURE

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 5,823 (63.02%) | 1,269 (63.64%) | 1,262 (63.29%) |
| negative | 565 (6.11%) | 113 (5.67%) | 117 (5.87%) |
| neutral | 385 (4.17%) | 81 (4.06%) | 76 (3.81%) |
| positive | 2,467 (26.70%) | 531 (26.63%) | 539 (27.03%) |

### SMELL

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 7,206 (77.99%) | 1,556 (78.03%) | 1,563 (78.39%) |
| negative | 305 (3.30%) | 71 (3.56%) | 63 (3.16%) |
| neutral | 95 (1.03%) | 21 (1.05%) | 14 (0.70%) |
| positive | 1,634 (17.68%) | 346 (17.35%) | 354 (17.75%) |

### PRICE

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 6,955 (75.27%) | 1,500 (75.23%) | 1,503 (75.38%) |
| negative | 9 (0.10%) | 7 (0.35%) | 5 (0.25%) |
| neutral | 9 (0.10%) | 6 (0.30%) | 11 (0.55%) |
| positive | 2,267 (24.53%) | 481 (24.12%) | 475 (23.82%) |

### COLOUR

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 3,999 (43.28%) | 871 (43.68%) | 864 (43.33%) |
| negative | 463 (5.01%) | 101 (5.07%) | 98 (4.91%) |
| neutral | 390 (4.22%) | 73 (3.66%) | 79 (3.96%) |
| positive | 4,388 (47.49%) | 949 (47.59%) | 953 (47.79%) |

### SHIPPING

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 5,454 (59.03%) | 1,174 (58.88%) | 1,178 (59.08%) |
| negative | 1,176 (12.73%) | 259 (12.99%) | 253 (12.69%) |
| neutral | 245 (2.65%) | 49 (2.46%) | 48 (2.41%) |
| positive | 2,365 (25.60%) | 512 (25.68%) | 515 (25.83%) |

### PACKING

| Class | Train | Val | Test |
|-------|-------|-----|------|
| nan | 7,132 (77.19%) | 1,549 (77.68%) | 1,541 (77.28%) |
| negative | 65 (0.70%) | 15 (0.75%) | 21 (1.05%) |
| neutral | 9 (0.10%) | 6 (0.30%) | 3 (0.15%) |
| positive | 2,034 (22.01%) | 424 (21.26%) | 429 (21.51%) |

---

## Imbalanced Classes (threshold: < 10% in training set)

> All seven aspects have at least one imbalanced class. The dominant pattern is that **negative** and **neutral** sentiment classes are severely under-represented relative to **positive** — a natural consequence of the review platform bias toward satisfied customers.

| Aspect | Class | Train count | Train % | Severity |
|--------|-------|------------|---------|----------|
| STAYINGPOWER | negative | 668 | 7.23% | Moderate |
| STAYINGPOWER | neutral | 224 | 2.42% | Severe |
| TEXTURE | negative | 565 | 6.11% | Moderate |
| TEXTURE | neutral | 385 | 4.17% | Severe |
| SMELL | negative | 305 | 3.30% | Severe |
| SMELL | neutral | 95 | 1.03% | Critical |
| PRICE | neutral | 9 | 0.10% | **Critical — worst in dataset** |
| PRICE | negative | 9 | 0.10% | **Critical — worst in dataset** |
| COLOUR | negative | 463 | 5.01% | Moderate |
| COLOUR | neutral | 390 | 4.22% | Severe |
| SHIPPING | neutral | 245 | 2.65% | Severe |
| PACKING | negative | 65 | 0.70% | Critical |
| PACKING | neutral | 9 | 0.10% | **Critical** |

