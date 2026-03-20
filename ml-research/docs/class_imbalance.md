# Class Imbalance Analysis

Project: Class balanced aspect base mixed sentiment resolution with XAI
Date: 2026-03-04 09:10:19

## Cleaning pipeline

| Stage | Technique | Purpose |
| --- | --- | --- |
| 1 | Unicode NFC normalisation | Unify combining characters |
| 2 | HTML tag & entity removal | Strip `<br>`, `&amp;`, `&#39;` etc. |
| 3 | URL / e-mail removal | Free up token budget |
| 4 | Translation artifact normalisation | Fix vi→en MT filler & punctuation |
| 5 | Garbled / keyboard-spam removal | Drop incoherent tokens |
| 6 | Whitespace collapse | Clean token boundaries |

## Dataset split

| Set | Samples | % |
| --- | --- | --- |
| Train | 9240 | 70% |
| Validation | 1994 | 15% |
| Test | 1994 | 15% |

## Per-aspect class distribution

### STAYINGPOWER

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 7289 (78.89%) | 1585 (79.49%) | 1582 (79.34%) |
| negative | 668 (7.23%) | 134 (6.72%) | 138 (6.92%) |
| neutral | 224 (2.42%) | 43 (2.16%) | 46 (2.31%) |
| positive | 1059 (11.46%) | 232 (11.63%) | 228 (11.43%) |

### TEXTURE

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 5823 (63.02%) | 1269 (63.64%) | 1262 (63.29%) |
| negative | 565 (6.11%) | 113 (5.67%) | 117 (5.87%) |
| neutral | 385 (4.17%) | 81 (4.06%) | 76 (3.81%) |
| positive | 2467 (26.70%) | 531 (26.63%) | 539 (27.03%) |

### SMELL

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 7206 (77.99%) | 1556 (78.03%) | 1563 (78.39%) |
| negative | 305 (3.30%) | 71 (3.56%) | 63 (3.16%) |
| neutral | 95 (1.03%) | 21 (1.05%) | 14 (0.70%) |
| positive | 1634 (17.68%) | 346 (17.35%) | 354 (17.75%) |

### PRICE

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 6955 (75.27%) | 1500 (75.23%) | 1503 (75.38%) |
| negative | 9 (0.10%) | 7 (0.35%) | 5 (0.25%) |
| neutral | 9 (0.10%) | 6 (0.30%) | 11 (0.55%) |
| positive | 2267 (24.53%) | 481 (24.12%) | 475 (23.82%) |

### COLOUR

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 3999 (43.28%) | 871 (43.68%) | 864 (43.33%) |
| negative | 463 (5.01%) | 101 (5.07%) | 98 (4.91%) |
| neutral | 390 (4.22%) | 73 (3.66%) | 79 (3.96%) |
| positive | 4388 (47.49%) | 949 (47.59%) | 953 (47.79%) |

### SHIPPING

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 5454 (59.03%) | 1174 (58.88%) | 1178 (59.08%) |
| negative | 1176 (12.73%) | 259 (12.99%) | 253 (12.69%) |
| neutral | 245 (2.65%) | 49 (2.46%) | 48 (2.41%) |
| positive | 2365 (25.60%) | 512 (25.68%) | 515 (25.83%) |

### PACKING

| Class | Train Count (%) | Val Count (%) | Test Count (%) |
| --- | --- | --- | --- |
| nan | 7132 (77.19%) | 1549 (77.68%) | 1541 (77.28%) |
| negative | 65 (0.70%) | 15 (0.75%) | 21 (1.05%) |
| neutral | 9 (0.10%) | 6 (0.30%) | 3 (0.15%) |
| positive | 2034 (22.01%) | 424 (21.26%) | 429 (21.51%) |

## Imbalanced classes (threshold: < 10% in training set)

**STAYINGPOWER:** negative 668 (7.23%), neutral 224 (2.42%)

**TEXTURE:** negative 565 (6.11%), neutral 385 (4.17%)

**SMELL:** negative 305 (3.30%), neutral 95 (1.03%)

**PRICE:** neutral 9 (0.10%), negative 9 (0.10%) — the worst by far

**COLOUR:** negative 463 (5.01%), neutral 390 (4.22%)

**SHIPPING:** neutral 245 (2.65%)

**PACKING:** negative 65 (0.70%), neutral 9 (0.10%)
