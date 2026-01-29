# Sentiment Distribution per Aspect (Detailed Integration & Deduplication)

This report provides a granular comparison of the sentiment distribution across all aspects, highlighting the impact of synthetic data integration and the subsequent deduplication process.

| Aspect | Sentiment | Original Records | Initially Augmented | Clean Augmented | Final Total | Total Increase |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **stayingpower** | positive | 1,060 | 22 | 17 | 1,077 | +17 |
| | negative | 647 | 61 | 59 | 706 | +59 |
| | neutral | 220 | 21 | 20 | 240 | +20 |
| **texture** | positive | 2,521 | 106 | 96 | 2,617 | +96 |
| | negative | 547 | 78 | 74 | 621 | +74 |
| | neutral | 358 | 37 | 35 | 393 | +35 |
| **smell** | positive | 1,625 | 37 | 34 | 1,659 | +34 |
| | negative | 313 | 34 | 30 | 343 | +30 |
| | neutral | 93 | 197 | 179 | 272 | +179 |
| **price** | positive | 2,244 | 37 | 28 | 2,272 | +28 |
| | negative | 17 | 197 | 192 | 209 | +192 |
| | neutral | 15 | 212 | 209 | 224 | +209 |
| **colour** | positive | 4,437 | 103 | 95 | 4,532 | +95 |
| | negative | 456 | 40 | 36 | 492 | +36 |
| | neutral | 366 | 39 | 38 | 404 | +38 |
| **shipping** | positive | 2,363 | 49 | 41 | 2,404 | +41 |
| | negative | 1,135 | 47 | 40 | 1,175 | +40 |
| | neutral | 256 | 23 | 22 | 278 | +22 |
| **packing** | positive | 2,026 | 36 | 34 | 2,060 | +34 |
| | negative | 69 | 218 | 171 | 240 | +171 |
| | neutral | 14 | 217 | 168 | 182 | +168 |

## Data Integrity & Deduplication Validations

To ensure the highest quality for the augmented training set, the following steps were performed:

1.  **Deduplication**: We removed **113** internal duplicate records from the synthetic data. This column (**Clean Augmented**) shows only the unique synthetic samples that were actually integrated.
2.  **Overlap Prevention**: Verified that no clean synthetic samples duplicate existing records from the original training set.
3.  **Final Tally**: The resulting cleaned augmented dataset `data/splits/train_aug.parquet` contains **10,078** unique training records.
