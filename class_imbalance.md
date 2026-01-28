# Class Imbalance Analysis

This report details the sentiment distribution for each aspect across the Train, Validation, and Test datasets.

## Train Dataset

| Aspect | Positive | Negative | Neutral | Total |
| :--- | :--- | :--- | :--- | :--- |
| stayingpower | 1060 | 647 | 220 | 1927 |
| texture | 2521 | 547 | 358 | 3426 |
| smell | 1625 | 313 | 93 | 2031 |
| price | 2244 | 17 | 15 | 2276 |
| colour | 4437 | 456 | 366 | 5259 |
| shipping | 2363 | 1135 | 256 | 3754 |
| packing | 2026 | 69 | 14 | 2109 |

## Validation Dataset

| Aspect | Positive | Negative | Neutral | Total |
| :--- | :--- | :--- | :--- | :--- |
| stayingpower | 144 | 94 | 26 | 264 |
| texture | 341 | 76 | 56 | 473 |
| smell | 248 | 37 | 16 | 301 |
| price | 315 | 2 | 7 | 324 |
| colour | 638 | 58 | 65 | 761 |
| shipping | 352 | 178 | 28 | 558 |
| packing | 287 | 11 | 1 | 299 |

## Test Dataset

| Aspect | Positive | Negative | Neutral | Total |
| :--- | :--- | :--- | :--- | :--- |
| stayingpower | 315 | 199 | 71 | 585 |
| texture | 676 | 172 | 132 | 980 |
| smell | 463 | 89 | 21 | 573 |
| price | 665 | 2 | 4 | 671 |
| colour | 1218 | 148 | 111 | 1477 |
| shipping | 679 | 376 | 58 | 1113 |
| packing | 574 | 21 | 3 | 598 |

> [!IMPORTANT]
> Aspects like `price` and `packing` show extreme class imbalance with very few negative and neutral examples. This confirms the need for Adaptive Focal Loss or specialized sampling.
