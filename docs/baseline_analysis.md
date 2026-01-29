# Baseline Model Analysis

This document summarizes the findings from the initial execution of the DistilBERT baseline model.

## 1. Class Imbalance Evidence
Analysis of the training data reveals severe class imbalance across all aspects.

| Aspect | Positive | Negative | Neutral | Note |
|:-------|:--------:|:--------:|:-------:|:-----|
| **Price** | 33 | **0** | **0** | **Critical**: No negative/neutral samples to learn from. |
| **Packing** | 27 | 3 | 0 | Severe imbalance. |
| **Smell** | 24 | 6 | 1 | Heavily skewed positive. |
| **Colour** | 67 | 10 | 5 | Heavily skewed positive. |
| **Texture** | 47 | 10 | 4 | Heavily skewed positive. |
| **Staying Power** | 17 | 14 | 1 | Relatively balanced (pos/neg), but low neutral. |
| **Shipping** | 33 | 20 | 2 | Best balance, but still low neutral. |

## 2. Baseline Weakness (Zero-shot performance)
The class imbalance directly impacts the model's ability to learn minority classes, resulting in **0.00 F1-scores** for negative and neutral classes in many aspects.

### Key Observations from Evaluation (Validation Set)
- **Zero Recall for Minority Classes**: For aspects like `stayingpower`, `texture`, and `smell`, the model failed to predict a single negative or neutral instance correctly (producing `Precision: 0.00`, `Recall: 0.00`).
- **Bias Towards Majority Class**: In almost all cases, the model defaults to predicting the majority class (Positive), leading to misleadingly high accuracy but poor macro-F1.
- **Example: Shipping**
  - Validation: 4 Negative, 3 Positive.
  - Model Recall: 0% for Negative, 100% for Positive.
  - Result: The model completely ignored the negative feedback.

## 3. Justification for Imbalance Handling
The evidence above proves that a standard cross-entropy loss is insufficient.
- **Problem**: The model minimizes loss by simply predicting "Positive" for everything.
- **Solution Required**:
    1.  **Weighted Loss**: Penalize the model more for missing rare classes (Negative/Neutral).
    2.  **Oversampling/Augmentation**: Artificially increase the number of minority class samples.
    3.  **Focal Loss**: Focus learning on hard-to-classify examples.

## 4. Next Steps
- Implement **Weighted CrossEntropyLoss** using the inverse class frequencies calculated above.
- Experiment with **Focal Loss** to reduce the impact of easy positives.
