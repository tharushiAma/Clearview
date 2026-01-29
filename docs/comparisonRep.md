# Advanced Model Comparison: RoBERTa-GCN Variants

This report compares four advanced variations of the RoBERTa-GCN model.

## Models Compared
1.  **GCN Base** (`roberta_gcn_metrics.txt`)
2.  **GCN Weighted** (`roberta_gcn_w_metrics.txt`)
3.  **GCN Focal Loss** (`roberta_gcn_fl_metrics.txt`)
4.  **GCN Focal Loss (Neutral)** (`roberta_gcn_fl_neu_metrics.txt`)

## Key Findings (The "Minority Class" Breakthrough)

The most significant finding is the performance of the **Focal Loss (FL)** model on the difficult **Price** and **Packing** aspects.

### 1. Price Aspect (Critical Improvement)
| Model | Negative Precision | Negative Recall | Negative F1 |
| :--- | :--- | :--- | :--- |
| **GCN Base** | 0.00 | 0.00 | 0.00 |
| **GCN Weighted** | 0.00 | 0.00 | 0.00 |
| **GCN FL** | **0.83** | **0.91** | **0.87** |
| **GCN FL Neu** | 0.00 | 0.00 | 0.00 |
> **Impact:** The **GCN FL** model is the *only* model that successfully identifies negative price sentiment. All other models failed completely (0% recall). This is a massive improvement.

### 2. Packing Aspect
| Model | Negative Precision | Negative Recall | Negative F1 |
| :--- | :--- | :--- | :--- |
| **GCN Base** | 0.62 | 0.45 | 0.53 |
| **GCN Weighted** | 0.62 | 0.45 | 0.53 |
| **GCN FL** | **0.85** | **0.85** | **0.85** |
| **GCN FL Neu** | 0.64 | 0.64 | 0.64 |
> **Impact:** **GCN FL** again dominates, nearly doubling the recall for negative packing reviews compared to the baseline.

## Trade-offs: The "Staying Power" Regression
While **GCN FL** excels at minority classes, it struggles with the positive class in **Staying Power**.

| Model | Staying Power (Pos Recall) | Staying Power (Weighted F1) |
| :--- | :--- | :--- |
| **GCN Base** | 0.83 | **0.79** |
| **GCN Weighted** | 0.83 | **0.79** |
| **GCN FL** | 0.53 | 0.69 |
| **GCN FL Neu** | 0.74 | **0.79** |
> **Note:** The **GCN FL** model drops significantly in Staying Power performance. The **GCN FL Neu** variant recovers this, but at the cost of losing the Price/Packing gains.

## Overall Comparison Matrix (Weighted F1)

| Aspect | GCN Base | GCN Weighted | GCN FL | GCN FL Neu | Best |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Staying Power | 0.79 | 0.79 | 0.69 | 0.79 | Base/W/Neu |
| Texture | 0.85 | 0.85 | 0.83 | 0.85 | Base/W/Neu |
| Smell | 0.90 | 0.90 | **0.91** | 0.90 | FL |
| Price | 0.96 | 0.96 | 0.96 | 0.92 | FL (by far on Neg) |
| Colour | 0.89 | 0.89 | **0.90** | **0.90** | FL / FL Neu |
| Shipping | 0.93 | 0.93 | 0.93 | 0.93 | Tie |
| Packing | 0.96 | 0.96 | **0.97** | 0.91 | FL |

## Final Recommendation

**Best Model: `roberta_gcn_fl` (Focal Loss)**

**Why?**
Although it has lower performance on "Staying Power", the **Focal Loss** model is the only one that effectively solves the **Data Imbalance** problem.
- It achieves **91% Recall** on Negative Price (vs 0% for others).
- It achieves **85% Recall** on Negative Packing (vs 45% for others).

If identifying negative customer sentiment is a priority (which usually is in sentiment analysis), **GCN FL** is superior. If you strictly need balanced metrics across all categories and can accept 0% detection on rare negative cases, then the Base or Weighted models are safer.

**Recommendation:** Use **GCN FL** and investigate why "Staying Power" positive class degraded (potentially tune the focal loss gamma or alpha specifically for that class).
