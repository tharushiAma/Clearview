# Model Training and Methodology Overview

This document provides a detailed technical overview of the model training process and the innovative methodologies implemented to achieve high-performance results for the Cosmetic Multi-Aspect Sentiment Analysis project.

## 1. Project Objective
The primary goal is **Class Balanced Aspect-Based Mixed Sentiment Resolution with Explainable AI (XAI)**. The system is designed to identify sentiments (Positive, Negative, Neutral) across 7 distinct aspects of cosmetic product reviews, effectively handling mixed sentiments and extreme class imbalances.

---

## 2. Core Methodology: Architecture

The model utilizes a sophisticated hybrid architecture:
- **Encoder**: `roberta-base` provides deep contextualized token embeddings.
- **Aspect-Aware Attention**: A specialized multi-head attention mechanism that focuses on tokens relevant to a specific aspect (e.g., focusing on "smells like vanilla" when analyzing requested aspect `smell`).
- **Aspect-Oriented Dependency GCN**:
  - Utilizes syntactic dependency parse trees (spaCy).
  - Employs a **graph convolutional network** to propagate sentiment information along grammatical paths.
  - Features an **Aspect-Oriented Gating** mechanism to filter messages relevant to the target aspect.

---

## 3. Mixed Sentiment Resolution: How it Works

Addressing sentences like *"The shipping was slow but the packing was elegant"* requires more than just keyword detection. We resolve this through:

### A. Syntactic Dependency Mapping
The model doesn't just see a sequence of words; it understands the structural relationship between tokens. By extracting the **Dependency Tree**, the model knows that *"slow"* is grammatically linked to *"shipping"* and *"elegant"* is linked to *"packing"*.

### B. Aspect-Oriented Gating (AOG)
When analyzing a specific aspect (e.g., `shipping`), the AOG mechanism acts as a "filter" in the GCN layers:
1. **Selection**: It selectively allows the flow of information from context words (like "slow") that have a direct or high-impact syntactic dependency with the target aspect.
2. **Isolation**: It blocks or suppresses "noise" from contradictory opinion words associated with other aspects (like "elegant").
3. **Representation**: This results in an aspect-specific representation that captures the correct polarity even when multiple sentiments are present in the same review.

---

## 4. Data-Driven Class Imbalance Mitigation

A critical challenge was the extreme imbalance (up to 185:1) in aspects like `price` and `packing`. We addressed this through a multi-pronged strategy:

### A. Synthetic Data Integration
- Augmented the minority classes (Negative/Neutral) using synthetic data generation.
- **Result**: Drastically improved imbalance ratios. For instance, `price` imbalance was reduced from **174:1** to **11:1**, while `packing` improved from **185:1** to **12:1**.

### B. Hybrid Loss Function
We implemented a weighted combination of three specialized loss functions:
1. **Focal Loss**: Focuses on "hard" examples by down-weighting easy-to-classify majorities.
2. **Class-Balanced Loss**: Re-weights samples based on the *effective* number of samples in a class.
3. **Dice Loss**: Directly optimizes for the F1-score to handle overlapping distributions.

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Device** | NVIDIA GeForce RTX 4060 Laptop GPU |
| **Batch Size** | 16 |
| **Learning Rate** | 2.0e-5 |
| **Optimizer** | AdamW |
| **Warmup Steps** | 500 |
| **Epochs** | 30 (with Early Stopping) |
| **Mixed Precision** | Enabled (AMP) |

---

## 5. Performance Results (Final Test Set)

The model achieved state-of-the-art performance for this specialized domain:

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **92.14%** |
| **Overall Macro-F1** | **0.7981** |
| **Overall Weighted-F1** | **0.9242** |
| **MCC (Matthews Correlation)** | **0.7842** |

### Per-Aspect Macro-F1 Highlights:
- **Shipping**: **0.8507**
- **Stayingpower**: **0.7920**
- **Colour**: **0.7791**
- **Texture**: **0.7726**
- **Smell**: **0.7381**
- **Packing**: **0.5989** (significant improvement!)
- **Price**: **0.4944** (stable despite data scarcity)

---

## 6. Explainable AI (XAI) Integration

To ensure model transparency and trust, we integrated three interpretability layers:
1. **Attention Visualizer**: Real-time heatmaps showing which tokens influenced the prediction.
2. **LIME**: Local perturbations to identify significant word-level contributions.
3. **Integrated Gradients (IG)**: Theoretically grounded attribution of predictions to specific input features.

---
*Created on 2026-02-23 for the Thesis Defense Documentation.*
