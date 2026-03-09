# Chapter 1: Introduction

## 1.1 Background and Motivation

The rapid growth of e-commerce platforms has generated massive volumes of user-generated product reviews. In the cosmetic industry, consumers increasingly rely on peer reviews to make purchasing decisions. Unlike traditional sentiment analysis which assigns a single overall polarity, real-world reviews frequently express multiple and often contradictory sentiments about different product attributes.

Consider the review: The lipstick colour is absolutely stunning and stays on all day, but the smell is overpowering and the price is far too high. A conventional sentiment classifier might label this as mixed, losing critical information. Aspect-Based Sentiment Analysis (ABSA) addresses this by identifying sentiments expressed towards specific product attributes.

Two key challenges motivate this research:

**Challenge 1 - Extreme Class Imbalance.** Cosmetic reviews overwhelmingly express positive sentiment. The positive-to-negative ratio for the price aspect reaches 132:1 and packing 71:1 in the raw dataset. Standard cross-entropy loss causes models to collapse to predicting positive for nearly every sample, achieving high accuracy while ignoring the commercially important minority classes.

**Challenge 2 - Mixed Sentiment Resolution.** When a review expresses positive sentiment about one aspect and negative about another, signals become entangled in the shared representation. Correctly separating which opinion words attach to which aspect requires structural syntactic reasoning that pooled representations cannot provide.

## 1.2 Problem Statement

This research addresses: How can a deep learning system simultaneously perform accurate multi-aspect sentiment classification across 7 cosmetic product aspects, handle severe class imbalance (up to 132:1 ratios), resolve mixed sentiments within individual reviews, and provide interpretable explanations for its predictions?

## 1.3 Research Objectives

- **O1 - Multi-Aspect Classification:** Simultaneously classify sentiment for all 7 aspects (stayingpower, texture, smell, price, colour, shipping, packing).
- **O2 - Class Imbalance Handling:** A three-pronged strategy using LLM augmentation, Hybrid Loss (Focal + Class-Balanced + Dice), and a two-phase stratified split.
- **O3 - Mixed Sentiment Resolution:** An Aspect-Oriented Dependency GCN with aspect-gated message passing.
- **O4 - Explainability:** Attention, LIME, SHAP, Integrated Gradients, and novel MSR Delta analysis.
- **O5 - System Deployment:** A web-based ClearView demonstration with real-time ABSA and interactive XAI.

## 1.4 Research Questions

- **RQ1:** To what extent does a Hybrid Loss (Focal + Class-Balanced + Dice) outperform individual loss functions for extreme class imbalance in ABSA?
- **RQ2:** Does an Aspect-Oriented Dependency GCN improve mixed sentiment resolution accuracy compared to transformer attention alone?
- **RQ3:** What is the contribution of LLM-generated synthetic augmentation to minority class recall, and does it affect majority class performance?
- **RQ4:** Can Integrated Gradients and MSR Delta provide evidence that the model correctly attributes sentiment to the appropriate aspect in mixed-sentiment reviews?

## 1.5 Original Contributions

1. **Three-pronged class imbalance framework** combining LLM synthetic augmentation, per-aspect Hybrid Loss configuration, and a two-phase stratified split guaranteeing minority class representation.
2. **Aspect-Oriented Dependency GCN** integrating spaCy parse trees with aspect-specific gating into a RoBERTa-based ABSA model.
3. **MSR Delta analysis** - a novel XAI method demonstrating mixed sentiment resolution by measuring per-token confidence changes for a focus aspect under token masking.
4. **Comprehensive ablation study** across 6 components with 4 baselines (19 experiments total).
5. **ClearView web system** - open-source deployable ABSA demo with interactive XAI.

## 1.6 Scope and Limitations

**Scope:** English-language cosmetic reviews, 7 predefined aspects, 3 sentiment classes (positive, negative, neutral).

**Limitations:** Keyword-based mention detection, no implicit aspect handling, cross-domain generalization not evaluated.

## 1.7 Thesis Structure

| Chapter | Content |
|---------|---------|
| 2 | Literature review: ABSA, class imbalance, XAI, cosmetic NLP |
| 3 | Research design and methodology overview |
| 4 | Data pipeline: collection, preprocessing, augmentation |
| 5 | Model architecture: RoBERTa + Aspect Attention + Dependency GCN |
| 6 | Class imbalance handling: loss functions and split strategy |
| 7 | Explainability: LIME, SHAP, Integrated Gradients, MSR Delta |
| 8 | Experiments: baselines and ablation studies |
| 9 | Results and discussion |
| 10 | Conclusion, limitations, future work |
