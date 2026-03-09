# Chapter 3: Research Design and Methodology

## 3.1 Research Approach

This research adopts a positivist, quantitative paradigm. The methodology is experimental with controlled ablation studies providing causal evidence for each design decision. It follows applied design science: novel computational artifacts (model architecture, loss functions, augmentation pipeline, XAI methods) are designed, implemented, and evaluated.

## 3.2 Methodology Overview

```
Phase 1: Data Pipeline
  - Data collection and annotation audit
  - Text cleaning (Unicode normalization, HTML removal, garbled detection)
  - Two-phase stratified split (rare-class guarantee)
  - LLM synthetic augmentation for minority classes

Phase 2: Model Design
  - RoBERTa-base backbone
  - Aspect-Aware Attention (learnable aspect queries in MHA)
  - Aspect-Oriented Dependency GCN
  - Per-aspect classifier heads (7 x Linear 768->384->3)

Phase 3: Class Imbalance Strategy
  - Per-aspect Hybrid Loss configuration (aspect-specific gamma/beta)
  - Integration with training pipeline

Phase 4: Training and Evaluation
  - AdamW optimizer with linear warmup schedule
  - Mixed precision (AMP), gradient clipping max_norm=1.0
  - Early stopping on validation Macro-F1 (patience=5)
  - Test evaluation with AspectSentimentEvaluator + MixedSentimentEvaluator

Phase 5: Ablation Study and XAI
  - 19-experiment evaluation framework
  - Multi-method XAI (Attention, LIME, SHAP, IG, MSR Delta)
```

## 3.3 Design Justifications

### 3.3.1 Why RoBERTa-base over BERT-base?
- 160GB training data vs 16GB for BERT
- Dynamic masking prevents memorization of masking patterns
- No Next Sentence Prediction objective (shown to degrade performance by Liu et al., 2019)
- Consistently outperforms BERT on GLUE, SQuAD, and sentiment benchmarks

Alternative considered: DistilRoBERTa - rejected because minority class recall gain from full model outweighs compute cost.

### 3.3.2 Why Aspect Query Embeddings rather than CLS pooling?
Standard ABSA with CLS: processes one aspect per forward pass (7x inference cost), attends globally without aspect conditioning.

Proposed approach - aspect embeddings as MHA queries:
1. All 7 aspects in one forward pass (7x faster inference)
2. Aspect-conditioned token attention (different aspects attend to different tokens)
3. Interpretable attention weights per aspect

Ablation A2 quantifies this choice.

### 3.3.3 Why a Dependency GCN for Mixed Sentiment Resolution?
Transformer attention is permutation-invariant and global. In "Great colour but awful smell," both "great" and "awful" receive attention from both aspect terms without structural constraints.

Dependency parse trees provide explicit structural constraints: "great" modifies "colour" (amod edge), "awful" modifies "smell." The GCN propagates only syntactically adjacent messages, guided by the aspect gate.

Alternative: Relative position encoding - rejected because syntactic distance more directly captures opinion-aspect attachment.

Ablation A1 quantifies the GCN contribution.

### 3.3.4 Why Three-Pronged Imbalance Strategy?
Each component addresses a different mechanism:
- LLM Augmentation: data-level, increases minority count before training
- Hybrid Loss: algorithm-level, corrects remaining imbalance during gradient descent
- Two-Phase Split: evaluation-level, ensures minority representation in val/test

No single strategy is sufficient at 132:1 ratio. Ablations A3 and A4 quantify each component.

### 3.3.5 Why Macro-F1 as Primary Metric?
For 132:1 imbalance, accuracy is misleading (predicting only positive achieves >95% accuracy on price). Macro-F1 treats all classes equally, standard for imbalanced classification (He and Garcia, 2009).

## 3.4 Experimental Protocol

### 3.4.1 Data Splits
- Train: ~80% original + synthetic augmentation = 10,050 samples
- Val: ~10% stratified (two-phase guarantee)
- Test: ~10% stratified (held out until final evaluation)

### 3.4.2 Hyperparameter Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 2e-5 | Standard for RoBERTa fine-tuning |
| Batch Size | 16 | RTX 4060 VRAM constraint |
| Warmup Steps | 500 | Prevents early LR spike damaging pre-trained weights |
| Dropout | 0.1 | Standard; higher degrades minority class performance |
| GCN Layers | 2 | More causes oversmoothing on short review graphs |
| Attention Heads | 8 | Standard for 768-dim |
| Max Seq Length | 128 | Covers 99% of reviews without padding waste |
| Epochs | 30 | With early stopping (patience=5) |

All hyperparameters selected via validation Macro-F1. Test set not consulted during development.

### 3.4.3 Reproducibility
Fixed random seed (42) for data splitting, model initialization, and training. Full config in ml-research/configs/config.yaml. Experiment runner provides CLI reproducibility.

## 3.5 Ethical Considerations

Data: Public e-commerce reviews. No personal identifiers beyond review text.

Synthetic Augmentation: LLM-generated content is labeled as synthetic and excluded from evaluation sets.

Bias: English-only, reflects demographics of beauty platform users. Cross-demographic and cross-language generalization not evaluated.
