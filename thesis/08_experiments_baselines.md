# Chapter 8: Experiments - Baselines and Ablation Studies

## 8.1 Experiment Framework Overview

All experiments use the same data splits (train_augmented.csv, val.csv, test.csv) and evaluation code (AspectSentimentEvaluator, MixedSentimentEvaluator). Experiments are implemented in src/experiments/experiment_runner.py and results automatically analyzed by results_analyzer.py.

Total experiments: 19 (4 baselines + 15 ablation variants)

Primary metric: Validation Macro-F1 (aggregated across all 7 aspects)
Secondary: Per-class F1 for negative (minority) class, Mixed Sentiment Resolution Accuracy

## 8.2 Baseline Models (4 experiments)

### B1: PlainRoBERTa
Architecture: RoBERTa-base with CLS token pooling, no aspect awareness, standard cross-entropy loss.
Purpose: Establishes the baseline performance of RoBERTa without any of the proposed innovations.
Expected weakness: No aspect-specific attention, treats all aspects identically, no imbalance handling.

### B2: RoBERTa+CE
Architecture: Full proposed architecture (RoBERTa + Aspect Attention + Dependency GCN + per-aspect heads) but with plain cross-entropy loss instead of Hybrid Loss.
Purpose: Isolates the contribution of the loss function while keeping the architecture constant.
Expected weakness: High accuracy but poor minority class recall (price-negative, packing-negative especially).

### B3: BERTBaseline
Architecture: BERT-base-uncased + CLS pooling + per-aspect heads + cross-entropy loss.
Purpose: Provides a comparison point between BERT and RoBERTa under equivalent architecture.
Implementation: Uses transformers.BertModel as backbone in src/experiments/baseline_models.py.

### B4: TF-IDF + SVM
Architecture: TF-IDF features (unigram + bigram, max 5000 features) + LinearSVC per aspect (7 independent classifiers).
Purpose: Classical machine learning baseline demonstrating the improvement from pre-trained transformers.
Implementation: sklearn TfidfVectorizer + LinearSVC with class_weight=balanced in src/experiments/baseline_models.py.

## 8.3 Ablation Studies (6 studies, 15 variants)

### Ablation A1: Dependency GCN
Component tested: Aspect-Oriented Dependency GCN
Variants:
  - A1_with_gcn: Full model (default)
  - A1_without_gcn: use_dependency_gcn=False (CLS-pooled GCN output, no dependency graph)
Research question: Does the GCN improve mixed sentiment resolution accuracy?
Key metric: Mixed Sentiment Resolution Accuracy, per-aspect F1 for aspects likely to appear together in reviews.

### Ablation A2: Aspect Attention
Component tested: Aspect-Aware Multi-Head Attention
Variants:
  - A2_with_attention: Full model (default)
  - A2_cls_pooling: use_aspect_attention=False (CLS token + aspect ID as learned offset)
Research question: Does aspect-conditioned attention improve over CLS pooling for aspect-specific prediction?
Key metric: Overall Macro-F1, per-aspect F1 variance (consistent improvement should appear across all aspects).

### Ablation A3: Loss Function
Component tested: Hybrid Loss composition
Variants:
  - A3_hybrid: Focal + CB + Dice (default)
  - A3_focal_only: Only Focal Loss
  - A3_cb_only: Only Class-Balanced Loss
  - A3_dice_only: Only Dice Loss
  - A3_ce: Plain cross-entropy (no imbalance handling)
Research question: Which loss combination best handles extreme class imbalance? Are the components complementary?
Key metric: Minority class (negative) F1 for price and packing aspects.

### Ablation A4: Data Augmentation
Component tested: LLM synthetic augmentation
Variants:
  - A4_with_aug: Training on train_augmented.csv (10,050 samples, default)
  - A4_without_aug: Training on train.csv (~8,000 samples, no synthetic data)
Research question: How much does LLM augmentation contribute to minority class recall? Does it harm majority class performance?
Key metric: Per-class F1 for negative class across all aspects; Macro-F1 to check majority class stability.

### Ablation A5: Classifier Head
Component tested: Per-aspect vs. shared classifier
Variants:
  - A5_per_aspect: 7 independent classifier heads (default)
  - A5_shared: use_shared_classifier=True (single classifier for all aspects)
Research question: Do different aspects benefit from dedicated classifier heads?
Key metric: Per-aspect F1 variance; aspects with highly different linguistic patterns expected to show larger degradation with shared head.

### Ablation A6: Text Preprocessing
Component tested: Text cleaning pipeline
Variants:
  - A6_with_preprocessing: Full cleaning pipeline (default)
  - A6_raw_text: Raw text without cleaning (HTML, garbled tokens, repeated punctuation retained)
Research question: Does the cleaning pipeline improve model performance?
Key metric: Overall Macro-F1, qualitative inspection of attention on cleaned vs. uncleaned inputs.

## 8.4 Running Experiments

All experiments accessible via CLI:

  # List all 19 experiments
  python src/experiments/experiment_runner.py --list

  # Run all baselines
  python src/experiments/experiment_runner.py --group baselines

  # Run specific ablation
  python src/experiments/experiment_runner.py --experiment A3_focal_only

  # Run all experiments
  python src/experiments/experiment_runner.py --group all

  # Generate analysis report
  python src/experiments/results_analyzer.py

Results are saved incrementally to results/experiments/all_results.json. The analyzer generates Markdown tables, LaTeX tables (for thesis inclusion), and bar charts.

## 8.5 Statistical Validation

With multiple experiments comparing variants, statistical significance testing is applied:
- Bootstrap confidence intervals (1000 iterations) for Macro-F1 estimates
- McNemar test for paired accuracy comparisons between model variants
- Threshold: p < 0.05 for statistical significance

Multiple comparison correction (Bonferroni) applied when comparing all ablation variants simultaneously.

## 8.6 Expected Experimental Outcomes

Based on the literature and architecture design:

| Ablation | Expected Primary Finding |
|---------|------------------------|
| A1 (GCN) | +3-5% mixed sentiment resolution accuracy with GCN |
| A2 (Attention) | +2-4% Macro-F1 with aspect attention vs CLS |
| A3 (Loss) | Hybrid significantly outperforms single-loss; CE shows near-zero negative recall |
| A4 (Augmentation) | +5-10% negative class F1 with augmentation; minimal majority class impact |
| A5 (Shared head) | Per-aspect heads better, especially for texture vs shipping (linguistically different) |
| A6 (Preprocessing) | Minor improvement (+1-2%), larger impact on aspects with domain vocabulary |
