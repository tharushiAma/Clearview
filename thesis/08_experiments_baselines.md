# Chapter 8: Experiments - Baselines and Ablation Studies

## 8.1 Experiment Framework Overview

All experiments use the same data splits (train_augmented.csv, val.csv, test.csv) and evaluation code (AspectSentimentEvaluator, MixedSentimentEvaluator). Experiments are implemented in src/experiments/experiment_runner.py and results automatically analyzed by results_analyzer.py. All experiments have been completed and results are stored in results/experiments/all_results.json and all_results.csv.

Total experiments run: 16 (4 baselines + 12 ablation variants, including 2 A7 loss weight variants)

Primary metric: Test Macro-F1 (aggregated across all 7 aspects)
Secondary: Per-class F1 for negative (minority) class, Mixed Sentiment Resolution Accuracy

## 8.2 Baseline Models (4 experiments)

### B1: PlainRoBERTa
Architecture: RoBERTa-base with CLS token pooling, no aspect awareness, standard cross-entropy loss.
Purpose: Establishes the baseline performance of RoBERTa without any of the proposed innovations.
Expected weakness: No aspect-specific attention, treats all aspects identically, no imbalance handling.

### B2: DistilBERTBaseline
Architecture: DistilBERT-base-uncased + CLS pooling + per-aspect heads + cross-entropy loss.
Purpose: Provides a lightweight transformer baseline comparison, demonstrating the trade-off between model size and performance.
Implementation: Uses transformers.DistilBertModel as backbone in src/experiments/baseline_models.py.

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

### Ablation A6: Mixed Sentiment Resolution (MSR) Evaluation
Component tested: MSR-specific evaluation of full model vs. GCN-ablated model
Variants:
  - A6_msr_with_gcn: Full model evaluated with MixedSentimentEvaluator (same as A1_full_model)
  - A6_msr_no_gcn: No-GCN model evaluated with MixedSentimentEvaluator (same as A1_no_gcn)
Research question: Does the Dependency GCN improve Mixed Sentiment Resolution accuracy specifically?
Key metric: MSR review-level accuracy (% of mixed reviews with all aspects correct), MSR aspect-level accuracy.

**Note on A6 vs. planned Text Preprocessing ablation:** The originally planned A6 (text preprocessing comparison) was not executed as a separate experiment, as the cleaning pipeline is a required pre-processing step shared by all models. The cleaning benefit is inherent to the baseline results. Instead, A6 was re-purposed to capture MSR-specific metrics for the two GCN variants, consolidating the MSR evaluation evidence.

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

## 8.6 Actual Experimental Outcomes Summary

All experiments completed. Summary of key findings vs expectations:

| Ablation | Expected | Actual Finding |
|---------|----------|----------------|
| A1 (GCN) | +3-5% MSR accuracy with GCN | MSR accuracy identical (66.56%) with/without GCN; Aspect-Aware Attention is the MSR driver |
| A2 (Attention) | +2-4% Macro-F1 with aspect attention | +24.78% Macro-F1; +46.65% MSR aspect-level accuracy; 0→66.56% review-level MSR |
| A3 (Loss) | Hybrid outperforms single-loss | A7 (Focal+CB, Dice=0.0) is best at 0.7944; Dice alone collapses to 0.2926 |
| A4 (Augmentation) | +5-10% negative class F1 | Negligible overall effect (−0.16% Macro-F1); extreme imbalance not fixed by augmentation volume |
| A5 (Shared head) | Per-aspect heads better | +0.59% Macro-F1, +1.44% MSR review-level accuracy |
| A6 (MSR evaluation) | MSR improvement with GCN | GCN and no-GCN identical MSR; Aspect Attention is primary MSR mechanism |
