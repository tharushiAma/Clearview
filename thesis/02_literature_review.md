# Chapter 2: Literature Review

## 2.1 Aspect-Based Sentiment Analysis (ABSA)

### 2.1.1 Problem Definition
ABSA identifies sentiment polarities towards specific product attributes. Formalized by Hu and Liu (2004) and standardized through SemEval shared tasks (Pontiki et al., 2014, 2015, 2016).

### 2.1.2 Traditional Approaches
Lexicon-based methods (Ding et al., 2008) and SVM classifiers with TF-IDF features established baselines but struggled with informal language and domain adaptation.

### 2.1.3 Neural ABSA
Attention-based models (Wang et al., 2016; Tang et al., 2016) introduced aspect-aware representations. Graph Convolutional Networks on dependency parse trees (Zhang et al., 2019 ASGCN; Wang et al., 2020 RGAT) showed syntactic structure provides complementary signals, particularly for multi-opinion sentences.

### 2.1.4 Transformer-Based ABSA
BERT-based models (Sun et al., 2019; Xu et al., 2019) achieved state-of-the-art. RoBERTa (Liu et al., 2019) outperforms BERT due to dynamic masking, more data, and removal of NSP objective. Key variants:
- SentiBERT (Yin et al., 2020): Sentiment-aware pre-training with tree attention
- BERT-ADA (Rietzler et al., 2020): Domain-adaptive pre-training
- DualGCN (Li et al., 2021): Syntactic and semantic graphs with BERT

### 2.1.5 Multi-Aspect Simultaneous Classification
Most work processes one aspect per forward pass. Multi-aspect joint classification is underexplored. This work uses learnable aspect query embeddings in MHA for simultaneous 7-aspect classification in a single forward pass.

## 2.2 Class Imbalance in Sentiment Analysis

### 2.2.1 Problem Severity
At 132:1 ratio (price aspect), standard CE training collapses to majority-class prediction. High accuracy but near-zero minority recall - pathological for practical systems.

### 2.2.2 Data-Level Strategies
SMOTE (Chawla et al., 2002) interpolates minority samples in feature space - incoherent for text at extreme ratios.

LLM-based Augmentation (Wang et al., 2021; Moller et al., 2023) generates domain-specific synthetic samples. Semantically coherent unlike SMOTE interpolation. Justified here because at ratios up to 132:1, only LLM augmentation generates sufficient coherent minority samples.

### 2.2.3 Algorithm-Level Strategies
Focal Loss (Lin et al., 2017): FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t). Down-weights easy examples, focusing training on hard minority samples.

Class-Balanced Loss (Cui et al., 2019): w_c = (1-beta)/(1-beta^n_c). More principled than inverse-frequency weighting.

Dice Loss (Li et al., 2020): Directly optimizes F1-equivalent Dice coefficient.

Research Gap: No prior work combines all three in an aspect-specific hybrid adapting gamma per-aspect based on individual imbalance severity.

## 2.3 Graph Convolutional Networks for NLP

Zhang et al. (2019) ASGCN showed syntactic distance captures opinion-aspect attachment. In "Great colour but awful smell," "great" modifies "colour" while "awful" modifies "smell" - dependency structure captures this directly.

Aspect-Oriented Gating: Standard GCN aggregates all neighbor messages uniformly. The proposed gate = sigmoid(W_g * aspect_emb) controls which syntactic relationships are relevant per aspect, preventing cross-aspect signal contamination.

## 2.4 Explainability in NLP

Attention weights: Not reliable as feature importance (Wiegreffe and Pinter, 2019). Useful as first-order visualization.

LIME (Ribeiro et al., 2016): Local linear approximation via input perturbation. Model-agnostic.

SHAP (Lundberg and Lee, 2017): Shapley values from game theory. Satisfies efficiency, symmetry, dummy axioms. Globally consistent.

Integrated Gradients (Sundararajan et al., 2017): Satisfies Sensitivity and Implementation Invariance. Completeness axiom: attributions sum to F(x) - F(x_baseline). Most theoretically rigorous - preferred for research claims.

MSR Delta (Novel Contribution): No prior work proposes explanation specifically for mixed sentiment resolution. MSR Delta measures per-token confidence changes under masking for a focus aspect, providing direct experimental evidence of aspect signal separation.

## 2.5 Cosmetic Domain Sentiment Analysis

Prior work (Sezgin and Akalin, 2020; Yusof et al., 2019; Kobia and Limbasiya, 2020; Nandini et al., 2021) addresses binary or coarse-grained sentiment without extreme imbalance handling, mixed sentiment resolution, or multi-method explainability.

## 2.6 Research Positioning

| Dimension | Prior Work | This Work |
|-----------|-----------|-----------|
| Aspects | Single aspect per pass | 7 simultaneously |
| Class Imbalance | CE or single strategy | Three-pronged hybrid |
| Mixed Sentiment | Implicit in attention | Explicit GCN + MSR Delta |
| Explainability | Single method | 5 complementary methods |
| Domain | General/restaurant/laptop | Cosmetic (domain-specific) |
