# ClearView Thesis Documentation

**Title:** Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in the Cosmetic Domain
**Author:** Tharushi Amasha  |  **Year:** 2025

## Chapter Files

| File | Chapter | Description |
|------|---------|-------------|
| [01_introduction.md](01_introduction.md) | 1 | Introduction, motivation, objectives, research questions |
| [02_literature_review.md](02_literature_review.md) | 2 | Related work: ABSA, class imbalance, XAI, cosmetic NLP |
| [03_research_design.md](03_research_design.md) | 3 | Research design, methodology overview, justifications |
| [04_data_pipeline.md](04_data_pipeline.md) | 4 | Dataset, preprocessing, stratified split, LLM augmentation |
| [05_model_architecture.md](05_model_architecture.md) | 5 | RoBERTa + Aspect Attention + Dependency GCN |
| [06_class_imbalance.md](06_class_imbalance.md) | 6 | Hybrid loss (Focal + Class-Balanced + Dice) |
| [07_explainability.md](07_explainability.md) | 7 | Attention, LIME, SHAP, Integrated Gradients, MSR Delta |
| [08_experiments_baselines.md](08_experiments_baselines.md) | 8 | 4 baselines + 6 ablation studies (19 experiments) |
| [09_results_analysis.md](09_results_analysis.md) | 9 | Results, discussion, statistical analysis |
| [10_conclusion.md](10_conclusion.md) | 10 | Conclusion, limitations, future work |
| [REFERENCES.md](REFERENCES.md) | Refs | Full bibliography |

## Key Contributions

1. Three-pronged class imbalance framework (LLM augmentation + Hybrid Loss + two-phase split)
2. Aspect-Oriented Dependency GCN with aspect-specific gating
3. MSR Delta analysis - novel XAI for mixed sentiment resolution evidence
4. 19-experiment ablation study (6 component studies + 4 baselines)
5. ClearView web system - deployable ABSA demo with interactive XAI

## Dataset Summary

| Aspect | Pre-aug Neg:Pos | Post-aug Neg:Pos |
|--------|-----------------|-----------------|
| Price | 1:132 | ~1:11 |
| Packing | 1:71 | ~1:12 |
| Smell | 1:17 | ~1:6 |
| Colour | 1:11 | ~1:5 |
| Texture | 1:8 | ~1:4 |
| Stayingpower | 1:6 | ~1:3 |
| Shipping | 1:9 | ~1:4 |

Total training samples (after augmentation): 10,050

## Model at a Glance

- Backbone: RoBERTa-base (12 layers, 768-dim, 125M params)
- Aspect Attention: 8-head MHA, learnable aspect embeddings 7x768
- Dependency GCN: 2-layer aspect-gated GCN on spaCy parse trees
- Classifiers: 7 aspect-specific heads (768->384->3)
- Total: ~132M parameters  |  Device: NVIDIA RTX 4060 Laptop
