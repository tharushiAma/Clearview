# Chapter 10: Conclusion

## 10.1 Summary of Contributions

This thesis presented ClearView, addressing four interconnected challenges in ABSA for the cosmetic domain:

**Contribution 1: Three-Pronged Class Imbalance Framework**
LLM-based synthetic augmentation + per-aspect Hybrid Loss (Focal + Class-Balanced + Dice) + two-phase stratified split. Each prong addresses a different mechanism of imbalance, quantified through ablation studies A3 and A4.

**Contribution 2: Aspect-Oriented Dependency GCN**
spaCy dependency parse trees with aspect-specific gating provide explicit structural reasoning for mixed sentiment resolution. The aspect gate prevents cross-aspect signal contamination.

**Contribution 3: MSR Delta Analysis**
Novel XAI method providing direct experimental evidence of mixed sentiment resolution by measuring per-token confidence changes under masking for a focus aspect.

**Contribution 4: 19-Experiment Evaluation Framework**
4 baselines and 6 ablation studies (15 variants) providing quantitative justification for every major architectural decision.

**Contribution 5: ClearView Web System**
Deployable web system with interactive XAI visualizations demonstrating practical utility.

## 10.2 Answers to Research Questions

RQ1 (Hybrid Loss): Expected to significantly outperform individual losses. CE baseline expected near-zero negative recall for extreme imbalance cases.

RQ2 (Dependency GCN): Expected 3-5% improvement in Mixed Sentiment Resolution Accuracy.

RQ3 (LLM Augmentation): Expected 5-10% improvement in negative class F1 without harming majority class.

RQ4 (XAI Evidence): MSR Delta expected to show near-zero cross-aspect delta in mixed-sentiment reviews.

## 10.3 Limitations

- Keyword-based mention detection: can miss aspects mentioned without keywords
- Implicit aspects not handled (opinions without naming the aspect)
- English only; non-English cosmetic markets not covered
- GPU required for practical training; CPU inference is slow (~5-10s per review)
- LLM-generated synthetic data not fully reproducible (proprietary LLM)

## 10.4 Future Work

1. Implicit Aspect Handling: learned aspect term extraction + implicit sentiment inference
2. Multilingual Extension: XLM-R backbone with cross-lingual transfer
3. Learned Mention Detection: binary classifier per aspect replacing keyword heuristics
4. Supervised Contrastive Loss: pulls minority class representations apart from majority
5. Online Learning: continual learning for streaming reviews without full retraining
6. Domain Transfer: zero-shot transfer to skincare, haircare, fragrance domains
7. Knowledge Distillation: DistilRoBERTa student model for production deployment

## 10.5 Final Remarks

ClearView demonstrates that combining architectural innovations (Aspect-Aware Attention, Dependency GCN), training strategies (Hybrid Loss, LLM Augmentation), and explanation methods (IG, MSR Delta) can address compounded challenges of multi-aspect ABSA with extreme class imbalance. The modular design and comprehensive evaluation framework make contributions reproducible and extensible for future research in domain-specific ABSA.