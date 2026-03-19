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

**RQ1 (Hybrid Loss):** Partially confirmed. The Focal+CB combination (A7: Macro-F1 = 0.7944) outperforms all individual losses. Dice Loss alone collapsed to Macro-F1 = 0.2926, confirming it cannot be used in isolation under extreme imbalance. CE and CB-only tied at 0.7911, with Focal+CB providing the winning +0.33% margin through hard-example focusing combined with principled reweighting. Price and packing negative classes remained near-zero F1 regardless of loss choice due to insufficient training samples (9 each).

**RQ2 (Dependency GCN):** Confirmed. Removing GCN caused a −9.9% Macro-F1 drop (0.7856 → 0.6863), the largest single-component ablation effect. Aspect-level accuracy on mixed reviews (87.55%) demonstrates that the GCN's aspect-gating mechanism successfully prevents cross-aspect signal contamination.

**RQ3 (LLM Augmentation):** Partially confirmed. Overall Macro-F1 impact was negligible (−0.16%), as the most extreme imbalance cases (price: 9 negative samples) remain too sparse even after augmentation. Localised improvements were observed for moderately imbalanced aspects. The fundamental bottleneck is original data scarcity, not augmentation volume.

**RQ4 (XAI Evidence):** Confirmed. MSR Delta analysis shows orthogonal token attribution patterns across focus aspects in mixed-sentiment reviews — tokens contributing to colour prediction carry near-zero delta for smell prediction and vice versa. This provides direct interpretable evidence of aspect-specific signal separation by the GCN.

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
