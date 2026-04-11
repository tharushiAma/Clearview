# Chapter 10: Conclusion

## 10.1 Summary of Contributions

This thesis presented ClearView, addressing four interconnected challenges in ABSA for the cosmetic domain:

**Contribution 1: Two-Pronged Class Imbalance Framework**
LLM-based synthetic augmentation + per-aspect Hybrid Loss (Focal + Class-Balanced, Dice=0.0) + two-phase stratified split. Each prong addresses a different mechanism of imbalance, quantified through ablation studies A3 and A4. The A7 configuration (Focal 1.0 + CB 0.5) achieves Macro-F1 = 0.7944, outperforming all individual loss functions.

**Contribution 2: Aspect-Aware Attention Module (primary MSR driver)**
Learnable aspect embeddings as MHA queries provide aspect-specific token selection and are the necessary condition for Mixed Sentiment Resolution. Ablation A2 shows that removing this module causes a −24.78% Macro-F1 drop and complete collapse of MSR capability (review-level accuracy 66.56% → 0%). The Dependency GCN provides complementary syntactic structure, but the Aspect-Aware Attention is the primary mechanism.

**Contribution 3: 16-Experiment Evaluation Framework**
4 baselines and 6 ablation studies providing quantitative justification for every major architectural decision. All 16 experiments completed with full results in all_results.csv.

**Contribution 4: ClearView Web System**
Deployable web system with interactive XAI visualizations demonstrating practical utility.

## 10.2 Answers to Research Questions

**RQ1 (Hybrid Loss):** Partially confirmed. The Focal+CB combination (A7: Macro-F1 = 0.7944) outperforms all individual losses. CE and CB-only tied at 0.7911, with Focal+CB providing the winning +0.33% margin through hard-example focusing combined with principled reweighting. Dice Loss alone collapses to 0.2926. Price and packing negative classes remained near-zero F1 regardless of loss choice due to insufficient training samples.

**RQ2 (Dependency GCN / MSR):** Revised finding. The Aspect-Aware Attention module — not the Dependency GCN alone — is the primary driver of Mixed Sentiment Resolution. Ablation A2 (removing Aspect Attention) causes a −24.78% Macro-F1 drop and complete review-level MSR collapse (66.56% → 0%). Ablation A1 (removing GCN, keeping Attention) shows no statistically meaningful MSR change (66.56% in both conditions). All CLS-pooled baselines (B1: 0.5731, B2: 0.5677, B3: 0.5697) achieve 0% review-level MSR accuracy, confirming that aspect-conditioned query separation is the necessary condition. The GCN provides complementary syntactic signal but is not the primary mechanism.

**RQ3 (LLM Augmentation):** Partially confirmed. Overall Macro-F1 impact was negligible (−0.16%), as the most extreme imbalance cases (price: 9 negative samples) remain too sparse even after augmentation. Localised improvements were observed for moderately imbalanced aspects. The fundamental bottleneck is original data scarcity, not augmentation volume.

**RQ4 (XAI Evidence):** Confirmed. Integrated Gradients analysis shows orthogonal token attribution patterns across focus aspects in mixed-sentiment reviews — tokens contributing to colour prediction carry near-zero attribution for smell prediction and vice versa. The 87.22% aspect-level and 66.56% review-level MSR accuracy validate the Aspect-Aware Attention's aspect-gating mechanism as the primary driver of mixed sentiment resolution.

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

ClearView demonstrates that combining architectural innovations (Aspect-Aware Attention, Dependency GCN), training strategies (Hybrid Loss, LLM Augmentation), and explanation methods (IG) can address compounded challenges of multi-aspect ABSA with extreme class imbalance. The modular design and comprehensive evaluation framework make contributions reproducible and extensible for future research in domain-specific ABSA.
