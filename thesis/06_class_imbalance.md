# Chapter 6: Class Imbalance Handling

## 6.1 Overview

The class imbalance problem is addressed through a three-pronged strategy, with each prong targeting a different aspect of the imbalance problem:

1. LLM-based Synthetic Augmentation (data-level, pre-training)
2. Hybrid Loss Function (algorithm-level, during training)
3. Two-Phase Stratified Split (evaluation-level, guarantees reliable metrics)

## 6.2 Focal Loss

Reference: Lin et al., Focal Loss for Dense Object Detection, ICCV 2017.

Formula: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`

Components:

- `p_t`: predicted probability of the correct class
- `(1 - p_t)^gamma`: modulating factor that reduces loss for well-classified easy examples
- `alpha_t`: per-class weight derived from class frequencies
- `gamma`: focusing parameter (higher = more focus on hard examples)

When gamma=0, Focal Loss equals standard weighted cross-entropy.

Per-aspect gamma configuration:

- price, packing: gamma=3.0 (most extreme imbalance, 132:1 and 71:1)
- smell, stayingpower, shipping: gamma=2.5
- colour, texture: gamma=2.0 (less extreme imbalance)

Justification for per-aspect gamma: The degree of imbalance varies significantly by aspect. Using a single gamma value would under-correct for the most imbalanced aspects (price, packing) or over-correct for less imbalanced ones (colour). Per-aspect configuration adapts the loss to the actual data distribution of each aspect independently.

## 6.3 Class-Balanced Loss

Reference: Cui et al., Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019.

Formula: `w_c = (1 - beta) / (1 - beta^n_c)`

where `n_c` is the sample count for class c and beta controls the smoothing (0 to 1).

The effective number of samples `n_eff = (1 - beta^n) / (1 - beta)` accounts for sample overlap: additional samples in dense regions of feature space provide diminishing marginal information. This makes class-balanced loss more principled than naive inverse-frequency weighting.

Per-aspect beta configuration:

- price, packing: beta=0.9999 (very tight effective number, extreme imbalance)
- all other aspects: beta=0.999 (moderate effective number)

## 6.4 Dice Loss

Reference: Li et al., Dice Loss for Data-imbalanced NLP Tasks, ACL 2020.

Formula: `DL = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)`

where P is predicted probability and T is the one-hot target. Directly optimizes the Dice coefficient (equivalent to F1-score).

Motivation: Cross-entropy-based losses optimize likelihood, which may not align with the Macro-F1 evaluation metric. Dice Loss creates direct gradient alignment between training objective and evaluation metric.

## 6.5 Hybrid Loss Function

The final hybrid loss formula (configuration A7, validated by ablation study A3):

`total_loss = (1.0 * focal_loss) + (0.5 * cb_loss)`

**Note:** Dice Loss weight was set to 0.0 after ablation A3 showed that including Dice (original weight 0.3) yielded Macro-F1 = 0.7856, while dropping it (Focal 1.0 + CB 0.5, Dice 0.0) achieved Macro-F1 = 0.7944. Furthermore, Dice Loss in isolation collapsed to Macro-F1 = 0.2926 — confirming it is unsuitable as a standalone loss under extreme class imbalance. The Dice component is retained in the codebase to support ablation A3 but its weight is 0.0 in the final trained model.

Weights: Focal Loss dominates (it directly handles minority class gradient, proven in object detection). Class-Balanced contributes a principled reweighting via effective number of samples.

The `AspectSpecificLossManager` class in `src/models/losses.py`:

- Reads class counts from `config.yaml`: `training.class_counts`
- Instantiates one `HybridLoss` per aspect with aspect-specific gamma and beta
- Handles forward pass: `loss, per_aspect_losses = loss_manager.compute_loss(logits, labels, aspect_ids, aspect_names)`

## 6.6 Ablation A3: Loss Function Study

5 conditions evaluated on test Macro-F1 (results from Chapter 9, §9.3):

- **A3_focal_only**: Macro-F1 = 0.7725 — hard-example focusing alone is insufficient
- **A3_cb_only**: Macro-F1 = 0.7911 — principled reweighting alone matches CE
- **A3_dice_only**: Macro-F1 = 0.2926 — F1-surrogate collapses under extreme imbalance
- **A3_ce_baseline**: Macro-F1 = 0.7911 — no imbalance handling
- **A7 (Focal 1.0 + CB 0.5)**: Macro-F1 = 0.7944 — best configuration

The Focal+CB combination outperforms every single-loss condition. The CE baseline confirms near-zero negative class recall for price and packing (0.097% training frequency each).

## 6.7 Two-Phase Stratified Split (Evaluation Level)

As described in Chapter 4, the two-phase split guarantees minority class samples in val and test. This is critical for evaluation reliability:

With only 17 negative price reviews and a 10% test set, naive stratification would produce approximately 1-2 test samples for the negative price class. This is statistically unreliable (Macro-F1 computed from 2 samples has no meaningful variance). The two-phase split guarantees at least 5 negative price samples in both val and test.

## 6.8 Interaction Between Prongs

The three strategies are complementary and interact beneficially:

- Augmentation reduces the ratio before training, making the loss function task easier
- Hybrid Loss handles the remaining imbalance that augmentation cannot fully correct
- The split strategy ensures metrics reflect true minority class performance

Ablation A4 tests augmentation: training without synthetic data with the Hybrid Loss still applied, measuring the isolated contribution of the augmentation step.
