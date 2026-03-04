"""
ablation_configs.py
Generates config overrides for each ablation study variant.

Each function returns a deep-copied, modified version of the base config dict,
paired with a human-readable experiment name and description.

Ablations:
  A1  — GCN component on/off
  A2  — Aspect attention vs CLS pooling
  A3  — Loss function variants (Hybrid / Focal / CB / CE)
  A4  — Data augmentation on/off
  A5  — Aspect-specific vs shared classifier
  A6  — Text preprocessing on/off
"""

import copy
from typing import Dict, List, Tuple


ExperimentSpec = Tuple[str, str, dict]  # (experiment_id, description, config_override)


def get_all_ablation_specs(base_config: dict) -> List[ExperimentSpec]:
    """
    Returns all ablation experiment specs as a list of:
        (experiment_id, description, modified_config)

    Args:
        base_config: The base config dict (from config.yaml)

    Returns:
        List of (id, description, config) tuples
    """
    specs = []
    specs.extend(ablation_1_gcn(base_config))
    specs.extend(ablation_2_aspect_attention(base_config))
    specs.extend(ablation_3_loss_function(base_config))
    specs.extend(ablation_4_augmentation(base_config))
    specs.extend(ablation_5_classifier_head(base_config))
    specs.extend(ablation_6_preprocessing(base_config))
    return specs


# ─── Ablation 1: GCN Component ───────────────────────────────────────────────
def ablation_1_gcn(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests whether the Dependency GCN helps.
    Hypothesis: GCN captures syntactic relationships useful for mixed sentiment.
    """
    full = copy.deepcopy(base_config)
    full['experiment']['name'] = 'A1_full_model'

    no_gcn = copy.deepcopy(base_config)
    no_gcn['model']['use_dependency_gcn'] = False
    no_gcn['data']['use_dependency_parsing'] = False
    no_gcn['experiment']['name'] = 'A1_no_gcn'

    return [
        ('A1_full_model', 'Full model (with Dependency GCN)', full),
        ('A1_no_gcn',     'No GCN — aspect attention only',   no_gcn),
    ]


# ─── Ablation 2: Aspect-Aware Attention vs CLS Pooling ───────────────────────
def ablation_2_aspect_attention(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests whether aspect-specific attention outperforms plain [CLS] pooling.
    The 'no_attention' variant is implemented via a flag in model.py
    (use_aspect_attention=False → use CLS token instead).
    """
    attention = copy.deepcopy(base_config)
    attention['experiment']['name'] = 'A2_aspect_attention'

    cls_only = copy.deepcopy(base_config)
    cls_only['model']['use_aspect_attention'] = False
    cls_only['experiment']['name'] = 'A2_cls_pooling'

    return [
        ('A2_aspect_attention', 'Aspect-guided MHA attention',     attention),
        ('A2_cls_pooling',      'CLS token pooling (no attention)', cls_only),
    ]


# ─── Ablation 3: Loss Function ───────────────────────────────────────────────
def ablation_3_loss_function(base_config: dict) -> List[ExperimentSpec]:
    """
    Isolates each component of the Hybrid Loss to show their individual
    and combined contributions to handling class imbalance.
    """
    def make_cfg(name: str, weights: dict) -> dict:
        cfg = copy.deepcopy(base_config)
        cfg['training']['loss_weights'] = weights
        cfg['experiment']['name'] = name
        return cfg

    hybrid    = make_cfg('A3_hybrid_loss',
                         {'focal': 1.0, 'cb': 0.5, 'dice': 0.3})
    focal_only  = make_cfg('A3_focal_only',
                           {'focal': 1.0, 'cb': 0.0, 'dice': 0.0})
    cb_only     = make_cfg('A3_cb_only',
                           {'focal': 0.0, 'cb': 1.0, 'dice': 0.0})
    dice_only   = make_cfg('A3_dice_only',
                           {'focal': 0.0, 'cb': 0.0, 'dice': 1.0})

    # CE loss — uses CrossEntropyLossWrapper (flag picked up by experiment_runner)
    ce = copy.deepcopy(base_config)
    ce['training']['use_ce_loss'] = True
    ce['experiment']['name'] = 'A3_ce_loss'

    return [
        ('A3_hybrid_loss',  'Hybrid Loss (Focal + CB + Dice)',    hybrid),
        ('A3_focal_only',   'Focal Loss only',                    focal_only),
        ('A3_cb_only',      'Class-Balanced Loss only',           cb_only),
        ('A3_dice_only',    'Dice Loss only',                     dice_only),
        ('A3_ce_loss',      'Cross-Entropy Loss (no imbalance handling)', ce),
    ]


# ─── Ablation 4: Data Augmentation ───────────────────────────────────────────
def ablation_4_augmentation(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests the impact of LLM-generated synthetic data on model performance,
    especially on rare aspect classes (price-neg, price-neu, packing-neu).
    """
    with_aug = copy.deepcopy(base_config)
    with_aug['data']['train_path'] = 'data/splits/train_augmented.csv'
    with_aug['experiment']['name'] = 'A4_with_augmentation'

    no_aug = copy.deepcopy(base_config)
    no_aug['data']['train_path'] = 'data/splits/train.csv'
    no_aug['experiment']['name'] = 'A4_no_augmentation'

    return [
        ('A4_with_augmentation', 'With LLM synthetic augmentation (10,050 samples)', with_aug),
        ('A4_no_augmentation',   'Without augmentation (9,240 samples)',              no_aug),
    ]


# ─── Ablation 5: Aspect-Specific vs Shared Classifier Head ───────────────────
def ablation_5_classifier_head(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests whether 7 dedicated classifier heads per aspect outperform
    a single shared head for all aspects.
    The flag 'use_shared_classifier' is picked up by a modified create_model().
    """
    aspect_specific = copy.deepcopy(base_config)
    aspect_specific['model']['use_shared_classifier'] = False
    aspect_specific['experiment']['name'] = 'A5_aspect_specific_heads'

    shared = copy.deepcopy(base_config)
    shared['model']['use_shared_classifier'] = True
    shared['experiment']['name'] = 'A5_shared_head'

    return [
        ('A5_aspect_specific_heads', '7 aspect-specific classifier heads', aspect_specific),
        ('A5_shared_head',           'Single shared classifier head',       shared),
    ]


# ─── Ablation 6: Text Preprocessing ─────────────────────────────────────────
def ablation_6_preprocessing(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests the impact of the cleaning pipeline.
    'no_preprocessing' uses the original raw text column — set via a special flag
    picked up in data_utils.py to skip clean_text_for_inference().
    This requires a separate unprocessed data split or using 'raw_data' path.
    """
    with_preprocess = copy.deepcopy(base_config)
    with_preprocess['experiment']['name'] = 'A6_with_preprocessing'

    no_preprocess = copy.deepcopy(base_config)
    # train.csv contains the cleaned text in 'data' column.
    # For the no-preprocess ablation we point to the raw original data split
    # (which must be created via a separate extraction step — see experiment_runner)
    no_preprocess['data']['skip_inference_cleaning'] = True
    no_preprocess['data']['train_path'] = 'data/splits/train_raw.csv'
    no_preprocess['data']['val_path']   = 'data/splits/val_raw.csv'
    no_preprocess['data']['test_path']  = 'data/splits/test_raw.csv'
    no_preprocess['experiment']['name'] = 'A6_no_preprocessing'

    return [
        ('A6_with_preprocessing', 'With text cleaning pipeline', with_preprocess),
        ('A6_no_preprocessing',   'Without cleaning (raw text)',  no_preprocess),
    ]


# ─── Summary ──────────────────────────────────────────────────────────────────
def get_all_baseline_specs(base_config: dict) -> List[ExperimentSpec]:
    """Returns baseline experiment specs."""
    specs = []

    # B1: Plain RoBERTa
    b1 = copy.deepcopy(base_config)
    b1['experiment']['name'] = 'B1_plain_roberta'
    specs.append(('B1_plain_roberta',
                  'Plain RoBERTa — [CLS] head, no aspect awareness, CE loss',
                  b1))

    # B2: Full architecture + CE loss
    b2 = copy.deepcopy(base_config)
    b2['training']['use_ce_loss'] = True
    b2['experiment']['name'] = 'B2_roberta_ce'
    specs.append(('B2_roberta_ce',
                  'RoBERTa + Aspect Attention + GCN + CrossEntropy (no hybrid loss)',
                  b2))

    # B3: BERT-base
    b3 = copy.deepcopy(base_config)
    b3['model']['roberta_model'] = 'bert-base-uncased'
    b3['experiment']['name'] = 'B3_bert_base'
    specs.append(('B3_bert_base',
                  'BERT-base-uncased — [CLS] head, aspect-unaware, CE loss',
                  b3))

    # B4: TF-IDF + SVM (handled separately — no GPU, no config)
    b4 = copy.deepcopy(base_config)
    b4['experiment']['name'] = 'B4_tfidf_svm'
    b4['_baseline_type'] = 'tfidf_svm'
    specs.append(('B4_tfidf_svm',
                  'Classical TF-IDF + LinearSVC — no deep learning',
                  b4))

    return specs


def print_experiment_plan(base_config: dict):
    """Print a summary of all planned experiments."""
    ablations = get_all_ablation_specs(base_config)
    baselines = get_all_baseline_specs(base_config)

    print("=" * 70)
    print("EXPERIMENT PLAN")
    print("=" * 70)

    print("\n-- Baseline Comparisons " + "-" * 46)
    for exp_id, desc, _ in baselines:
        print(f"  [{exp_id}]  {desc}")

    print("\n-- Ablation Studies " + "-" * 50)
    for exp_id, desc, _ in ablations:
        print(f"  [{exp_id}]  {desc}")

    total = len(ablations) + len(baselines)
    print(f"\nTotal experiments: {total}")
    print("=" * 70)
