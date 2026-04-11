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
  A6  — Mixed Sentiment Resolution evaluation (GCN with/without MSR)
  A7  — Hybrid Loss weight fine-tuning

Redundancy Management:
  Includes 'validate_ablation' to identify any experiment identical to the 
  Full Model (A1), alerting the user before GPU time is wasted.
"""

import copy
from typing import Dict, List, Tuple


ExperimentSpec = Tuple[str, str, dict]  # (experiment_id, description, config_override)


def validate_ablation(exp_id: str, modified: dict, base: dict, keys: list = None):
    """
    Checks if an ablation variant is accidentally identical to the base_config
    (the Full Model). If so, prints a NOTE to the user.
    
    This function acts as an 'Early Warning' system during config generation.
    Enforcement (skipping) happens later in experiment_runner.py.

    Args:
        exp_id:   The unique ID of the experiment (e.g., 'A2_aspect_attention')
        modified: The deep-copied and potentially modified configuration dict
        base:     The base_config (A1_full_model reference) for comparison
        keys:     Optional list of key paths (e.g., ['data.train_path']) that are 
                  EXPECTED to differ. If they don't, a WARNING is issued.
    """
    import warnings
    import json

    def _get(d, dotted_key):
        """Helper to navigate nested dictionaries with dotted keys."""
        parts = dotted_key.split('.')
        for p in parts:
            d = d.get(p, {})
        return d

    def _canonical(cfg):
        """
        Creates a 'fingerprint' of a config by stripping out name-specific
        and evaluation-specific fields (like 'name' and 'evaluate_msr').
        This allows us to see if the core logic (architecture/data) is duplicate.
        """
        import copy
        c = copy.deepcopy(cfg)
        c.get('experiment', {}).pop('name', None)
        c.get('experiment', {}).pop('evaluate_msr', None)
        return json.dumps(c, sort_keys=True)

    # 1. Global Redundancy Check (Canonical comparison)
    # If the stripped-down configs match, they are effectively the same model.
    if _canonical(modified) == _canonical(base):
        if exp_id != 'A1_full_model':
            # This is an "Info" message to inform the user that training
            # is not required and results will be reused.
            print(f"[ablation_configs] NOTE: '{exp_id}' is identical to the "
                  f"base Full Model. The runner will automatically reuse A1 "
                  f"results for this row to save GPU time.")
        return

    # 2. Specific key checks (Validation for intended changes)
    # If the user specified that certain keys MUST change (e.g., a path),
    # we double-check that they actually did change.
    if keys:
        for key in keys:
            if _get(modified, key) == _get(base, key):
                warnings.warn(
                    f"\n[ablation_configs] WARNING: '{exp_id}' key '{key}' is "
                    f"identical to base_config. This ablation may not behave as intended or might be redundant.",
                    stacklevel=2
                )


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
    specs.extend(ablation_6_mixed_sentiment(base_config))
    return specs


# ─── Ablation 1: GCN Component ───────────────────────────────────────────────
def ablation_1_gcn(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests whether the Dependency GCN helps.
    Hypothesis: GCN captures syntactic relationships useful for mixed sentiment.
    """
    full = copy.deepcopy(base_config)
    full['experiment']['name'] = 'A1_full_model'
    full['experiment']['evaluate_msr'] = True  # Capture MSR metrics for the full model
    validate_ablation('A1_full_model', full, base_config)

    no_gcn = copy.deepcopy(base_config)
    no_gcn['model']['use_dependency_gcn'] = False
    no_gcn['data']['use_dependency_parsing'] = False
    no_gcn['experiment']['name'] = 'A1_no_gcn'
    no_gcn['experiment']['evaluate_msr'] = True  # Capture MSR metrics without GCN

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
    attention['experiment']['evaluate_msr'] = True  # Compare MSR with/without attention
    validate_ablation('A2_aspect_attention', attention, base_config)

    cls_only = copy.deepcopy(base_config)
    cls_only['model']['use_aspect_attention'] = False
    cls_only['experiment']['name'] = 'A2_cls_pooling'
    cls_only['experiment']['evaluate_msr'] = True  # Compare MSR with/without attention

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
    # Guard: A3_hybrid_loss must differ from base on loss_weights (dice component added)
    validate_ablation('A3_hybrid_loss', hybrid, base_config,
                      ['training.loss_weights'])

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
    validate_ablation('A4_with_augmentation', with_aug, base_config,
                      ['data.train_path'])

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
    validate_ablation('A5_aspect_specific_heads', aspect_specific, base_config)

    shared = copy.deepcopy(base_config)
    shared['model']['use_shared_classifier'] = True
    shared['experiment']['name'] = 'A5_shared_head'

    # Guard: create_model() defaults use_shared_classifier=False, so if base_config
    # doesn't set it (or sets it False), A5_aspect_specific_heads == A1_full_model.
    # In that case, A1_full_model IS the aspect-specific result — use it directly
    # in your ablation table instead of re-running.
    base_shared = base_config.get('model', {}).get('use_shared_classifier', False)
    if not base_shared:
        import warnings
        warnings.warn(
            "[ablation_configs] A5_aspect_specific_heads: base_config already uses "
            "use_shared_classifier=False (the default). A5_aspect_specific_heads is "
            "identical to A1_full_model. Use A1's result for the 'aspect-specific' "
            "row in your ablation table — do NOT re-run A5_aspect_specific_heads.",
            stacklevel=2,
        )

    return [
        ('A5_aspect_specific_heads', '7 aspect-specific classifier heads', aspect_specific),
        ('A5_shared_head',           'Single shared classifier head',       shared),
    ]


# ─── Ablation 6: Mixed Sentiment Resolution (MSR) Evaluation ────────────────
def ablation_6_mixed_sentiment(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests whether the Dependency GCN specifically improves Mixed Sentiment
    Resolution (MSR) — i.e., correctly separating conflicting sentiments
    across aspects within the same review.

    Hypothesis: The GCN's aspect-oriented gating allows the model to isolate
    aspect-relevant tokens, which is critical for MSR. Removing the GCN
    should degrade MSR accuracy more than overall accuracy.

    Both variants are evaluated with the dedicated MixedSentimentEvaluator
    (in addition to standard metrics) so that experiment_runner can compare:
      - Overall Macro-F1 drop with/without GCN
      - MSR review-level accuracy drop with/without GCN
    This pair of numbers proves the GCN's specific contribution to MSR.

    The 'evaluate_msr' flag in the config is picked up by experiment_runner.py
    to trigger MixedSentimentEvaluator after the standard test evaluation.
    """
    with_gcn = copy.deepcopy(base_config)
    with_gcn['experiment']['name']         = 'A6_msr_with_gcn'
    with_gcn['experiment']['evaluate_msr'] = True  # Signal runner to run MSR eval
    validate_ablation('A6_msr_with_gcn', with_gcn, base_config)

    no_gcn = copy.deepcopy(base_config)
    no_gcn['model']['use_dependency_gcn']    = False
    no_gcn['data']['use_dependency_parsing'] = False
    no_gcn['experiment']['name']             = 'A6_msr_no_gcn'
    no_gcn['experiment']['evaluate_msr']     = True

    return [
        ('A6_msr_with_gcn', 'MSR Eval: Full model + GCN (mixed sent resolution)', with_gcn),
        ('A6_msr_no_gcn',   'MSR Eval: No GCN (attention only, no dep parsing)',  no_gcn),
    ]


# ─── Ablation 7: Hybrid Loss Weights ─────────────────────────────────────────
def ablation_7_hybrid_weights(base_config: dict) -> List[ExperimentSpec]:
    """
    Tests different combinations of focal and class-balanced weights without dice loss.
    """
    def make_cfg(name: str, weights: dict) -> dict:
        cfg = copy.deepcopy(base_config)
        cfg['training']['loss_weights'] = weights
        cfg['experiment']['name'] = name
        return cfg

    cb_05 = make_cfg('A7_hybrid_cb_05', {'focal': 1.0, 'cb': 0.5, 'dice': 0.0})
    validate_ablation('A7_hybrid_cb_05', cb_05, base_config)
    
    cb_10 = make_cfg('A7_hybrid_cb_10', {'focal': 1.0, 'cb': 1.0, 'dice': 0.0})

    return [
        ('A7_hybrid_cb_05', 'Hybrid Loss (Focal 1.0 + CB 0.5 + Dice 0.0)', cb_05),
        ('A7_hybrid_cb_10', 'Hybrid Loss (Focal 1.0 + CB 1.0 + Dice 0.0)', cb_10),
    ]




# ─── Summary ──────────────────────────────────────────────────────────────────
def get_all_baseline_specs(base_config: dict) -> List[ExperimentSpec]:
    """Returns baseline experiment specs."""
    specs = []

    # B1: Plain RoBERTa
    b1 = copy.deepcopy(base_config)
    b1['experiment']['name'] = 'B1_plain_roberta'
    b1['experiment']['evaluate_msr'] = True
    b1['_baseline_type'] = 'plain_roberta'   # needed so redundancy checker doesn’t clone A1
    specs.append(('B1_plain_roberta',
                  'Plain RoBERTa — [CLS] head, no aspect awareness, CE loss',
                  b1))

    # B2: DistilBERTBaseline
    b2 = copy.deepcopy(base_config)
    b2['model']['roberta_model'] = 'distilbert-base-uncased'
    b2['experiment']['name'] = 'B2_distilbert_base'
    b2['experiment']['evaluate_msr'] = True
    specs.append(('B2_distilbert_base',
                  'DistilBERT-base-uncased — [CLS] head, aspect-unaware, CE loss',
                  b2))

    # B3: BERT-base
    b3 = copy.deepcopy(base_config)
    b3['model']['roberta_model'] = 'bert-base-uncased'
    b3['experiment']['name'] = 'B3_bert_base'
    b3['experiment']['evaluate_msr'] = True
    specs.append(('B3_bert_base',
                  'BERT-base-uncased — [CLS] head, aspect-unaware, CE loss',
                  b3))

    # B4: TF-IDF + SVM (handled separately — no GPU, no config)
    b4 = copy.deepcopy(base_config)
    b4['experiment']['name'] = 'B4_tfidf_svm'
    b4['experiment']['evaluate_msr'] = True
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
