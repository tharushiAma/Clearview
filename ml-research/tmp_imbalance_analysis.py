import json

d = json.load(open('results/experiments/all_results.json'))

experiments = [
    ('A3_ce_loss',        'CE Loss (no imbalance handling)'),
    ('A3_focal_only',     'Focal Loss only'),
    ('A3_cb_only',        'CB Loss only'),
    ('A3_dice_only',      'Dice Loss only'),
    ('A3_hybrid_loss',    'Hybrid: Focal+CB+Dice (original)'),
    ('A7_hybrid_cb_05',   'Hybrid: Focal+CB 0.5 (no Dice) [NEW]'),
    ('A7_hybrid_cb_10',   'Hybrid: Focal+CB 1.0 (no Dice)'),
    ('A4_no_augmentation','Full model WITHOUT augmentation'),
    ('A4_with_augmentation','Full model WITH augmentation'),
]

print(f"{'Experiment':<42} | {'Macro-F1':^10} | {'Accuracy':^10} | {'price-neg':^10} | {'price-neu':^10} | {'pack-neu':^10} | {'smell-neu':^10}")
print('-' * 115)

for exp_id, label in experiments:
    r = d.get(exp_id, {})
    overall = r.get('overall', {})
    f1 = overall.get('macro_f1', None)
    acc = overall.get('accuracy', None)
    per_aspect = r.get('per_aspect', {})

    def rare_f1(aspect, cls_idx):
        asp_data = per_aspect.get(aspect, {})
        pcf1 = asp_data.get('per_class_f1', [])
        if len(pcf1) > cls_idx:
            return f'{pcf1[cls_idx]:.4f}'
        return '  N/A '

    price_neg = rare_f1('price', 0)
    price_neu = rare_f1('price', 1)
    pack_neu  = rare_f1('packing', 1)
    smell_neu = rare_f1('smell', 1)

    f1_str  = f'{f1:.4f}' if isinstance(f1, float) else '  N/A  '
    acc_str = f'{acc:.4f}' if isinstance(acc, float) else '  N/A  '

    print(f"{label:<42} | {f1_str:^10} | {acc_str:^10} | {price_neg:^10} | {price_neu:^10} | {pack_neu:^10} | {smell_neu:^10}")
