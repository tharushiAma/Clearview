"""
results_analyzer.py
Loads all_results.json and generates:
  1. Baseline comparison table (Markdown + LaTeX)
  2. Ablation study tables (Markdown + LaTeX)
  3. Per-class F1 comparison for rare classes (price-neg, packing-neu)
  4. Summary bar chart per experiment group

Usage:
    python src/experiments/results_analyzer.py --results results/experiments/all_results.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

ASPECTS = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
CLASSES = ['negative', 'neutral', 'positive']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _f(val, decimals=4) -> str:
    """Format a float or return '—' if missing."""
    if val is None or val == '':
        return '—'
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return '—'


def _pct(val) -> str:
    if val is None:
        return '—'
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return '—'


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Overall Metric Comparison
# ─────────────────────────────────────────────────────────────────────────────
def overall_comparison_table(results: dict, exp_ids: list, title: str) -> str:
    """Generates a Markdown table comparing overall metrics across experiments."""
    rows = []

    for exp_id in exp_ids:
        if exp_id not in results:
            continue
        r = results[exp_id]
        if r['status'] != 'done' or not r.get('overall'):
            rows.append((exp_id, r.get('description', ''), '—', '—', '—', '—', '—'))
            continue

        o = r['overall']
        rows.append((
            exp_id,
            r.get('description', exp_id),
            _f(o.get('accuracy')),
            _f(o.get('macro_f1')),
            _f(o.get('weighted_f1')),
            _f(o.get('mcc')),
            f"{r.get('duration_mins', '—')} min" if r.get('duration_mins') else '—',
        ))

    lines = [f"## {title}\n"]
    header = "| Experiment | Accuracy | Macro-F1 | Weighted-F1 | MCC | Time |"
    sep    = "|-----------|----------|----------|-------------|-----|------|"
    lines.append(header)
    lines.append(sep)
    for _, desc, acc, mf1, wf1, mcc, t in rows:
        lines.append(f"| {desc} | {acc} | **{mf1}** | {wf1} | {mcc} | {t} |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Per-Aspect Macro-F1
# ─────────────────────────────────────────────────────────────────────────────
def per_aspect_table(results: dict, exp_ids: list, title: str) -> str:
    """Generates a per-aspect Macro-F1 comparison table."""
    lines = [f"## {title} — Per-Aspect Macro-F1\n"]
    header = "| Experiment | " + " | ".join(ASPECTS) + " | Avg |"
    sep    = "|---|" + "---|" * (len(ASPECTS) + 1)
    lines.append(header)
    lines.append(sep)

    for exp_id in exp_ids:
        if exp_id not in results:
            continue
        r = results[exp_id]
        desc = r.get('description', exp_id)
        per_aspect = r.get('per_aspect', {})

        f1_vals = []
        cells = []
        for asp in ASPECTS:
            m = per_aspect.get(asp, {})
            f1 = m.get('macro_f1')
            cells.append(_f(f1))
            if f1 is not None:
                f1_vals.append(float(f1))

        avg = _f(np.mean(f1_vals)) if f1_vals else '—'
        lines.append(f"| {desc} | " + " | ".join(cells) + f" | {avg} |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Rare-Class F1 (Price-neg, Price-neu, Packing-neu)
# ─────────────────────────────────────────────────────────────────────────────
def rare_class_table(results: dict, exp_ids: list, title: str) -> str:
    """
    Shows per-class F1 for the rarest aspect-class combinations.
    label_idx: negative=0, neutral=1, positive=2
    """
    rare_cols = [
        ('price',   'negative', 0),
        ('price',   'neutral',  1),
        ('packing', 'neutral',  1),
        ('smell',   'neutral',  1),
    ]

    lines = [f"## {title} — Rare Class F1\n"]
    col_headers = " | ".join([f"{asp}-{lbl}" for asp, lbl, _ in rare_cols])
    header = f"| Experiment | {col_headers} |"
    sep    = "|---|" + "---|" * len(rare_cols)
    lines.append(header)
    lines.append(sep)

    for exp_id in exp_ids:
        if exp_id not in results:
            continue
        r = results[exp_id]
        desc = r.get('description', exp_id)
        per_aspect = r.get('per_aspect', {})

        cells = []
        for asp, lbl, idx in rare_cols:
            m = per_aspect.get(asp, {})
            f1_list = m.get('per_class_f1', [])
            if idx < len(f1_list):
                cells.append(_f(f1_list[idx]))
            else:
                cells.append('—')

        lines.append(f"| {desc} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_latex_table(results: dict, exp_ids: list, title: str) -> str:
    """LaTeX table for embedding in the FYP report."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        f"\\caption{{{title}}}",
        "\\label{tab:" + title.lower().replace(' ', '_')[:30] + "}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Model & Acc. & Macro-F1 & Weighted-F1 & MCC \\\\",
        "\\hline",
    ]
    for exp_id in exp_ids:
        if exp_id not in results:
            continue
        r = results[exp_id]
        desc = r.get('description', exp_id).replace('_', '\\_').replace('&', '\\&')
        o = r.get('overall', {})
        lines.append(
            f"{desc} & {_f(o.get('accuracy'))} & {_f(o.get('macro_f1'))} & "
            f"{_f(o.get('weighted_f1'))} & {_f(o.get('mcc'))} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_macro_f1_comparison(results: dict, exp_ids: list, title: str, save_path: str):
    """Bar chart of macro-F1 across experiments."""
    labels, values = [], []
    for exp_id in exp_ids:
        if exp_id not in results or results[exp_id]['status'] != 'done':
            continue
        f1 = results[exp_id].get('overall', {}).get('macro_f1')
        if f1 is not None:
            labels.append(results[exp_id].get('description', exp_id)[:30])
            values.append(float(f1))

    if not values:
        print(f"  No data for chart: {title}")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    colors = ['#4a90d9' if v < max(values) else '#2ecc71' for v in values]
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Macro-F1')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim([0, min(1.0, max(values) + 0.1)])
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Report Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(results: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Group experiments
    baseline_ids = [k for k in results if k.startswith('B')]
    ablation_ids = [k for k in results if k.startswith('A')]

    # Group ablations by study
    ablation_groups = {
        'A1': ('Ablation 1: GCN Component',            [k for k in ablation_ids if k.startswith('A1')]),
        'A2': ('Ablation 2: Aspect Attention',          [k for k in ablation_ids if k.startswith('A2')]),
        'A3': ('Ablation 3: Loss Function',             [k for k in ablation_ids if k.startswith('A3')]),
        'A4': ('Ablation 4: Data Augmentation',        [k for k in ablation_ids if k.startswith('A4')]),
        'A5': ('Ablation 5: Classifier Head',          [k for k in ablation_ids if k.startswith('A5')]),
        'A6': ('Ablation 6: Text Preprocessing',        [k for k in ablation_ids if k.startswith('A6')]),
    }

    report_lines = [
        "# Experiment Results Report",
        f"\nGenerated from {len(results)} experiments\n",
        "---",
    ]

    # ── Baseline Comparisons ──────────────────────────────────────────────
    if baseline_ids:
        report_lines.append("\n# Baseline Comparisons\n")
        report_lines.append(overall_comparison_table(results, baseline_ids, "Overall Metrics"))
        report_lines.append("\n")
        report_lines.append(per_aspect_table(results, baseline_ids, "Baseline Comparison"))
        report_lines.append("\n")
        report_lines.append(rare_class_table(results, baseline_ids, "Baseline Comparison"))

        plot_macro_f1_comparison(results, baseline_ids,
                                  "Baseline Comparison — Macro-F1",
                                  str(save_dir / 'baselines_macro_f1.png'))

        # LaTeX
        with open(save_dir / 'baselines_latex.tex', 'w') as f:
            f.write(generate_latex_table(results, baseline_ids, "Baseline Comparison"))
        print(f"  LaTeX table saved: {save_dir / 'baselines_latex.tex'}")

    # ── Ablation Studies ──────────────────────────────────────────────────
    report_lines.append("\n---\n# Ablation Studies\n")

    for group_key, (group_title, ids) in ablation_groups.items():
        if not ids:
            continue
        report_lines.append(f"\n### {group_title}\n")
        report_lines.append(overall_comparison_table(results, ids, group_title))
        report_lines.append("\n")
        report_lines.append(rare_class_table(results, ids, group_title))

        plot_macro_f1_comparison(results, ids,
                                  f"{group_title} — Macro-F1",
                                  str(save_dir / f'{group_key}_macro_f1.png'))

    # LaTeX for all ablations together
    with open(save_dir / 'ablations_latex.tex', 'w') as f:
        f.write(generate_latex_table(results, ablation_ids, "Ablation Study Results"))
    print(f"  LaTeX table saved: {save_dir / 'ablations_latex.tex'}")

    # ── Write Markdown Report ─────────────────────────────────────────────
    report_path = save_dir / 'experiment_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results',  default='results/experiments/all_results.json',
                        help='Path to all_results.json')
    parser.add_argument('--save_dir', default='results/experiments/analysis',
                        help='Directory to save analysis outputs')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run experiment_runner.py first to generate results.")
        return

    results  = load_results(args.results)
    save_dir = Path(args.save_dir)
    generate_report(results, save_dir)


if __name__ == '__main__':
    main()
