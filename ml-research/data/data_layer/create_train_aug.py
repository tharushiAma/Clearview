"""
create_train_aug.py
Merge original (cleaned) training data with synthetic augmentation CSVs.
Produces:
  - data/splits/train_augmented.csv
  - augmentation_impact.md   (before / after class distribution report)

Usage:
    python data/data_layer/create_train_aug.py
    (run from the ml-research root directory)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths (relative to ml-research root)
SPLITS_DIR    = Path("data/splits")
AUGMENTED_DIR = Path("data/augmented")

# Aspect columns (must match model config)
ASPECT_COLUMNS = [
    "stayingpower", "texture", "smell", "price",
    "colour", "shipping", "packing",
]

LABELS = ["positive", "neutral", "negative"]


# ── Distribution helpers ──────────────────────────────────────────────────────

def get_class_counts(df: pd.DataFrame) -> dict:
    """Return {aspect: {label: count}} for all aspects."""
    dist = {}
    for aspect in ASPECT_COLUMNS:
        if aspect not in df.columns:
            continue
        vc = df[aspect].value_counts(dropna=True)
        dist[aspect] = {str(lbl): int(vc.get(lbl, 0)) for lbl in LABELS}
    return dist


def build_report(before: dict, after: dict,
                 n_original: int, n_synthetic: int, n_combined: int,
                 synthetic_files: list) -> str:
    """Generate augmentation_impact.md content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Data Augmentation — Class Imbalance Impact Report

**Project**: Class Imbalanced Multi-Aspect Mixed Sentiment Resolution with XAI  
**Generated**: {now}

## Dataset Overview

| Metric | Count |
|--------|-------|
| Original training samples | {n_original:,} |
| Synthetic samples added | {n_synthetic:,} |
| **Combined (augmented) training samples** | **{n_combined:,}** |

### Synthetic Data Sources

| File | Records |
|------|---------|
"""
    for name, count in synthetic_files:
        md += f"| {name} | {count:,} |\n"

    md += """
---

## Before vs After Augmentation — Per-Aspect Class Distribution

"""

    for aspect in ASPECT_COLUMNS:
        if aspect not in before:
            continue

        b = before[aspect]
        a = after[aspect]
        total_b = sum(b.values())
        total_a = sum(a.values())

        md += f"### {aspect.upper()}\n\n"
        md += "| Class | Before Count | Before % | After Count | After % | Δ Count | Δ % |\n"
        md += "|-------|-------------|----------|-------------|---------|---------|-----|\n"

        for lbl in LABELS:
            bc = b.get(lbl, 0)
            ac = a.get(lbl, 0)
            bp = (bc / total_b * 100) if total_b else 0
            ap = (ac / total_a * 100) if total_a else 0
            delta_c = ac - bc
            delta_p = ap - bp
            arrow = "↑" if delta_c > 0 else ("↓" if delta_c < 0 else "—")
            md += (
                f"| {lbl} | {bc:,} | {bp:.1f}% "
                f"| {ac:,} | {ap:.1f}% "
                f"| {arrow} {delta_c:+,} | {delta_p:+.1f}% |\n"
            )

        md += f"| **Total** | **{total_b:,}** | | **{total_a:,}** | | **+{total_a - total_b:,}** | |\n"
        md += "\n"

    # Summary section — which classes improved most
    md += "---\n\n## Summary of Improvements\n\n"
    md += "| Aspect | Class | Before Count | After Count | Samples Added | % Change |\n"
    md += "|--------|-------|-------------|-------------|---------------|----------|\n"

    for aspect in ASPECT_COLUMNS:
        if aspect not in before:
            continue
        b = before[aspect]
        a = after[aspect]
        for lbl in LABELS:
            bc = b.get(lbl, 0)
            ac = a.get(lbl, 0)
            delta = ac - bc
            if delta > 0:
                pct_change = (delta / bc * 100) if bc > 0 else float('inf')
                pct_str = f"+{pct_change:.1f}%" if pct_change != float('inf') else "new"
                md += f"| {aspect} | {lbl} | {bc:,} | {ac:,} | +{delta:,} | {pct_str} |\n"
                

    return md


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("Creating Augmented Training Set")
    log.info("=" * 65)

    # 1. Load original (cleaned) training data
    train_path = SPLITS_DIR / "train.csv"
    if not train_path.exists():
        log.error("Original training file not found: %s", train_path)
        log.error("Run preprocess_and_split.py first!")
        return

    train_df = pd.read_csv(train_path)
    log.info("Loaded original train: %d records", len(train_df))

    # Snapshot BEFORE distribution
    before_dist = get_class_counts(train_df)

    # 2. Load all synthetic CSVs from the augmented directory
    csv_files = sorted(AUGMENTED_DIR.glob("*.csv"))
    if not csv_files:
        log.warning("No augmented CSV files found in %s — nothing to merge.", AUGMENTED_DIR)
        return

    aug_dfs = []
    synthetic_file_info = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        aug_dfs.append(df)
        synthetic_file_info.append((csv_path.name, len(df)))
        log.info("  Loaded %s: %d records", csv_path.name, len(df))

    # 3. Concatenate and deduplicate
    synthetic_df = pd.concat(aug_dfs, ignore_index=True)
    log.info("Total synthetic records: %d", len(synthetic_df))

    # Internal deduplication (LLMs often repeat exact sentences)
    before = len(synthetic_df)
    synthetic_df = synthetic_df.drop_duplicates(subset=["data"])
    log.info("Removed %d internal synthetic duplicates → %d remain",
             before - len(synthetic_df), len(synthetic_df))

    # Remove synthetic rows that duplicate original training data
    synthetic_df = synthetic_df[~synthetic_df["data"].isin(train_df["data"])]
    log.info("After overlap check with original train: %d synthetic records", len(synthetic_df))

    # 4. Combine
    # Only keep columns that exist in both DataFrames
    common_cols = [c for c in train_df.columns if c in synthetic_df.columns]
    combined_df = pd.concat(
        [train_df[common_cols], synthetic_df[common_cols]],
        ignore_index=True,
    )
    n_original  = len(train_df)
    n_synthetic = len(synthetic_df)
    n_combined  = len(combined_df)
    log.info("Combined augmented train: %d records (%d original + %d synthetic)",
             n_combined, n_original, n_synthetic)

    # 5. Save augmented CSV
    out_path = SPLITS_DIR / "train_augmented.csv"
    combined_df.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    # 6. Snapshot AFTER distribution & generate report
    after_dist = get_class_counts(combined_df)

    report = build_report(
        before_dist, after_dist,
        n_original, n_synthetic, n_combined,
        synthetic_file_info,
    )
    report_path = Path("augmentation_impact.md")
    report_path.write_text(report, encoding="utf-8")
    log.info("Saved augmentation impact report → %s", report_path)

    log.info("=" * 65)
    log.info("AUGMENTATION COMPLETE!")
    log.info("=" * 65)


if __name__ == "__main__":
    main()

