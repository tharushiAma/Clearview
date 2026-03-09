"""
Data Preprocessing and Stratified Splitting Script
For: Class balanced aspect base mixed sentiment resolution with XAI

This script:
1. Loads translated data
2. Applies deep text cleaning pipeline:
   a) HTML tag & entity removal
   b) URL / email scrubbing
   c) Garbled / keyboard-spam detection & removal
   d) Translation artifact normalisation (Vietnamese-origin noise)
3. Handles missing values
4. Processes aspect sentiment labels
5. Creates stratified train/val/test splits (70%/15%/15%)
6. Analyzes and documents class imbalance
"""

import re
import unicodedata
import html
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import json
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH   = "data/raw/full_data_en.csv"
OUTPUT_DIR  = "data/splits"
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# Text columns that should be cleaned (model reads "data" column directly)
TEXT_COLUMNS = ["data"]

# Aspect columns
ASPECT_COLUMNS = [
    "stayingpower", "texture", "smell", "price",
     "colour", "shipping", "packing",
]

# ── Garbled-text detection parameters ─────────────────────────────────────────
# A token is considered garbled when it looks like keyboard spam
GARBLED_MIN_LENGTH       = 6     # only inspect tokens >= this length
GARBLED_CONSONANT_RATIO  = 0.82  # if consonant proportion exceeds this → garbled
GARBLED_REPEAT_RATIO     = 0.60  # if a single char makes up this share → garbled
# A whole sentence is considered spam when garbled-token ratio exceeds:
SPAM_TOKEN_RATIO         = 0.40

# ── Vietnamese translation-artifact patterns ───────────────────────────────────
# Common literal phrases that survive vi→en machine translation unchanged or
# turn into semantically empty filler:
TRANSLATION_ARTIFACTS = [
    # Filler phrases
    r"\bthe product is\b",
    r"\bthe goods\b",
    r"\bgoods received\b",
    r"\bpackaging is\b",
    r"\bthe seller\b(?= (is|sent|ships))",
    r"\border received\b",
    r"\bfast delivery\b\.?\s*$",   # standalone at end
    r"\bwill buy again\b\.?\s*$",
    r"\bgood product\b\.?\s*$",
    # Emoji + punctuation duplicates produced by some translators
    r"\.{3,}",                     # excessive ellipsis → single …
    r"!{2,}",                      # multiple exclamation → single
    r"\?{2,}",                     # multiple question marks → single
    # Zero-width / invisible Unicode
    r"[\u200b-\u200f\u202a-\u202e\ufeff]",
]

# Pre-compile for speed
_ARTIFACT_RE = [re.compile(p, re.IGNORECASE) for p in TRANSLATION_ARTIFACTS]

# ── HTML / URL / Email patterns ────────────────────────────────────────────────
_HTML_TAG_RE     = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE  = re.compile(r"&(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]+);")
_URL_RE          = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+", re.IGNORECASE
)
_EMAIL_RE        = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", re.IGNORECASE)

# Consonant set for garbled detection (excluding 'y' which is often vowel-like)
_CONSONANTS = set("bcdfghjklmnpqrstvwxz")


# ── Text Cleaning Functions ────────────────────────────────────────────────────

def remove_html(text: str) -> str:
    """
    1. Decode HTML entities (&amp; → &, &lt; → <, &#39; → ', etc.)
    2. Strip remaining HTML tags (<br>, <strong>, etc.)
    """
    text = html.unescape(text)                 # handles named + numeric entities
    text = _HTML_ENTITY_RE.sub(" ", text)      # catch any stragglers
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def remove_urls_and_emails(text: str) -> str:
    """Replace URLs and e-mail addresses with a single space."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return text


def _is_garbled_token(token: str) -> bool:
    """
    Heuristic: a token is 'garbled' (keyboard spam) when:
      - It is long enough to be worth checking
      - Its consonant ratio is abnormally high  OR
      - A single character dominates the token
    """
    t = token.lower()
    if len(t) < GARBLED_MIN_LENGTH:
        return False

    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False

    consonant_ratio = sum(1 for c in letters if c in _CONSONANTS) / len(letters)
    if consonant_ratio >= GARBLED_CONSONANT_RATIO:
        return True

    # Single-character repetition (e.g. "aaaaaaa", "hhhhhh")
    max_char_ratio = max(t.count(c) for c in set(t)) / len(t)
    if max_char_ratio >= GARBLED_REPEAT_RATIO:
        return True

    return False


def remove_garbled_text(text: str) -> str:
    """
    Remove individual garbled tokens.
    If the resulting sentence still has a large proportion of garbled content,
    mark the whole sentence for removal (returns empty string).
    
    Short reviews (< 5 tokens) are protected — only individual garbled tokens
    are stripped, the whole sentence is never blanked.
    """
    tokens = text.split()
    if not tokens:
        return text

    clean_tokens = [tok for tok in tokens if not _is_garbled_token(tok)]

    removed = len(tokens) - len(clean_tokens)
    # Only blank the entire sentence for longer texts with high spam ratio
    # Short reviews (e.g. "Beautiful hhhhhh") should keep the clean tokens
    if len(tokens) >= 5 and removed / len(tokens) >= SPAM_TOKEN_RATIO:
        # The whole sentence was basically spam
        return ""

    return " ".join(clean_tokens)


def fix_translation_artifacts(text: str) -> str:
    """
    Normalise or remove patterns that commonly survive vi→en MT pipelines.
    """
    # Collapse excessive punctuation first
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    # Remove zero-width and other invisible Unicode characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

    return text


def normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces / tabs / newlines into a single space."""
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_text(text) -> str:
    """
    Master cleaning pipeline applied to every review text column.
    Ordering matters:
      1. Validate input
      2. Unicode normalisation (NFC)
      3. HTML removal
      4. URL / e-mail removal
      5. Translation artifact normalisation
      6. Garbled-text removal
      7. Whitespace collapse
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode NFC normalisation (combines combining characters)
    text = unicodedata.normalize("NFC", text)

    # 2. HTML
    text = remove_html(text)

    # 3. URLs / emails
    text = remove_urls_and_emails(text)

    # 4. Translation artifacts
    text = fix_translation_artifacts(text)

    # 5. Garbled tokens / keyboard spam
    text = remove_garbled_text(text)

    # 6. Whitespace
    text = normalise_whitespace(text)

    return text


# ── Data Loading & Exploration ────────────────────────────────────────────────

def load_and_explore_data() -> pd.DataFrame:
    """Load translated data and perform initial exploration."""
    log.info("Loading translated data from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    log.info("Column names: %s", df.columns.tolist())
    log.info("Missing values:\n%s", df.isnull().sum().to_string())
    return df


# ── Cleaning Pipeline ─────────────────────────────────────────────────────────

def apply_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text cleaning pipeline to all text columns."""
    log.info("Applying cleaning pipeline to text columns: %s", TEXT_COLUMNS)

    stats = {}
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            log.warning("Column '%s' not found — skipping.", col)
            continue

        before_empty = df[col].isna().sum() + (df[col] == "").sum()

        df[col] = df[col].apply(clean_text)
        # After cleaning, empty strings → NaN for consistency
        df[col] = df[col].replace("", np.nan)

        after_empty = df[col].isna().sum()
        newly_empty = int(after_empty - before_empty)
        stats[col] = {
            "rows_emptied_by_cleaning": newly_empty,
            "total_missing_after": int(after_empty),
        }
        log.info(
            "  %-15s  newly emptied: %4d  |  total missing: %4d",
            col, newly_empty, after_empty,
        )

    return df


def log_cleaning_examples(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                           col: str = "data", n: int = 5):
    """Print before/after examples for a quick sanity check."""
    log.info("=== Cleaning examples (column: %s) ===", col)
    if col not in df_raw.columns:
        return
    sample = df_raw[col].dropna().head(n)
    for idx, raw_val in sample.items():
        clean_val = df_clean.loc[idx, col] if idx in df_clean.index else "N/A"
        print(f"\n  [BEFORE] {str(raw_val)[:150]}")
        print(f"  [AFTER ] {str(clean_val)[:150]}")


# ── Aspect Label Processing ────────────────────────────────────────────────────

def process_aspect_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise aspect sentiment labels."""
    log.info("Processing aspect sentiment labels …")

    sentiment_mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral":  "neutral",
        "pos":      "positive",
        "neg":      "negative",
        "neu":      "neutral",
    }

    for col in ASPECT_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].replace("", np.nan)
        df[col] = df[col].map(
            lambda x: sentiment_mapping.get(str(x).lower().strip(), x)
            if pd.notna(x) else x
        )

    return df


# ── Stratified Splitting ───────────────────────────────────────────────────────

def create_multi_label_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a composite key over all aspect labels for stratified splitting."""
    log.info("Building stratification key …")

    def combine_labels(row):
        labels = [
            f"{col}_{row[col]}"
            for col in ASPECT_COLUMNS
            if col in df.columns and pd.notna(row[col]) and row[col] != ""
        ]
        return "|".join(sorted(labels)) if labels else "no_labels"

    df["_stratify_key"] = df.apply(combine_labels, axis=1)

    key_counts = df["_stratify_key"].value_counts()
    rare_keys  = key_counts[key_counts < 2].index
    log.info(
        "Rare stratification keys (n=1): %d → grouped as 'other_rare_combination'",
        len(rare_keys),
    )
    df["_stratify_key"] = df["_stratify_key"].apply(
        lambda x: "other_rare_combination" if x in rare_keys else x
    )
    return df


def perform_stratified_split(df: pd.DataFrame):
    """
    Stratified train / val / test split with **rare-class minimum guarantee**.

    Problem: Pure proportional splitting (70/15/15) leaves classes like
    price-negative (21 total) with only 3 samples in val and 3 in test —
    too few for reliable F1 computation.

    Solution — Two-phase split:
      Phase 1: Identify rows with ANY rare aspect-class (< MIN_EVAL_SAMPLES
               in a 15% slice). Reserve them and split so that val/test each
               get at least MIN_EVAL_SAMPLES (or half the class, whichever
               is smaller).
      Phase 2: Do normal stratified split on the remaining majority rows.
      Merge: Combine the reserved + majority splits.
    """
    MIN_EVAL_SAMPLES = 8   # minimum samples per class in val AND test

    log.info("Performing stratified split (%.0f/%.0f/%.0f) with rare-class guarantee (min %d per eval set) …",
             TRAIN_SPLIT * 100, VAL_SPLIT * 100, TEST_SPLIT * 100, MIN_EVAL_SAMPLES)

    # ── Phase 1: Identify rows that belong to rare aspect-classes ─────────
    # A class is "rare" if its total count would give < MIN_EVAL_SAMPLES
    # under a proportional 15% allocation.
    rare_threshold = int(MIN_EVAL_SAMPLES / (TEST_SPLIT))  # need this many total for 15% to yield MIN_EVAL_SAMPLES
    log.info("  Rare-class threshold: aspect-class with < %d total samples gets reserved allocation", rare_threshold)

    rare_indices = set()
    rare_classes = {}  # {(aspect, label): [list of indices]}

    for aspect in ASPECT_COLUMNS:
        if aspect not in df.columns:
            continue
        vc = df[aspect].value_counts(dropna=True)
        for label, count in vc.items():
            if count < rare_threshold and count >= 3:  # need at least 3 to split
                mask = df[aspect] == label
                indices = df.index[mask].tolist()
                rare_classes[(aspect, label)] = indices
                rare_indices.update(indices)
                log.info("    Rare class: %s=%s (%d samples) → reserved for balanced allocation",
                         aspect, label, count)

    # Split rare rows: give val and test each their fair share
    rare_idx_list = sorted(rare_indices)
    rare_df = df.loc[rare_idx_list].copy()
    majority_df = df.drop(index=rare_idx_list).copy()

    if len(rare_df) > 0:
        log.info("  Reserving %d rows for rare-class guarantee, %d rows for normal split",
                 len(rare_df), len(majority_df))

        # For rare rows: allocate ~30% each to val and test, ~40% to train
        # This over-represents rare classes in val/test intentionally
        val_test_ratio = 0.30  # each of val & test gets 30% of rare rows
        train_rare_ratio = 1.0 - 2 * val_test_ratio

        # Shuffle rare rows before splitting
        rare_df = rare_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        n_rare = len(rare_df)
        n_val_rare = max(1, int(n_rare * val_test_ratio))
        n_test_rare = max(1, int(n_rare * val_test_ratio))
        n_train_rare = n_rare - n_val_rare - n_test_rare

        train_rare = rare_df.iloc[:n_train_rare]
        val_rare = rare_df.iloc[n_train_rare:n_train_rare + n_val_rare]
        test_rare = rare_df.iloc[n_train_rare + n_val_rare:]

        log.info("  Rare rows split → Train: %d  Val: %d  Test: %d",
                 len(train_rare), len(val_rare), len(test_rare))
    else:
        train_rare = pd.DataFrame(columns=df.columns)
        val_rare = pd.DataFrame(columns=df.columns)
        test_rare = pd.DataFrame(columns=df.columns)

    # ── Phase 2: Normal stratified split on majority rows ──────────────────
    # Re-create stratification keys for the majority subset
    majority_df = create_multi_label_target(majority_df)

    train_val_majority, test_majority = train_test_split(
        majority_df,
        test_size=TEST_SPLIT,
        stratify=majority_df["_stratify_key"],
        random_state=RANDOM_SEED,
    )

    val_size_adj = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
    train_majority, val_majority = train_test_split(
        train_val_majority,
        test_size=val_size_adj,
        stratify=train_val_majority["_stratify_key"],
        random_state=RANDOM_SEED,
    )

    # Drop stratification key from majority splits
    for split_df in (train_majority, val_majority, test_majority):
        split_df.drop("_stratify_key", axis=1, inplace=True, errors="ignore")

    # ── Merge rare + majority ─────────────────────────────────────────────
    train_df = pd.concat([train_majority, train_rare], ignore_index=True)
    val_df = pd.concat([val_majority, val_rare], ignore_index=True)
    test_df = pd.concat([test_majority, test_rare], ignore_index=True)

    # Shuffle each split so rare rows aren't clustered at the end
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Drop _stratify_key if it leaked through
    for split_df in (train_df, val_df, test_df):
        if "_stratify_key" in split_df.columns:
            split_df.drop("_stratify_key", axis=1, inplace=True)

    n = len(df)
    log.info(
        "Split → Train: %d (%.1f%%)  Val: %d (%.1f%%)  Test: %d (%.1f%%)",
        len(train_df), len(train_df) / n * 100,
        len(val_df),   len(val_df)   / n * 100,
        len(test_df),  len(test_df)  / n * 100,
    )

    # Log rare-class counts in val/test for verification
    log.info("Rare-class counts in eval sets:")
    for (aspect, label), indices in rare_classes.items():
        val_count = (val_df[aspect] == label).sum()
        test_count = (test_df[aspect] == label).sum()
        log.info("    %s=%s → Val: %d  Test: %d", aspect, label, val_count, test_count)

    return train_df, val_df, test_df


# ── Distribution Analysis ──────────────────────────────────────────────────────

def analyze_class_distribution(df: pd.DataFrame, dataset_name: str = "Full") -> dict:
    """Return per-aspect class counts & percentages."""
    log.info("Class distribution — %s dataset", dataset_name)
    distribution = {}
    total = len(df)

    for aspect in ASPECT_COLUMNS:
        if aspect not in df.columns:
            continue
        vc = df[aspect].value_counts(dropna=False)
        aspect_dist = {}
        for label, count in vc.items():
            pct = count / total * 100
            aspect_dist[str(label)] = {"count": int(count), "percentage": float(pct)}
        distribution[aspect] = aspect_dist
    return distribution


def identify_imbalanced_classes(train_df, val_df, test_df, threshold: float = 10.0) -> dict:
    """Identify rare classes in the training set."""
    log.info("Identifying imbalanced classes (threshold < %.1f%%) …", threshold)

    info = {
        "threshold_percentage": threshold,
        "rare_classes": {},
        "recommendations": [],
    }

    for aspect in ASPECT_COLUMNS:
        if aspect not in train_df.columns:
            continue
        vc    = train_df[aspect].value_counts(dropna=True)
        total = len(train_df)
        rare  = [
            {"label": str(lbl), "count_train": int(cnt),
             "percentage_train": float(cnt / total * 100)}
            for lbl, cnt in vc.items()
            if cnt / total * 100 < threshold
        ]
        if rare:
            info["rare_classes"][aspect] = rare
            log.warning("⚠  %s — rare classes: %s", aspect.upper(),
                        [r["label"] for r in rare])



    return info


# ── I/O Helpers ───────────────────────────────────────────────────────────────

def save_splits(train_df, val_df, test_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        split.to_csv(path, index=False)
        log.info("Saved %s → %s (%d rows)", name, path, len(split))


# ── Markdown Report ───────────────────────────────────────────────────────────

def create_class_imbalance_report(train_dist, val_dist, test_dist, imbalance_info) -> str:

    report = f"""# Class Imbalance Analysis

**Project**: Class balanced aspect base mixed sentiment resolution with XAI  
**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Cleaning Pipeline Applied

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | Unicode NFC normalisation | Unify combining characters |
| 2 | HTML tag & entity removal | Strip `<br>`, `&amp;`, `&#39;` etc. |
| 3 | URL / e-mail removal | Free up token budget |
| 4 | Translation artifact normalisation | Fix vi→en MT filler & punctuation |
| 5 | Garbled / keyboard-spam removal | Drop incoherent tokens |
| 6 | Whitespace collapse | Clean token boundaries |

## Dataset Split Distribution

| Set | Samples | % |
|-----|---------|---|
| Train | {train_dist.get('_total', 'N/A')} | 70% |
| Validation | {val_dist.get('_total', 'N/A')} | 15% |
| Test | {test_dist.get('_total', 'N/A')} | 15% |

## Aspect-wise Class Distribution

"""

    for aspect in ASPECT_COLUMNS:
        if aspect not in train_dist:
            continue
        report += f"\n### {aspect.upper()}\n\n"
        report += "| Class | Train Count (%) | Val Count (%) | Test Count (%) |\n"
        report += "|-------|----------------|---------------|----------------|\n"

        all_labels = (
            set(train_dist.get(aspect, {}).keys()) |
            set(val_dist.get(aspect, {}).keys())   |
            set(test_dist.get(aspect, {}).keys())
        )
        for label in sorted(all_labels):
            def _fmt(dist, a, lbl):
                d = dist.get(a, {}).get(lbl, {"count": 0, "percentage": 0.0})
                return f"{d['count']} ({d['percentage']:.2f}%)"
            report += (
                f"| {label} | {_fmt(train_dist, aspect, label)} | "
                f"{_fmt(val_dist, aspect, label)} | "
                f"{_fmt(test_dist, aspect, label)} |\n"
            )

    report += "\n## Imbalanced Classes Identified\n\n"
    report += f"**Threshold**: < {imbalance_info['threshold_percentage']}% in training set\n\n"

    if imbalance_info["rare_classes"]:
        for aspect, rare_list in imbalance_info["rare_classes"].items():
            report += f"### {aspect.upper()}\n\n"
            for rc in rare_list:
                report += (
                    f"- **{rc['label']}**: {rc['count_train']} samples "
                    f"({rc['percentage_train']:.2f}%)\n"
                )
            report += "\n"

        report += "## Recommendations\n\n"
        for i, rec in enumerate(imbalance_info["recommendations"], 1):
            report += f"{i}. {rec}\n"
    else:
        report += "*No severely imbalanced classes detected.*\n"

    report += """
## Stratification Strategy

Dataset split with **stratified sampling** over multi-label aspect-sentiment keys to ensure:
- Proportional representation of all aspect-sentiment combinations across every split
- Rare classes maintain the same percentage in train / val / test
- Garbled / empty rows are excluded before splitting so they do not dilute any split
"""
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("Data Preprocessing and Stratified Splitting")
    log.info("=" * 65)

    # 1. Load
    df = load_and_explore_data()
    df_raw = df.copy()   # keep original for before/after logging

    # 2. Clean text columns
    df = apply_cleaning_pipeline(df)
    log_cleaning_examples(df_raw, df, col=TEXT_COLUMNS[0] if TEXT_COLUMNS else "data")

    # 2b. Drop rows where review text is empty after cleaning
    before_drop = len(df)
    df = df.dropna(subset=["data"])
    dropped = before_drop - len(df)
    log.info("Dropped %d rows with empty 'data' after cleaning (%d remaining)", dropped, len(df))

    # 3. Standardise aspect labels
    df = process_aspect_labels(df)

    # 4. Stratification key
    df = create_multi_label_target(df)

    # 5. Split
    train_df, val_df, test_df = perform_stratified_split(df)

    # 6. Distribution analysis
    train_dist = analyze_class_distribution(train_df, "Training")
    val_dist   = analyze_class_distribution(val_df,   "Validation")
    test_dist  = analyze_class_distribution(test_df,  "Test")
    train_dist["_total"] = len(train_df)
    val_dist["_total"]   = len(val_df)
    test_dist["_total"]  = len(test_df)

    # 7. Imbalance detection
    imbalance_info = identify_imbalanced_classes(train_df, val_df, test_df)

    # 8. Save splits
    save_splits(train_df, val_df, test_df)

    # 9. Markdown report
    report = create_class_imbalance_report(train_dist, val_dist, test_dist, imbalance_info)
    report_path = "class_imbalance.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Saved class imbalance report → %s", report_path)

    # 10. JSON distribution dump
    dist_json = {
        "train": train_dist, "val": val_dist, "test": test_dist,
        "imbalance_info": imbalance_info,
    }
    json_path = os.path.join(OUTPUT_DIR, "class_distribution.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dist_json, f, indent=2)
    log.info("Saved distribution data → %s", json_path)

    log.info("=" * 65)
    log.info("PREPROCESSING COMPLETE!")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
