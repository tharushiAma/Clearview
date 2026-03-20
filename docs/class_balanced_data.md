# Synthetic Data Augmentation — Notes

**Date:** 2026-02-11

The dataset had extreme class imbalance — price was 174:1 positive-to-negative, packing was 185:1. Standard cross-entropy on that basically means the model never learns to predict negative or neutral for those aspects.

I generated 923 synthetic reviews using an LLM targeting the worst-affected minority classes, then ran a 5-step preprocessing pipeline before merging them into training.

## What the preprocessing did

1. **Label standardization** — LLM output used mixed casing ("Positive", "POSITIVE"). Lowercased everything to match the real data format. Fixed 4,650 labels.

2. **Label validation** — checked all labels were valid (`positive`, `negative`, `neutral`, `nan`). No hallucinated labels found — the generator was well-prompted.

3. **Deduplication** — LLMs repeat themselves, especially for packing complaints. Removed 113 exact duplicates (12.2%). This was the main source of data loss.

4. **Test set leakage check** — hash-based exact match against the test set. Zero matches found. Synthetic text patterns are distinct enough from real reviews that fuzzy matching wasn't needed.

5. **Noise injection** — synthetic reviews are too clean and formal compared to real ones. Added slang, typos, extra punctuation to ~9% of them so the model doesn't learn to distinguish real vs synthetic by text style.

## Final numbers

| | Count |
| --- | --- |
| Raw synthetic generated | 923 |
| After deduplication | 810 (87.8% retained) |
| Original training samples | 9,267 |
| Combined training set | 10,077 |

## Imbalance before and after

| Aspect | Before | After |
| --- | --- | --- |
| Price | 174:1 | ~11:1 |
| Packing | 185:1 | ~12:1 |
| Smell | 18:1 | ~6:1 |

These are still imbalanced — that's why I also use Hybrid Loss (Focal + Class-Balanced) during training. The augmentation just gets them to a more learnable range.

## Key things to remember

- Validation and test sets use only real data — never add synthetic there
- The augmented file is `data/splits/train_augmented.csv`
- Config already points to it by default

## Per-aspect detail

### Price

Before: 174:1 (only 13 negative + 22 neutral samples in training). Added 429 synthetic reviews targeting negative/neutral price opinions. After: 11:1.

### Packing

Before: 185:1 (only 70 negative, 11 neutral). Had the highest duplicate rate (20.8%) — the LLM kept writing "The packaging was damaged when it arrived." in different ways. After dedup: 373 reviews added. After: 12:1.

### Smell

Before: 18:1, mostly positive. Neutral was the rarest class (92 samples). Targeted neutral smell opinions specifically. After: 6:1.

## Why only synthetic for train, not other strategies

I also tried weighted random sampling (handled via the `use_sampler` flag in training). The augmentation handles the pre-training distribution; the sampler adjusts batch composition during training; the hybrid loss corrects the remaining gradient signal. The three approaches complement each other.
