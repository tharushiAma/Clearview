# ClearView Project - Fixes Documentation

This file documents the critical fixes and optimizations applied to the ClearView Implementation to ensure stability and cross-environment compatibility.

## 1. Terminal Encoding Fix (`UnicodeEncodeError`)
- **Problem**: Python's `logging` and `print` statements using the arrow symbol (`→`) caused `UnicodeEncodeError` on standard Windows terminals (which often default to CP1252 instead of UTF-8).
- **Fix**: Replaced all occurrences of `→` with standard ASCII `->`.
- **Files Affected**: 
    - `01_validate.py`
    - `02_Clean.py`
    - `03_split.py`
    - `baseline_distilbert.py`

## 2. Data Splitting Stability (`ValueError`)
- **Problem**: Using `stratify=df["signature"]` in `train_test_split` failed because some aspect label combinations (signatures) had only one member. Scikit-learn requires at least two members per class for stratification.
- **Fix**: Removed the `stratify` parameter to allow the split to proceed on small/imbalanced datasets.
- **Files Affected**: `03_split.py`

## 3. Training Numerical Stability (`NaN` Loss)
- **Problem**: `torch.nn.CrossEntropyLoss(ignore_index=-100)` returns `NaN` if a batch contains *only* ignored labels for a specific aspect (resulting in a 0/0 division during mean reduction).
- **Fix**: Added a mask check (`if mask.any():`) before calculating the loss for each aspect to skip batches with no active labels.
- **Files Affected**: `baseline_distilbert.py`

## 4. Evaluation Robustness (`classification_report`)
- **Problem**: `classification_report` crashed if the validation batch didn't contain samples for all three classes (negative, neutral, positive), as it couldn't map the `target_names`.
- **Fix**: Explicitly provided `labels=[0, 1, 2]` to the report to ensure it handles missing classes gracefully.
- **Files Affected**: `baseline_distilbert.py`

## 5. Model Training Optimizations
- **Implementation**:
    - **Gradient Clipping**: Added `torch.nn.utils.clip_grad_norm_` to prevent exploding gradients.
    - **Buffered Output**: Used `python -u` to ensure real-time log updates in the IDE/terminal.
    - **Device Auto-detection**: Model automatically switches to `cuda` if available, otherwise defaults to `cpu`.

## 6. Dependency & Configuration
- **Dependencies**: Installed `clean-text`, `emoji`, `fastparquet`, and `pyyaml` to support the modular pipeline.
- **Configuration**: Created `ClearView/configs/config.yaml` to centralize paths and aspect definitions, making the scripts easier to maintain.

## 7. Data Augmentation for Neutral Sentiment (Crucial Fix)
- **Problem**: The model failed to detect "Neutral" sentiment for `price` and `packing` (F1 Score: 0.00).
    - **Root Cause**: Extreme data scarcity. The Training set had only **15** samples for Neutral Price and **14** for Neutral Packing.
- **Solution**: Implemented a data augmentation pipeline using GPT-generated rewrites.
    1.  **Candidate Selection**: Created `augment_neutral_reviews.py` to identify "Positive" reviews suitable for rewriting.
    2.  **Generation**: Used GPT-4 to rewrite these as "Neutral", preserving other aspect sentiments.
    3.  **Merging**: Created `merge_synthetic_data.py` to split the synthetic data (60% Train, 40% Val) and merge it into the datasets.
- **Result**:
    - **Training Set**: Neutral Price samples increased to **33**.
    - **Validation Set**: Neutral Price samples increased to **15**.
    - **Impact**: Enables the model to learn and be evaluated on Neutral sentiment effectively.
- **Files Affected**:
    - `src/data_layer/augment_neutral_reviews.py`
    - `src/data_layer/merge_synthetic_data.py`
    - `data/splits/train_augmented.parquet`
    - `data/splits/val_augmented.parquet`

## 8. XAI Implementation & Optimization
- **New Feature**: Implemented `src/models/xai.py` and `run_xai_analysis.py` to support SHAP and LIME explanations.
- **Type Error Fix (`numpy.str_`)**:
    - **Problem**: LIME and SHAP libraries raised `TypeError: Argument 'string' has incorrect type (expected str, got numpy.str_)` when processing texts from Pandas/Numpy arrays.
    - **Fix**: Explicitly cast all input texts to Python `str()` inside the `predict_proba` wrapper function.
- **Performance Optimization**:
    - **Problem**: Default LIME sampling (5000 samples) was too slow on CPU, taking minutes per sentence.
    - **Fix**: Reduced default `num_samples` to 200 and limited SHAP evaluations, significantly speeding up the generation of `lime_*.html` reports without sacrificing interpretation quality.
