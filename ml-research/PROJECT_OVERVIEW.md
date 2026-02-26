# Project Files Overview

## Complete Multi-Aspect Sentiment Analysis System

This project provides a production-ready implementation for multi-aspect sentiment analysis with class imbalance handling and explainability.

---

## 📁 File Structure

```
cosmetic_sentiment_project/
│
├── 📄 README.md                    # Complete documentation
├── 📄 QUICKSTART.md               # Step-by-step guide
├── 📄 requirements.txt            # Python dependencies
├── 🔧 setup.sh                    # Automated setup script
│
├── 📂 configs/
│   └── config.yaml                # Main configuration file
│
├── 📂 models/
│   ├── model.py                   # RoBERTa + Dependency GCN architecture
│   └── losses.py                  # Focal, CB, Dice, Hybrid losses
│
├── 📂 utils/
│   ├── data_utils.py             # Dataset, DataLoader, Dependency Parser
│   └── metrics.py                # Evaluation metrics and visualization
│
├── 🚀 train.py                    # Main training script
├── 📊 evaluate.py                 # Evaluation script
├── 🔮 inference.py                # Inference and prediction
│
└── 📂 data/                       # (Create this - put your CSV files here)
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 🎯 Key Files Explained

### Configuration (configs/config.yaml)
**Purpose**: Central configuration for all experiments
**Contains**:
- Model architecture settings (RoBERTa variant, hidden dims, layers)
- Training hyperparameters (batch size, learning rate, epochs)
- Loss function parameters (focal gamma, CB beta per aspect)
- Data augmentation settings
- Explainability options
- Hardware settings (GPU, mixed precision)

**Edit this file to**:
- Change model size (roberta-base → roberta-large)
- Adjust batch size for your GPU
- Enable/disable GCN or dependency parsing
- Tune loss functions for better minority class performance

---

### Model Architecture (models/model.py)
**Purpose**: Complete model implementation
**Contains**:
- `AspectAwareRoBERTa`: RoBERTa with aspect-specific attention
- `AspectOrientedDepGCN`: Dependency GCN with aspect gating
- `MultiAspectSentimentModel`: Complete integrated model

**Key Features**:
- Aspect embeddings (learnable 768-dim vectors)
- Multi-head attention (8 heads) for aspect-text interaction
- GCN message passing on dependency trees
- Aspect-specific classification heads

**When to modify**:
- Add more GCN layers for complex dependencies
- Change attention mechanism (e.g., add self-attention)
- Experiment with different pooling strategies

---

### Loss Functions (models/losses.py)
**Purpose**: Handle severe class imbalance
**Contains**:
- `FocalLoss`: Focus on hard examples
- `ClassBalancedLoss`: Reweight by effective samples
- `DiceLoss`: Directly optimize F1-score
- `HybridLoss`: Combine all three losses
- `AspectSpecificLossManager`: Auto-configure per aspect

**Key Parameters**:
- Focal gamma: 2.0 (moderate) to 4.0 (extreme imbalance)
- CB beta: 0.999 (moderate) to 0.9999 (extreme)
- Loss weights: {focal: 1.0, cb: 0.5, dice: 0.3}

**When to modify**:
- Poor minority class recall → Increase focal gamma
- Model overfitting → Reduce loss weights
- Need custom weighting → Modify HybridLoss

---

### Data Utilities (utils/data_utils.py)
**Purpose**: Load and preprocess data
**Contains**:
- `CosmenticReviewDataset`: Basic dataset
- `DependencyParsingDataset`: With syntactic trees
- `DependencyParser`: Extract dependency graphs
- `collate_fn_with_dependencies`: Batch processing
- `create_dataloaders`: Factory function

**Handles**:
- Missing aspect labels (sparse annotation)
- Text tokenization (RoBERTa tokenizer)
- Dependency tree extraction (spaCy)
- Edge index creation for GCN

**When to modify**:
- Add data augmentation in __getitem__
- Implement custom collate_fn for special batching
- Add pre-processing steps (cleaning, normalization)

---

### Evaluation Metrics (utils/metrics.py)
**Purpose**: Comprehensive evaluation
**Contains**:
- `AspectSentimentEvaluator`: Per-aspect and overall metrics
- `ErrorAnalyzer`: Misclassification analysis
- Confusion matrix plotting
- LaTeX table generation

**Computes**:
- Accuracy, Precision, Recall, F1 (macro, weighted, per-class)
- Matthews Correlation Coefficient (MCC)
- Confusion matrices
- Error distribution analysis

**When to modify**:
- Add custom metrics (e.g., Cohen's Kappa)
- Implement statistical significance tests
- Add visualization methods

---

### Training Script (train.py)
**Purpose**: Main training loop
**Contains**:
- `Trainer` class: Complete training pipeline
- Mixed precision training (FP16)
- Early stopping
- Checkpoint saving
- WandB integration

**Features**:
- Automatic loss selection per aspect
- Gradient clipping
- Learning rate scheduling (linear warmup)
- Validation during training
- Best model saving

**When to modify**:
- Add gradient accumulation for larger effective batch
- Implement curriculum learning
- Add custom callbacks

---

### Evaluation Script (evaluate.py)
**Purpose**: Evaluate trained models
**Produces**:
- Detailed per-aspect metrics
- Confusion matrices (PNG)
- Error analysis (CSV)
- Predictions (CSV)
- LaTeX tables for thesis

**Usage**:
```bash
python evaluate.py \
    --checkpoint results/best_model.pt \
    --data data/test.csv \
    --output-dir results/evaluation
```

---

### Inference Script (inference.py)
**Purpose**: Use trained model for predictions
**Contains**:
- `SentimentPredictor`: Easy-to-use prediction class
- Interactive mode
- Attention visualization
- Batch prediction

**Usage**:
```bash
# Single prediction
python inference.py \
    --checkpoint results/best_model.pt \
    --text "The lipstick color is beautiful but smells bad" \
    --aspect "smell"

# Interactive mode
python inference.py --checkpoint results/best_model.pt
```

---

## 🚀 Typical Workflow

### 1. Initial Setup
```bash
./setup.sh
source venv/bin/activate
cp /path/to/data/*.csv data/
```

### 2. Quick Test (Verify Everything Works)
Edit `configs/config.yaml`:
```yaml
training:
  num_epochs: 2  # Just to test
  batch_size: 8
model:
  use_dependency_gcn: false  # Faster
```

Run:
```bash
python train.py --config configs/config.yaml
```

Should complete in ~10-20 minutes on GPU.

### 3. Full Training
Reset config to:
```yaml
training:
  num_epochs: 30
  batch_size: 16
model:
  use_dependency_gcn: true
```

Run:
```bash
python train.py --config configs/config.yaml
```

Will take 2-4 hours on GPU.

### 4. Evaluation
```bash
python evaluate.py \
    --checkpoint results/cosmetic_sentiment_v1/best_model.pt \
    --data data/test.csv
```

### 5. Generate Thesis Materials
Results will be in `results/cosmetic_sentiment_v1/evaluation_results/`:
- `metrics.json`: All numerical results
- `confusion_matrices.png`: Visual results
- `latex_table.tex`: Copy-paste into thesis
- `predictions.csv`: Detailed predictions
- `error_analysis.csv`: Misclassifications

---

## 🔧 Customization Guide

### Change Model Size
```yaml
# In config.yaml
model:
  roberta_model: "roberta-large"  # Larger, slower, better
  # OR
  roberta_model: "distilroberta-base"  # Smaller, faster, good
```

### Adjust for Your GPU Memory
```yaml
training:
  batch_size: 4  # Reduce if OOM
hardware:
  mixed_precision: true  # Enable FP16
```

### Focus on Specific Aspects
```yaml
# In config.yaml - increase loss weights for important aspects
training:
  focal_gamma:
    smell: 4.0  # Really focus on smell minority classes
    price: 3.5
```

### Add Custom Augmentation
```python
# In utils/data_utils.py
def augment_text(text, aspect, label):
    if label == 'negative':
        # Add your augmentation logic
        return augmented_text
    return text
```

---

## 📊 Understanding Results

### Good Results:
```
Overall Macro-F1: 0.75+
Price/Packing Macro-F1: 0.55+ (they're hard!)
Smell/Texture/Colour: 0.75-0.85
All minority class recall > 0.50
```

### Signs of Issues:

1. **Low Minority Class Recall (<0.30)**
   - Increase focal gamma
   - Add more augmentation
   - Check class weights

2. **High Training Loss, Low Val Loss**
   - Underfitting
   - Increase model capacity
   - Train longer

3. **Low Training Loss, High Val Loss**
   - Overfitting
   - Reduce model size
   - Add more dropout
   - Use data augmentation

4. **Unstable Training (Loss Spikes)**
   - Reduce learning rate
   - Increase warmup steps
   - Check for bad samples

---

## 🎓 For Your Thesis

### Required Experiments:

1. **Ablation Study**:
   - Model without GCN
   - Model without aspect attention
   - Different loss functions
   - With/without augmentation

2. **Baseline Comparisons**:
   - Simple RoBERTa fine-tuning
   - BiLSTM
   - Traditional ML (SVM)
   - MAFESA (from paper)

3. **Analysis**:
   - Per-aspect performance
   - Confusion matrices
   - Error analysis
   - Attention visualization examples

### Thesis Sections:

1. **Methodology**: Use code comments as explanation
2. **Experiments**: Use results from `evaluation_results/`
3. **Results**: Use LaTeX tables generated
4. **Discussion**: Use error analysis insights
5. **Appendix**: Include code snippets

---

## 💡 Tips for Best Results

1. **Start Simple**: Train without GCN first
2. **Monitor Closely**: Use WandB or TensorBoard
3. **Save Everything**: You never know what you'll need
4. **Test Incrementally**: Don't change everything at once
5. **Document Changes**: Keep a log of experiments
6. **Focus on F1**: Accuracy misleads on imbalanced data
7. **Validate Claims**: Run multiple seeds, report std dev
8. **Be Honest**: Report what didn't work too

---

## 🆘 Getting Help

1. **Check error messages**: They're usually informative
2. **Review config**: Most issues are configuration
3. **Test components**: Run individual files with `__main__`
4. **Reduce complexity**: Disable features to isolate issues
5. **Check GitHub issues**: Others may have same problem

---

## ✅ Pre-submission Checklist

Before submitting your thesis:

- [ ] Run final experiments with 3 different seeds
- [ ] Generate all figures and tables
- [ ] Perform statistical significance tests
- [ ] Complete ablation studies
- [ ] Write clear limitations section
- [ ] Include code in appendix or GitHub link
- [ ] Verify all numbers in thesis match results
- [ ] Have someone else review your code
- [ ] Create GitHub repository with clear README
- [ ] Add requirements.txt and setup instructions

---

## 🎯 Expected Timeline

| Task | Time | Notes |
|------|------|-------|
| Setup & data prep | 1 day | First time |
| Initial testing | 0.5 day | Quick experiments |
| Baseline training | 1 day | Simple models |
| Main model training | 2-3 days | Multiple configs |
| Ablation studies | 2 days | 5-7 experiments |
| Analysis & visualization | 1 day | Figures for thesis |
| Writing methodology | 2-3 days | Code → text |
| Writing results | 2 days | Numbers → insights |
| Review & revision | 2-3 days | Polish everything |
| **Total** | **~2 weeks** | Serious research work |

---

**Remember**: Good research takes time. Don't rush. Be thorough. Be honest. Good luck! 🚀
